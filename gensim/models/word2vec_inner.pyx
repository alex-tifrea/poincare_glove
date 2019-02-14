#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# Copyright (C) 2013 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import cython
import numpy as np
cimport numpy as np

from libc.math cimport exp
from libc.math cimport log
from libc.string cimport memset

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from libc.stdio cimport printf
from libc.math cimport acosh, sqrt, fabs, tanh, atanh, isnan, cos, sin, fmod, pow
from libc.stdlib cimport exit

REAL = np.float32

DEF MAX_SENTENCE_LEN = 10000
DEF MAX_EMB_SIZE = 1000
DEF EPS = 0.00001
DEF GRAD_CLAMP_THRESHOLD = 10000
DEF PI = 3.14159265359
DEF REG_WORD_INDEX_THRESHOLD = 189533
DEF TORUS_RADIUS_NORM_THRESHOLD = 10

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x (float)
cdef ssbmv_ptr ssbmv=<ssbmv_ptr>PyCObject_AsVoidPtr(fblas.ssbmv._cpointer) # y = alpha * A * x + beta * y (float)

cdef dcopy_ptr dcopy=<dcopy_ptr>PyCObject_AsVoidPtr(fblas.dcopy._cpointer)  # y = x (double)
cdef daxpy_ptr daxpy=<daxpy_ptr>PyCObject_AsVoidPtr(fblas.daxpy._cpointer)  # y += alpha * x (double)
cdef ddot_ptr ddot=<ddot_ptr>PyCObject_AsVoidPtr(fblas.ddot._cpointer)  # double = dot(x, y) (double)
cdef dnrm2_ptr dnrm2=<dnrm2_ptr>PyCObject_AsVoidPtr(fblas.dnrm2._cpointer)  # sqrt(x^2) (double)
cdef dscal_ptr dscal=<dscal_ptr>PyCObject_AsVoidPtr(fblas.dscal._cpointer) # x = alpha * x (double)
cdef dsbmv_ptr dsbmv=<dsbmv_ptr>PyCObject_AsVoidPtr(fblas.dsbmv._cpointer) # y = alpha * A * x + beta * y (double)

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
DEF MAX_TANH_ARG = 15

cdef REAL_t[EXP_TABLE_SIZE] EXP_TABLE
cdef REAL_t[EXP_TABLE_SIZE] LOG_TABLE

DEF COS_TABLE_SIZE = 10000
DEF MAX_COS = PI
cdef REAL_t[COS_TABLE_SIZE] COS_TABLE
cdef REAL_t[COS_TABLE_SIZE] SIN_TABLE

cdef int ZERO = 0
cdef float ZEROF = <REAL_t>0.0
cdef char L_CHAR = 'L'
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef DOUBLE_t ONED = <DOUBLE_t>1.0
cdef REAL_t MINUS_ONEF = <REAL_t>-1.0
cdef DOUBLE_t MINUS_ONED = <DOUBLE_t>-1.0
cdef DOUBLE_t EPS_ARR[MAX_EMB_SIZE]

cdef enum OptType:
    SGD=1  # regular SGD (like in vanilla Word2Vec)
    WSGD=2  # SGD where the lr is weighted by a function that depends on the frequency of the word whose embedding is being updated
    RSGD=3  # approximate Riemannian SGD i.e. uses a retraction to approximate the exponential map from the tangent space to the manifold
    WeightedRSGD=4  # approximate RSGD where the lr is weighted by a function that depends on the frequency of the word whose embedding is being updated
    FullRSGD=5  # full (exact) Riemannian SGD that uses the closed form of the exponential map in the Poincare ball
    WeightedFullRSGD=6  # full (exact) RSGD where the lr is weighted by a function that depends on the frequency of the word whose embedding is being updated
    RMSprop=7  # RMSprop (for RSGD it uses the approximate version, with the retraction)

cdef enum SimFuncType:
    DIST_SQ=1  # use -poincare_dist^2 as similarity during training
    COSH_DIST_SQ=2  # use -cosh(poincare_dist)^2 as similarity during training
    COSH_DIST=3  # use -cosh(poincare_dist) as similarity function during training
    LOG_DIST=4  # use -log(poincare_dist) as similarity during training
    LOG_DIST_SQ=5  # use -log(poincare_dist^2 + 1) as similarity during training
    EXP_DIST=6  # use -exp(poincare_dist) as similarity during training
    COSH_DIST_POW_K=7  # use -cosh(poincare_dist)^k as similarity during training

# for when fblas.sdot returns a double
cdef REAL_t our_dot_double(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>dsdot(N, X, incX, Y, incY)

# for when fblas.sdot returns a float
cdef REAL_t our_dot_float(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    return <REAL_t>sdot(N, X, incX, Y, incY)

# for when no blas available
cdef REAL_t our_dot_noblas(const int *N, const float *X, const int *incX, const float *Y, const int *incY) nogil:
    # not a true full dot()-implementation: just enough for our cases
    cdef int i
    cdef REAL_t a
    a = <REAL_t>0.0
    for i from 0 <= i < N[0] by 1:
        a += X[i] * Y[i]
    return a

# for when no blas available
cdef void our_saxpy_noblas(const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil:
    cdef int i
    for i from 0 <= i < N[0] by 1:
        Y[i * (incY[0])] = (alpha[0]) * X[i * (incX[0])] + Y[i * (incY[0])]


# Mitigate potential numerical errors
cdef double our_tanh(const double x) nogil:
    return tanh(min(max(x, -MAX_TANH_ARG), MAX_TANH_ARG))

cdef double* clamp(double *v, int size, double _min, double _max) nogil:
    for i in range(size):
        v[i] = max(_min, min(v[i], _max))

cdef void fast_sentence_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_epoch_training_loss_param) nogil:

    cdef long long a, b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f, g, f_dot, lprob

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelen):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _epoch_training_loss_param[0] = _epoch_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, &syn0[row1], &ONE, &syn1[row2], &ONE)

    our_saxpy(&size, &word_locks[word2_index], work, &ONE, &syn0[row1], &ONE)


# to support random draws from negative-sampling cum_table
cdef inline unsigned long long bisect_left(np.uint32_t *a, unsigned long long x, unsigned long long lo, unsigned long long hi) nogil:
    cdef unsigned long long mid
    while hi > lo:
        mid = (lo + hi) >> 1
        if a[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo

# this quick & dirty RNG apparently matches Java's (non-Secure)Random
# note this function side-effects next_random to set up the next number
cdef inline unsigned long long random_int32(unsigned long long *next_random) nogil:
    cdef unsigned long long this_random = next_random[0] >> 16
    next_random[0] = (next_random[0] * <unsigned long long>25214903917ULL + 11) & 281474976710655ULL
    return this_random


cdef inline void compute_sgns_loss(REAL_t logit, REAL_t *batch_loss, REAL_t *epoch_loss) nogil:
    cdef REAL_t log_e_logit

    if logit >= MAX_EXP:
        log_e_logit = 0.0
    elif logit <= -MAX_EXP:
        log_e_logit = logit
    else:
        log_e_logit = LOG_TABLE[<int>((logit + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    epoch_loss[0] = epoch_loss[0] - log_e_logit
    batch_loss[0] = batch_loss[0] - log_e_logit


cdef inline void compute_sgns_loss_dbl(DOUBLE_t logit, DOUBLE_t *batch_loss, DOUBLE_t *epoch_loss) nogil:
    cdef DOUBLE_t log_e_logit

    if logit >= MAX_EXP:
        log_e_logit = 0.0
    elif logit <= -MAX_EXP:
        log_e_logit = logit
    else:
        log_e_logit = <DOUBLE_t> LOG_TABLE[<int>((logit + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
    epoch_loss[0] = epoch_loss[0] - log_e_logit
    batch_loss[0] = batch_loss[0] - log_e_logit


# Update an embedding using the following formula:
#       emb = r * x + emb
# The update is performed atomically, such that the global storage (where all the embeddings are stored and accessed by
# all the concurrent threads) is only updated once.
# Parameters:
#       emb : array
#           pointer to the beginning of the embedding vector
#       r : scalar
#           see formula above
#       x : array
#           see formula above
#       size : scalar
#           embedding size
#       normalized : {0,1}
#           if 1, then normalize the embedding after the update
#       debug : {0,1}
#           if 1, then run in debug mode
cdef inline void update_embedding(REAL_t *emb, REAL_t r, REAL_t *x, const int size, const int normalized,
                                  const int debug) nogil:
    cdef REAL_t new_emb[size]
    cdef REAL_t norm_factor, emb_norm

    if normalized == 1:
        # If we need to normalize, we MUST compute the update and the scaling separately and then update the
        # embedding vector in the global storage. This is absolutely necessary because otherwise issues will emerge
        # due to the asynchrony of the system. Each thread, needs to update the global storage only once!
        scopy(&size, emb, &ONE, new_emb, &ONE)
        our_saxpy(&size, &r, x, &ONE, new_emb, &ONE)
        # Scale embedding to unit norm.
        norm_factor = 1.0 / snrm2(&size, new_emb, &ONE)
        sscal(&size, &norm_factor, new_emb, &ONE)
        # Update the embedding in the global storage.
        scopy(&size, new_emb, &ONE, emb, &ONE)

        if debug:
            emb_norm = snrm2(&size, new_emb, &ONE)
            if fabs(1.0 - emb_norm) > 0.0001:
                printf("Failed to normalize the norm of the embeddings. Expected norm 1.0, and got %f.\n", emb_norm)
                exit(-1)
    else:
        our_saxpy(&size, &r, x, &ONE, emb, &ONE)


# Update a Poincare embedding using the following formula (used for approximate RSGD using a simple retraction onto the Poincare ball x = x + v):
#       emb = r * x + emb
# The update is performed atomically, such that the global storage (where all the embeddings are stored and accessed by
# all the concurrent threads) is only updated once.
# Parameters:
#       emb : array
#           pointer to the beginning of the embedding vector
#       r : scalar
#           see formula above
#       x : array
#           see formula above
#       size : scalar
#           embedding size
# For Poincare embeddings, we always project the embeddings
# back to the radius-1 hypersphere, if they are outside it, after the update.
cdef inline void update_poincare_embedding(DOUBLE_t *emb, DOUBLE_t r, DOUBLE_t *x, const int size) nogil:
    cdef DOUBLE_t new_emb[size]
    cdef DOUBLE_t norm_factor, emb_norm, threshold = 1.0 - EPS

    memset(new_emb, 0, size * cython.sizeof(DOUBLE_t))

    dcopy(&size, emb, &ONE, new_emb, &ONE)
    daxpy(&size, &r, x, &ONE, new_emb, &ONE)

    # Project embedding back to the Poincare ball i.e. scale to unit norm if it is outside the ball.
    # Currently, we do no clipping after the normalization; instead, we normalize the vectors to 1-EPS, instead of 1.
    emb_norm = dnrm2(&size, new_emb, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        dscal(&size, &norm_factor, new_emb, &ONE)

    # Update the embedding in the global storage.
    dcopy(&size, new_emb, &ONE, emb, &ONE)


# Update a Poincare embedding using the full RSGD:
#       emb = exp_emb(lr * riemannian_grad)
# Parameters:
#       emb : array
#           pointer to the beginning of the embedding vector
#       lr : scalar
#           learning rate
#       grad : array
#           Riemannian gradient of loss wrt emb
#       size : scalar
#           embedding size
cdef inline void update_poincare_embedding_full_rsgd(DOUBLE_t *emb, DOUBLE_t lr, DOUBLE_t *grad, const int size) nogil:
    cdef DOUBLE_t new_emb[size]
    cdef DOUBLE_t emb_norm, v_norm, dot_emb_v, alpha, gamma, denominator, coef, threshold = 1.0 - EPS, norm_factor

    dcopy(&size, emb, &ONE, new_emb, &ONE)

    emb_norm = dnrm2(&size, new_emb, &ONE)
    v_norm = lr * dnrm2(&size, grad, &ONE) + 1e-15
    dot_emb_v = ddot(&size, new_emb, &ONE, grad, &ONE) * lr
    alpha = our_tanh(v_norm / (1 - emb_norm * emb_norm)) / v_norm
    gamma = alpha * alpha * v_norm * v_norm
    denominator = 1 + 2 * alpha * dot_emb_v + emb_norm * emb_norm * gamma
    coef = (1 + 2 * alpha * dot_emb_v + gamma) / denominator
    dscal(&size, &coef, new_emb, &ONE)

    coef = (1 - emb_norm * emb_norm) * alpha * lr / denominator
    daxpy(&size, &coef, grad, &ONE, new_emb, &ONE)

    # Update the embedding in the global storage.
    dcopy(&size, new_emb, &ONE, emb, &ONE)


# Update a Poincare embedding using the full RSGD and RMSprop updates:
#       emb = exp_emb(lr * riemannian_grad)
# Parameters:
#       emb : array
#           pointer to the beginning of the embedding vector
#       lr : scalar
#           learning rate
#       grad : array
#           Riemannian gradient of loss wrt emb
#       G : array
#           used by RMSprop to accumulate historical gradients
#       size : scalar
#           embedding size
cdef inline void update_poincare_embedding_rmsprop(DOUBLE_t *emb, const DOUBLE_t lr, const DOUBLE_t *grad, DOUBLE_t *G, const int size) nogil:
    cdef DOUBLE_t new_emb[size]
    cdef DOUBLE_t emb_norm, v_norm, dot_emb_v, alpha, gamma, denominator, coef, adaptive_lr
    cdef DOUBLE_t norm_factor, threshold = 1.0 - EPS, coef_old = 0.9, coef_curr = 1-coef_old

    dcopy(&size, emb, &ONE, new_emb, &ONE)

    # Accumulate squared gradient in G.
    # # TODO: if we put here some value of `gamma` e.g. 0.9 for ALPHA and 1-`gamma` for BETA then it becomes RMSprop
    # # dsbmv(&L_CHAR, &size, &ZERO, &ONED, &grad, &ONE, grad, &ONE, &ONED, G, &ONE)
    # dsbmv(&L_CHAR, &size, &ZERO, &coef_old, &grad, &ONE, grad, &ONE, &coef_curr, G, &ONE)

    # TODO: work-around: ideally, we want to take element-wise sqrt from G, but that is computationally costly
    # Take sqrt from the dot product of G. This means that there will be one learning rate for the whole embedding
    # (instead of having different learning rates for each of the components of the embedding).
    # adaptive_lr = lr / log(dnrm2(&size, G, &ONE))

    G[0] = coef_old * G[0] + coef_curr * ddot(&size, grad, &ONE, grad ,&ONE) + 1e-15
    adaptive_lr = lr / sqrt(G[0])

    memset(new_emb, 0, size * cython.sizeof(DOUBLE_t))

    dcopy(&size, emb, &ONE, new_emb, &ONE)
    daxpy(&size, &adaptive_lr, grad, &ONE, new_emb, &ONE)

    # Project embedding back to the Poincare ball i.e. scale to unit norm if it is outside the ball.
    # Currently, we do no clipping after the normalization; instead, we normalize the vectors to 1-EPS, instead of 1.
    emb_norm = dnrm2(&size, new_emb, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        dscal(&size, &norm_factor, new_emb, &ONE)

    # emb_norm = dnrm2(&size, new_emb, &ONE)
    # v_norm = adaptive_lr * dnrm2(&size, grad, &ONE) + 1e-15
    # dot_emb_v = ddot(&size, new_emb, &ONE, grad, &ONE) * adaptive_lr
    # alpha = our_tanh(v_norm / (1 - emb_norm * emb_norm)) / v_norm
    # gamma = alpha * alpha * v_norm * v_norm
    # denominator = 1 + 2 * alpha * dot_emb_v + emb_norm * emb_norm * gamma
    # coef = (1 + 2 * alpha * dot_emb_v + gamma) / denominator
    # dscal(&size, &coef, new_emb, &ONE)
    #
    # coef = (1 - emb_norm * emb_norm) * alpha * adaptive_lr / denominator
    # daxpy(&size, &coef, grad, &ONE, new_emb, &ONE)
    #
    # if dnrm2(&size, new_emb, &ONE) > 1.0:
    #     printf("emb_norm %lf; y_norm %lf; v_norm %lf; dot_emb_v %lf; alpha %lf; gamma %lf; denominator %lf; new_emb norm %lf\n",
    #            emb_norm, alpha*v_norm, v_norm, dot_emb_v, alpha, gamma, denominator, dnrm2(&size, new_emb, &ONE))

    # Update the embedding in the global storage.
    dcopy(&size, new_emb, &ONE, emb, &ONE)


# result will contain 3 values:
#   result[0] = Poincare distance between two vectors in hyperbolic space, v and w
#   result[1] = derivative of the distance wrt v multiplied by the inverse of the conformal factor at v
#   result[1+size] = derivative of the distance wrt w multiplied by the inverse of the conformal factor at w
cdef void poincare_distance_float(REAL_t *result, REAL_t *v, REAL_t *w, const int size) nogil:
    cdef int start
    cdef REAL_t dot_vw, dot_vv, dot_ww, g, alpha_v, beta_w, coef, dist, dacosh, factor, squared_dist

    memset(result, 0, (2 * size + 1) * cython.sizeof(REAL_t))

    dot_vv = our_dot(&size, v, &ONE, v, &ONE)
    dot_ww = our_dot(&size, w, &ONE, w, &ONE)
    dot_vw = our_dot(&size, v, &ONE, w, &ONE)

    # Sanity check
    if dot_vv >= 1 or dot_ww >= 1:
        printf("Cannot compute Poincare distance between points. Points need to be inside the unit ball, but their squared norm is %f and %f.\n", dot_vv, dot_ww)
        exit(-1)

    alpha_v = 1 - dot_vv
    beta_w = 1 - dot_ww
    squared_dist = dot_vv + dot_ww - 2 * dot_vw
    g = 1 + 2 * squared_dist / (alpha_v * beta_w)

    result[0] = acosh(g)

    # Derivative of acosh contribution.
    dacosh = sqrt(g * g - 1)

    # Compute derivative wrt v multiplied by the inverse of the conformal factor at v.
    factor = 1.0 / dacosh * alpha_v / beta_w
    start = 1
    # result[1] = coef1 * v + coef2 * w
    scopy(&size, v, &ONE, &result[start], &ONE)
    coef = (1.0 + squared_dist / alpha_v) * factor
    sscal(&size, &coef, &result[start], &ONE)
    coef = -factor
    our_saxpy(&size, &coef, w, &ONE, &result[start], &ONE)

    # Compute derivative wrt w multiplied by the inverse of the conformal factor at w.
    factor = 1.0 / dacosh * beta_w / alpha_v
    start = 1 + size
    # result[1+size] = coef1 * w + coef2 * v
    scopy(&size, w, &ONE, &result[start], &ONE)
    coef = (1.0 + squared_dist / beta_w) * factor
    sscal(&size, &coef, &result[start], &ONE)
    coef = -factor
    our_saxpy(&size, &coef, v, &ONE, &result[start], &ONE)

# result will contain 3 values:
#   result[0] = Poincare similarity between two vectors in hyperbolic space, v and w, defined as -poincare_dist(v,w)^2
#   result[1] = Riemannian derivative of the similarity wrt v (i.e. Euclidean derivative / conformal factor at v)
#   result[1+size] = Riemannian derivative of the similarity wrt w (i.e. Euclidean derivative / conformal factor at w)
cdef void poincare_similarity(DOUBLE_t *result, DOUBLE_t *v, DOUBLE_t *w, const int size, const SimFuncType sim_func,
                              const DOUBLE_t cosh_dist_pow, int *num_projections) nogil:
    cdef int start
    cdef DOUBLE_t dot_vv, dot_ww, g, alpha_v, beta_w, coef, p_dist, dS_dg, factor, squared_dist
    cdef DOUBLE_t threshold = 1-EPS, norm_factor, emb_norm, scale
    cdef DOUBLE_t diff[size]

    emb_norm = dnrm2(&size, v, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        dscal(&size, &norm_factor, v, &ONE)
        dot_vv = threshold * threshold
        if num_projections != NULL:
            num_projections[0] = num_projections[0] + 1
    else:
        dot_vv = emb_norm * emb_norm

    emb_norm = dnrm2(&size, w, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        dscal(&size, &norm_factor, w, &ONE)
        dot_ww = threshold * threshold
        if num_projections != NULL:
            num_projections[0] = num_projections[0] + 1
    else:
        dot_ww = emb_norm * emb_norm

    # Sanity check
    if dot_vv >= 1 or dot_ww >= 1 or isnan(dot_vv) or isnan(dot_ww):
        printf("[%d, %d] Cannot compute Poincare distance between points. Points need to be inside the unit ball, but their squared norm is %f and %f.\n", v, w, dot_vv, dot_ww)
        exit(-1)

    dcopy(&size, v, &ONE, diff, &ONE)
    daxpy(&size, &MINUS_ONED, w, &ONE, diff, &ONE)
    squared_dist = ddot(&size, diff, &ONE, diff, &ONE)

    alpha_v = 1 - dot_vv
    beta_w = 1 - dot_ww
    g = 1 + 2 * squared_dist / (alpha_v * beta_w)

    if sim_func == SimFuncType.DIST_SQ:
        # Similarity is -poincare_dist(v, w)^2.
        p_dist = acosh(g)
        result[0] = -p_dist * p_dist
    elif sim_func == SimFuncType.COSH_DIST_SQ:
        # Similarity is -cosh(poincare_dist(v, w))^2.
        p_dist = g
        result[0] = -p_dist * p_dist
    elif sim_func == SimFuncType.COSH_DIST_POW_K:
        # Similarity is -cosh(poincare_dist(v, w))^k.
        p_dist = pow(g, cosh_dist_pow)
        result[0] = -p_dist
    elif sim_func == SimFuncType.COSH_DIST:
        # Similarity is -cosh(poincare_dist(v, w)).
        p_dist = g
        result[0] = -p_dist
    elif sim_func == SimFuncType.LOG_DIST:
        # Similarity is -log(poincare_dist(v, w)).
        p_dist = acosh(g)
        result[0] = -log(p_dist)
    elif sim_func == SimFuncType.LOG_DIST_SQ:
        # Similarity is -log(poincare_dist(v, w)^2 + 1).
        p_dist = acosh(g)
        result[0] = -log(p_dist * p_dist + 1)
    elif sim_func == SimFuncType.EXP_DIST:
        # Similarity is -exp(poincare_dist(v, w)).
        p_dist = acosh(g)
        result[0] = -g - sqrt(g * g - 1)

    # Derivative of similarity wrt g.
    if sim_func == SimFuncType.DIST_SQ:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -2.0 * p_dist / sqrt(g * g - 1)
        else:
            # dS_dg is -2 for lim g->1
            dS_dg = -2.0
    elif sim_func == SimFuncType.COSH_DIST_SQ:
        dS_dg = -2.0 * p_dist
    elif sim_func == SimFuncType.COSH_DIST_POW_K:
        dS_dg = -cosh_dist_pow * p_dist / g
    elif sim_func == SimFuncType.COSH_DIST:
        dS_dg = -1.0
    elif sim_func == SimFuncType.LOG_DIST:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -1.0 / sqrt(g * g - 1) / p_dist
        else:
            # dS_dg is -1 for lim g->1
            dS_dg = -1.0
    elif sim_func == SimFuncType.LOG_DIST_SQ:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -2.0 * p_dist / sqrt(g * g - 1) / (p_dist * p_dist + 1)
        else:
            # dS_dg is -2 for lim g->1
            dS_dg = -2.0
    elif sim_func == SimFuncType.EXP_DIST:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -1.0 - g / sqrt(g * g - 1)
        else:
            # dS_dg is -1 for lim g->1
            dS_dg = -1.0

    scale = 1
    result[0] = result[0] * scale
    dS_dg = dS_dg * scale

    # Compute derivative wrt v multiplied by the inverse of the conformal factor at v.
    factor = dS_dg * alpha_v / beta_w
    start = 1
    # result[1] = coef1 * v + coef2 * w
    dcopy(&size, v, &ONE, &result[start], &ONE)
    coef = (1.0 + squared_dist / alpha_v) * factor
    dscal(&size, &coef, &result[start], &ONE)
    coef = -factor
    daxpy(&size, &coef, w, &ONE, &result[start], &ONE)
    clamp(&result[start], size, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)

    # Compute derivative wrt w multiplied by the inverse of the conformal factor at w.
    factor = dS_dg * beta_w / alpha_v
    start = 1 + size
    # result[1+size] = coef1 * w + coef2 * v
    dcopy(&size, w, &ONE, &result[start], &ONE)
    coef = (1.0 + squared_dist / beta_w) * factor
    dscal(&size, &coef, &result[start], &ONE)
    coef = -factor
    daxpy(&size, &coef, v, &ONE, &result[start], &ONE)
    clamp(&result[start], size, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)


cdef unsigned long long fast_sentence_nll_poincare(
        const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
        DOUBLE_t *syn0, DOUBLE_t *syn1neg, DOUBLE_t *b0, DOUBLE_t *b1, DOUBLE_t *Gsyn0, DOUBLE_t *Gsyn1neg,
        DOUBLE_t *Gb0, DOUBLE_t *Gb1, const int size, const np.uint32_t word_index,
        const np.uint32_t word2_index, const DOUBLE_t alpha, const DOUBLE_t l2reg_coef, const OptType optimizer,
        const SimFuncType sim_func, const DOUBLE_t cosh_dist_pow, DOUBLE_t *work, unsigned long long next_random,
        DOUBLE_t *word_locks, const np.uint32_t *index2freq, const int _compute_loss, const int with_bias,
        int *num_projections, DOUBLE_t *_batch_training_loss_param, DOUBLE_t *_epoch_training_loss_param) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size
    cdef long long row2[1+negative]
    cdef unsigned long long modulo = 281474976710655ULL
    cdef DOUBLE_t gamma = 0, gamma_coef, coef, acc_regularizer_loss = 0.0, emb_sq_norm, word2_dot = 0.0
    cdef DOUBLE_t syn0_copy[size], syn1neg_copies[size*(1+negative)], exp_sims_and_grads[(1 + 2 * size) * (1 + negative)]
    cdef np.uint32_t target_index
    cdef int d, start, increment = 1+2*size, idx, syn1neg_idx
    cdef DOUBLE_t adaptive_lr

    memset(work, 0, size * cython.sizeof(DOUBLE_t))
    dcopy(&size, &syn0[row1], &ONE, syn0_copy, &ONE)

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                row2[d] = -1
                continue

        row2[d] = target_index * size
        syn1neg_idx = size * d

        # Compute Poincare distance between vectors. First argument is word2, the second one is word/word_j.
        dcopy(&size, &syn1neg[row2[d]], &ONE, &syn1neg_copies[syn1neg_idx], &ONE)
        start = increment * d
        poincare_similarity(&exp_sims_and_grads[start], syn0_copy, &syn1neg_copies[syn1neg_idx], size, sim_func,
                            cosh_dist_pow, num_projections)
        if with_bias:
            exp_sims_and_grads[start] = exp(exp_sims_and_grads[start] + b1[target_index])
        else:
            exp_sims_and_grads[start] = exp(exp_sims_and_grads[start])
        gamma += exp_sims_and_grads[start]

        daxpy(&size, &exp_sims_and_grads[start], &exp_sims_and_grads[start + 1], &ONE, work, &ONE)

        if _compute_loss and target_index < REG_WORD_INDEX_THRESHOLD:
            emb_sq_norm = ddot(&size, &syn1neg_copies[syn1neg_idx], &ONE, &syn1neg_copies[syn1neg_idx], &ONE)
            acc_regularizer_loss += l2reg_coef * emb_sq_norm

            # IMPORTANT!!! syn1neg_copies[syn1neg_idx] will now store -2*lambda*output_emb*1/conformal_factor, instead of output_emb !!!!!!!!!!!!!!!!!!!
            # Multiply by -1 because when performing the update we do gradient ascent, so we need to have negative sign here
            # for a penalty. Also multiply by inverse of the conformal factor at w2.
            coef = -2 * l2reg_coef * (1 - emb_sq_norm) * (1 - emb_sq_norm) / 4
            dscal(&size, &coef, &syn1neg_copies[syn1neg_idx], &ONE)

    gamma_coef = - 1.0 / gamma
    # Gradient wrt w2. NOTE: exp_sims_and_grads[1] is modified here and should not be used past this point!!
    daxpy(&size, &gamma_coef, work, &ONE, &exp_sims_and_grads[1], &ONE)
    # Update input embedding of word2.
    if optimizer == OptType.RSGD:
        update_poincare_embedding(&syn0[row1], alpha, &exp_sims_and_grads[1], size)
    elif optimizer == OptType.WeightedRSGD:
        adaptive_lr = alpha / sqrt(index2freq[word2_index] / 100.0)
        update_poincare_embedding(&syn0[row1], adaptive_lr, &exp_sims_and_grads[1], size)
    elif optimizer == OptType.FullRSGD:
        update_poincare_embedding_full_rsgd(&syn0[row1], alpha, &exp_sims_and_grads[1], size)
    elif optimizer == OptType.WeightedFullRSGD:
        adaptive_lr = alpha / sqrt(index2freq[word2_index] / 100.0)
        update_poincare_embedding_full_rsgd(&syn0[row1], adaptive_lr, &exp_sims_and_grads[1], size)
    elif optimizer == OptType.RMSprop:
        update_poincare_embedding_rmsprop(&syn0[row1], alpha, &exp_sims_and_grads[1], &Gsyn0[row1], size)


    # Update output embedding for the target context w (i.e. d==0) and for the negative samples w_j.
    for d in range(negative+1):
        if row2[d] == -1:
            continue
        start = increment * d
        syn1neg_idx = size * d
        idx = word_index if d == 0 else row2[d] / size
        coef = (1 + gamma_coef * exp_sims_and_grads[start]) if d == 0 else gamma_coef * exp_sims_and_grads[start]
        adaptive_lr = alpha / sqrt(index2freq[idx] / 100.0)
        if l2reg_coef and idx < REG_WORD_INDEX_THRESHOLD:
            # Gradient wrt w/w_j + contribution of the regularizer (i.e. syn1neg_copies == -2*lambda*output_emb*1/conformal_factor * emb).
            daxpy(&size, &coef, &exp_sims_and_grads[start + size + 1], &ONE, &syn1neg_copies[syn1neg_idx], &ONE)
            if optimizer == OptType.RSGD:
                update_poincare_embedding(&syn1neg[row2[d]], alpha, &syn1neg_copies[syn1neg_idx], size)
            elif optimizer == OptType.WeightedRSGD:
                update_poincare_embedding(&syn1neg[row2[d]], adaptive_lr, &syn1neg_copies[syn1neg_idx], size)
            elif optimizer == OptType.FullRSGD:
                update_poincare_embedding_full_rsgd(&syn1neg[row2[d]], alpha, &syn1neg_copies[syn1neg_idx], size)
            elif optimizer == OptType.WeightedFullRSGD:
                update_poincare_embedding_full_rsgd(&syn1neg[row2[d]], adaptive_lr, &syn1neg_copies[syn1neg_idx], size)
            elif optimizer == OptType.RMSprop:
                update_poincare_embedding_rmsprop(&syn1neg[row2[d]], alpha, &syn1neg_copies[syn1neg_idx], &Gsyn1neg[row2[d]], size)
        else:
            # Gradient wrt w/w_j.
            if optimizer == OptType.RSGD:
                update_poincare_embedding(&syn1neg[row2[d]], coef * alpha, &exp_sims_and_grads[start + size + 1], size)
            elif optimizer == OptType.WeightedRSGD:
                update_poincare_embedding(&syn1neg[row2[d]], coef * adaptive_lr, &exp_sims_and_grads[start + size + 1], size)
            elif optimizer == OptType.FullRSGD:
                dscal(&size, &coef, &exp_sims_and_grads[start + size + 1], &ONE)
                update_poincare_embedding_full_rsgd(&syn1neg[row2[d]], alpha, &exp_sims_and_grads[start + size + 1], size)
            elif optimizer == OptType.WeightedFullRSGD:
                dscal(&size, &coef, &exp_sims_and_grads[start + size + 1], &ONE)
                update_poincare_embedding_full_rsgd(&syn1neg[row2[d]], adaptive_lr, &exp_sims_and_grads[start + size + 1], size)
            elif optimizer == OptType.RMSprop:
                dscal(&size, &coef, &exp_sims_and_grads[start + size + 1], &ONE)
                update_poincare_embedding_rmsprop(&syn1neg[row2[d]], alpha, &exp_sims_and_grads[start + size + 1], &Gsyn1neg[row2[d]], size)

        if with_bias:
            if optimizer == OptType.RSGD or optimizer == OptType.FullRSGD:
                b1[idx] += alpha * coef
            elif optimizer == OptType.WeightedRSGD or optimizer == OptType.WeightedFullRSGD:
                b1[idx] += adaptive_lr * coef
            elif optimizer == OptType.RMSprop:
                Gb1[idx] += coef * coef
                adaptive_lr = alpha / sqrt(Gb1[idx])
                b1[idx] += adaptive_lr * coef

    if _compute_loss == 1:
        coef = log(exp_sims_and_grads[0]) - log(gamma) + acc_regularizer_loss + l2reg_coef * word2_dot
        _batch_training_loss_param[0] -= coef
        _epoch_training_loss_param[0] -= coef

    return next_random


cdef unsigned long long fast_sentence_sg_poincare(
        const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
        DOUBLE_t *syn0, DOUBLE_t *syn1neg, DOUBLE_t *b0, DOUBLE_t *b1, const int size, const np.uint32_t word_index,
        const np.uint32_t word2_index, const DOUBLE_t alpha, const DOUBLE_t l2reg_coef, const OptType optimizer,
        const SimFuncType sim_func, const DOUBLE_t cosh_dist_pow, DOUBLE_t *work, unsigned long long next_random, DOUBLE_t *word_locks,
        const np.uint32_t *index2freq, const int _compute_loss, const int with_bias,
        DOUBLE_t *_batch_training_loss_param, DOUBLE_t *_epoch_training_loss_param) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef DOUBLE_t f, g, label, f_sim, word2_dot, coef, dL_dS, emb_sq_norm
    cdef DOUBLE_t poincare_sim[2 * size + 1], syn0_copy[size], syn1neg_copy[size]
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(DOUBLE_t))
    dcopy(&size, &syn0[row1], &ONE, syn0_copy, &ONE)
    word2_dot = ddot(&size, syn0_copy, &ONE, syn0_copy, &ONE)

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONED
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <DOUBLE_t>0.0

        row2 = target_index * size

        # Compute Poincare distance between vectors. First argument is word2, the second one is word/word_j.
        dcopy(&size, &syn1neg[row2], &ONE, syn1neg_copy, &ONE)
        poincare_similarity(poincare_sim, syn0_copy, syn1neg_copy, size, sim_func, cosh_dist_pow, NULL)
        f_sim = poincare_sim[0]
        if with_bias:
            f_sim = f_sim + b0[word2_index] + b1[target_index]

        if f_sim >= MAX_EXP:
            f = 1.0
        elif f_sim <= -MAX_EXP:
            f = 0.0
        else:
            f = <DOUBLE_t> EXP_TABLE[<int>((f_sim + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        dL_dS = label - f

        if _compute_loss == 1:
            f_sim = (f_sim if d == 0 else -f_sim)
            compute_sgns_loss_dbl(f_sim, _batch_training_loss_param, _epoch_training_loss_param)

        # Accumulate value by which we will update the input embedding of word2.
        # Multiply g by dS/dv_w2 (i.e. the result of poincare_similarity).
        g = dL_dS * alpha
        if optimizer == OptType.WeightedRSGD or optimizer == OptType.WeightedFullRSGD:
            g = g / sqrt(index2freq[word2_index] / 100.0)
        daxpy(&size, &g, &poincare_sim[1], &ONE, work, &ONE)

        # Update output embeddings (for either word or one of the negative samples).
        # Multiply g by dS/dv_w or dS/dv_w_j (i.e. the result of poincare_similarity).
        g = dL_dS * alpha
        if optimizer == OptType.WeightedRSGD or optimizer == OptType.WeightedFullRSGD:
            g = g / sqrt(index2freq[target_index] / 100.0)
        if optimizer == OptType.RSGD or optimizer == OptType.WeightedRSGD:
            update_poincare_embedding(&syn1neg[row2], g, &poincare_sim[size+1], size)
        elif optimizer == OptType.FullRSGD or optimizer == OptType.WeightedFullRSGD:
            update_poincare_embedding_full_rsgd(&syn1neg[row2], g, &poincare_sim[size+1], size)
        # Update biases.
        if with_bias:
            b0[word2_index] += g
            b1[target_index] += g

    # Add the gradient of the regularization term.
    # Update input embedding of word2.
    if optimizer == OptType.RSGD or optimizer == OptType.WeightedRSGD:
        update_poincare_embedding(&syn0[row1], word_locks[word2_index], work, size)
    elif optimizer == OptType.FullRSGD or optimizer == OptType.WeightedFullRSGD:
        update_poincare_embedding_full_rsgd(&syn0[row1], word_locks[word2_index], work, size)

    return next_random


cdef unsigned long long fast_sentence_sg_euclid(
        const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
        REAL_t *syn0, REAL_t *syn1neg, REAL_t *b0, REAL_t *b1, const int size, const np.uint32_t word_index,
        const np.uint32_t word2_index, const REAL_t alpha, REAL_t *work,
        unsigned long long next_random, REAL_t *word_locks, const int normalized, const int _compute_loss,
        const int with_bias, REAL_t *_batch_training_loss_param, REAL_t *_epoch_training_loss_param,
        const int debug) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_sim
    cdef REAL_t diff[size]
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))
    memset(diff, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size

        # Compute Euclidean distance between vectors.
        # diff = v’_word - v_word2
        scopy(&size, &syn1neg[row2], &ONE, diff, &ONE)
        our_saxpy(&size, &MINUS_ONEF, &syn0[row1], &ONE, diff, &ONE)
        # Similarity is -dist(v, v2)^2.
        f_sim = -our_dot(&size, diff, &ONE, diff, &ONE)
        if with_bias:
            f_sim = f_sim + b0[word2_index] + b1[target_index]

        if f_sim >= MAX_EXP:
            f = 1.0
        elif f_sim <= -MAX_EXP:
            f = 0.0
        else:
            f = EXP_TABLE[<int>((f_sim + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_sim = (f_sim if d == 0 else -f_sim)
            compute_sgns_loss(f_sim, _batch_training_loss_param, _epoch_training_loss_param)

        # Accumulate value by which we will update the input embedding of word2.
        # Multiply g by dS/dv_w2 i.e. 2 * diff (we drop the constant factor though).
        our_saxpy(&size, &g, diff, &ONE, work, &ONE)
        # Update output embeddings (for either word or one of the negative samples).
        # Multiply g by dS/dv_w or dS/dv_w_j i.e. 2 * (-diff) (we drop the constant factor though).
        update_embedding(&syn1neg[row2], -g, diff, size, normalized, debug)
        # Update biases.
        if with_bias:
            b0[word2_index] += g
            b1[target_index] += g

    # Update input embedding of word2.
    update_embedding(&syn0[row1], word_locks[word2_index], work, size, normalized, debug)

    return next_random


cdef unsigned long long fast_sentence_nll_neg(
        const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
        REAL_t *syn0, REAL_t *syn1neg, REAL_t *b0, REAL_t *b1, REAL_t *Gsyn0, REAL_t *Gsyn1neg,
        REAL_t *Gb0, REAL_t *Gb1, const int size, const np.uint32_t word_index,
        const np.uint32_t word2_index, const REAL_t alpha, const OptType optimizer, REAL_t *work,
        unsigned long long next_random, REAL_t *word_locks, const np.uint32_t *index2freq, const int normalized,
        const int _compute_loss, const int with_bias, REAL_t *_batch_training_loss_param,
        REAL_t *_epoch_training_loss_param, const int debug) nogil:

    cdef long long row1 = word2_index * size
    cdef long long row2[1+negative]
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t gamma = 0, gamma_coef, coef
    cdef np.uint32_t target_index
    cdef REAL_t adaptive_lr, f_sim
    cdef REAL_t grad[size], exp_terms[1+negative], syn0_copy[size]

    memset(work, 0, size * cython.sizeof(REAL_t))
    scopy(&size, &syn0[row1], &ONE, syn0_copy, &ONE)

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                row2[d] = -1
                continue

        row2[d] = target_index * size

        # Compute dot product between vectors.
        f_sim = our_dot(&size, syn0_copy, &ONE, &syn1neg[row2[d]], &ONE)

        if with_bias:
            exp_terms[d] = exp(f_sim + b1[target_index])
        else:
            exp_terms[d] = exp(f_sim)
        gamma += exp_terms[d]

        saxpy(&size, &exp_terms[d], &syn1neg[row2[d]], &ONE, work, &ONE)

    gamma_coef = - 1.0 / gamma
    # Gradient wrt w2.
    scopy(&size, &syn1neg[row2[0]], &ONE, grad, &ONE)
    saxpy(&size, &gamma_coef, work, &ONE, grad, &ONE)
    # Update input embedding of word2.
    if optimizer == OptType.SGD:
        update_embedding(&syn0[row1], alpha, grad, size, normalized, debug)
    elif optimizer == OptType.WSGD:
        adaptive_lr = alpha / sqrt(index2freq[word2_index] / 100.0)
        update_embedding(&syn0[row1], adaptive_lr, grad, size, normalized, debug)

    # Update output embedding for the target context w (i.e. d==0) and for the negative samples w_j.
    for d in range(negative+1):
        if row2[d] == -1:
            continue
        idx = word_index if d == 0 else row2[d] / size
        coef = (1 + gamma_coef * exp_terms[d]) if d == 0 else gamma_coef * exp_terms[d]
        adaptive_lr = alpha / sqrt(index2freq[idx] / 100.0)
        # Gradient wrt w/w_j.
        if optimizer == OptType.SGD:
            update_embedding(&syn1neg[row2[d]], alpha * coef, syn0_copy, size, normalized, debug)
        elif optimizer == OptType.WSGD:
            update_embedding(&syn1neg[row2[d]], adaptive_lr * coef, syn0_copy, size, normalized, debug)

        if with_bias:
            if optimizer == OptType.SGD:
                b1[idx] += alpha * coef
            elif optimizer == OptType.WSGD:
                b1[idx] += adaptive_lr * coef

    if _compute_loss == 1:
        coef = log(exp_terms[0]) - log(gamma)
        _batch_training_loss_param[0] -= coef
        _epoch_training_loss_param[0] -= coef

    return next_random


cdef unsigned long long fast_sentence_sg_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len,
    REAL_t *syn0, REAL_t *syn1neg, REAL_t *b0, REAL_t *b1, REAL_t *Gsyn0, REAL_t *Gsyn1neg,
        REAL_t *Gb0, REAL_t *Gb1, const int size, const np.uint32_t word_index,
    const np.uint32_t word2_index, const REAL_t alpha, OptType optimizer, REAL_t *work,
    unsigned long long next_random, REAL_t *word_locks, const np.uint32_t *index2freq, const int normalized, const int _compute_loss,
    const int with_bias, REAL_t *_batch_training_loss_param, REAL_t *_epoch_training_loss_param,
    const int debug) nogil:

    cdef long long a
    cdef long long row1 = word2_index * size, row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, label, f_sim, adaptive_lr, coef_old=0.9, coef_curr = 1.0-coef_old
    cdef np.uint32_t target_index
    cdef int d

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_sim = our_dot(&size, &syn0[row1], &ONE, &syn1neg[row2], &ONE)
        if with_bias:
            f_sim = f_sim + b0[word2_index] + b1[target_index]

        if f_sim >= MAX_EXP:
            f = 1.0
        elif f_sim <= -MAX_EXP:
            f = 0.0
        else:
            f = EXP_TABLE[<int>((f_sim + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]

        if _compute_loss == 1:
            f_sim = (f_sim if d == 0 else -f_sim)
            compute_sgns_loss(f_sim, _batch_training_loss_param, _epoch_training_loss_param)

        g = (label - f) * alpha

        if optimizer == OptType.SGD:
            # Accumulate value by which we will update the input embeddings.
            our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
            # Update output embedding.
            update_embedding(&syn1neg[row2], g, &syn0[row1], size, normalized, debug)
        elif optimizer == OptType.WSGD:
            adaptive_lr = alpha / sqrt(index2freq[word2_index] / 100.0)
            g = (label - f) * adaptive_lr

            # Accumulate value by which we will update the input embeddings.
            our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
            # Update output embedding.
            update_embedding(&syn1neg[row2], g, &syn0[row1], size, normalized, debug)
        elif optimizer == OptType.RMSprop:
            # XXX: This is only a modified version of RMSprop, where we use the same lr for one embedding,
            # not a different lr for each of the elements of an embedding.
            g = label - f
            Gsyn1neg[row2] = coef_old * Gsyn1neg[row2] + coef_curr * g * g * dsdot(&size, &syn0[row1], &ONE, &syn0[row1], &ONE) + 1e-15
            adaptive_lr = alpha / sqrt(Gsyn1neg[row2])

            # Accumulate value by which we will update the input embeddings.
            our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
            # Update output embedding.
            update_embedding(&syn1neg[row2], adaptive_lr * g, &syn0[row1], size, normalized, debug)

        # Update biases.
        if with_bias:
            # TODO: use adaptive_lr for RMSprop for biases
            if optimizer != SGD:
                printf("Biases are currently only supported with SGD\n")
                exit(-1)
            b0[word2_index] += g
            b1[target_index] += g

    if optimizer == OptType.SGD or optimizer == OptType.WSGD:
        # Update input embedding of word2.
        update_embedding(&syn0[row1], word_locks[word2_index], work, size, normalized, debug)
    elif optimizer == OptType.RMSprop:
        Gsyn0[row1] = coef_old * Gsyn0[row1] + coef_curr * dsdot(&size, work, &ONE, work, &ONE) + 1e-15
        adaptive_lr = alpha / sqrt(Gsyn0[row1])
        update_embedding(&syn0[row1], adaptive_lr * word_locks[word2_index], work, size, normalized, debug)

    return next_random


cdef void fast_sentence_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_epoch_training_loss_param) nogil:

    cdef long long a, b
    cdef long long row2, sgn
    cdef REAL_t f, g, count, inv_count = 1.0, f_dot, lprob
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))
    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (1 - word_code[b] - f) * alpha

        if _compute_loss == 1:
            sgn = (-1)**word_code[b]  # ch function: 0-> 1, 1 -> -1
            lprob = -1*sgn*f_dot
            if lprob <= -MAX_EXP or lprob >= MAX_EXP:
                continue
            lprob = LOG_TABLE[<int>((lprob + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _epoch_training_loss_param[0] = _epoch_training_loss_param[0] - lprob

        our_saxpy(&size, &g, &syn1[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j, k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m] * size], &ONE)


cdef unsigned long long fast_sentence_cbow_neg(
    const int negative, np.uint32_t *cum_table, unsigned long long cum_table_len, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1,  REAL_t *syn0, REAL_t *syn1neg, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], const REAL_t alpha, REAL_t *work,
    int i, int j, int k, int cbow_mean, unsigned long long next_random, REAL_t *word_locks,
    const int _compute_loss, REAL_t *_epoch_training_loss_param) nogil:

    cdef long long a
    cdef long long row2
    cdef unsigned long long modulo = 281474976710655ULL
    cdef REAL_t f, g, count, inv_count = 1.0, label, log_e_f_dot, f_dot
    cdef np.uint32_t target_index, word_index
    cdef int d, m

    word_index = indexes[i]

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)  # (does this need BLAS-variants like saxpy?)

    memset(work, 0, size * cython.sizeof(REAL_t))

    for d in range(negative+1):
        if d == 0:
            target_index = word_index
            label = ONEF
        else:
            target_index = bisect_left(cum_table, (next_random >> 16) % cum_table[cum_table_len-1], 0, cum_table_len)
            next_random = (next_random * <unsigned long long>25214903917ULL + 11) & modulo
            if target_index == word_index:
                continue
            label = <REAL_t>0.0

        row2 = target_index * size
        f_dot = our_dot(&size, neu1, &ONE, &syn1neg[row2], &ONE)
        if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
            continue
        f = EXP_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        if _compute_loss == 1:
            f_dot = (f_dot if d == 0  else -f_dot)
            if f_dot <= -MAX_EXP or f_dot >= MAX_EXP:
                continue
            log_e_f_dot = LOG_TABLE[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
            _epoch_training_loss_param[0] = _epoch_training_loss_param[0] - log_e_f_dot

        our_saxpy(&size, &g, &syn1neg[row2], &ONE, work, &ONE)
        our_saxpy(&size, &g, neu1, &ONE, &syn1neg[row2], &ONE)

    if not cbow_mean:  # divide error over summed window vectors
        sscal(&size, &inv_count, work, &ONE)  # (does this need BLAS-variants like saxpy?)

    for m in range(j,k):
        if m == i:
            continue
        else:
            our_saxpy(&size, &word_locks[indexes[m]], work, &ONE, &syn0[indexes[m]*size], &ONE)

    return next_random


def train_batch_sg(model, sentences, alpha, _work, compute_loss):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int euclid = model.get_attr('euclid', 0)
    cdef int poincare = model.get_attr('poincare', 0)
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int normalized = (1 if model.get_attr('normalized', False) == True else 0)
    cdef int with_bias = (1 if model.with_bias == True else 0)
    cdef int debug = (1 if model.get_attr('debug', False) else 0)
    cdef int is_nll = (1 if model.get_attr('is_nll', False) == True else 0)

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _epoch_training_loss = model.running_training_loss

    cdef str opt_str = model.get_attr('optimizer', 'rsgd' if poincare == 1 else 'sgd')
    cdef OptType optimizer

    cdef REAL_t batch_loss = 0.0

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    # For biases
    cdef REAL_t *b0, *b1
    # Arrays used by RMSprop.
    cdef REAL_t *Gsyn0, *Gsyn1neg, *Gb0, *Gb1

    cdef np.uint32_t *index2freq = <np.uint32_t *>(np.PyArray_DATA(model.wv.index2freq))

    # Convert optimizer from string to one of the enum values.
    if opt_str == "sgd":
        optimizer = OptType.SGD
    elif opt_str == "wsgd":
        optimizer = OptType.WSGD
    elif opt_str == "rmsprop":
        optimizer = OptType.RMSprop
    else:
        print("Unrecognized optimizer type", opt_str)
        exit(-1)

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
        if with_bias:
            b0 = <REAL_t *>(np.PyArray_DATA(model.trainables.b0))
            b1 = <REAL_t *>(np.PyArray_DATA(model.trainables.b1))
        if optimizer == OptType.RMSprop:
            Gsyn0 = <REAL_t *>(np.PyArray_DATA(model.trainables.Gsyn0))
            Gsyn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.Gsyn1neg))
            if with_bias:
                Gb0 = <REAL_t *>(np.PyArray_DATA(model.trainables.Gb0))
                Gb1 = <REAL_t *>(np.PyArray_DATA(model.trainables.Gb1))
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        fast_sentence_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], _alpha, work, word_locks, _compute_loss, &_epoch_training_loss)
                    if negative:
                        if euclid == 1:
                            # Use a Euclidean distance-based similarity between words for training.
                            next_random = fast_sentence_sg_euclid(negative, cum_table, cum_table_len, syn0, syn1neg, b0, b1, size, indexes[i], indexes[j], _alpha, work, next_random, word_locks, normalized, _compute_loss, with_bias, &batch_loss, &_epoch_training_loss, debug)
                        elif poincare == 1:
                            printf("Float embeddings are not supported for the Poincare space. Try calling the double precision train_batch function instead.\n")
                            exit(-1)
                        else:
                            if is_nll:
                                # Use dot product between words for training and NLL loss.
                                next_random = fast_sentence_nll_neg(negative, cum_table, cum_table_len, syn0, syn1neg, b0, b1, Gsyn0, Gsyn1neg, Gb0, Gb1, size, indexes[i], indexes[j], _alpha, optimizer, work, next_random, word_locks, index2freq, normalized, _compute_loss, with_bias, &batch_loss, &_epoch_training_loss, debug)
                            else:
                                # Use dot product between words for training. This yields vanilla Skip-gram word2vec.
                                next_random = fast_sentence_sg_neg(negative, cum_table, cum_table_len, syn0, syn1neg, b0, b1, Gsyn0, Gsyn1neg, Gb0, Gb1, size, indexes[i], indexes[j], _alpha, optimizer, work, next_random, word_locks, index2freq, normalized, _compute_loss, with_bias, &batch_loss, &_epoch_training_loss, debug)

    model.running_training_loss = _epoch_training_loss
    if debug and random_int32(&next_random) % 100 == 0:
        print("min radius {}".format(np.min(model.wv.vectors[0])))
        print("max radius {}".format(np.max(model.wv.vectors[0])))
        print("`the` max angle {}".format(np.max(model.wv["the"])))
    return effective_words, batch_loss


def train_batch_sg_dbl(model, sentences, alpha, _work, compute_loss):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int euclid = model.get_attr('euclid', 0)
    cdef int poincare = model.get_attr('poincare', 0)
    cdef int sample = (model.vocabulary.sample != 0)
    cdef float _l2reg_coef = model.get_attr('l2reg_coef', 0.0)
    cdef int normalized = (1 if model.get_attr('normalized', False) == True else 0)
    cdef int is_nll = (1 if model.get_attr('is_nll', False) == True else 0)
    cdef int with_bias = (1 if model.with_bias == True else 0)
    cdef int debug = (1 if model.get_attr('debug', False) else 0)

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef DOUBLE_t _epoch_training_loss = model.running_training_loss
    cdef int num_projections = 0

    cdef DOUBLE_t batch_loss = 0.0

    cdef str opt_str = model.get_attr('optimizer', 'rsgd' if poincare == 1 else 'sgd')
    cdef OptType optimizer

    cdef str sim_func_str = model.get_attr('sim_func', 'dist_sq')
    cdef SimFuncType sim_func
    cdef DOUBLE_t cosh_dist_pow = float(model.get_attr('cosh_dist_pow', 0.0))

    cdef DOUBLE_t *syn0 = <DOUBLE_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef DOUBLE_t *word_locks = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef DOUBLE_t *work
    cdef DOUBLE_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef DOUBLE_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef DOUBLE_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random
    # For biases
    cdef DOUBLE_t *b0, *b1
    # Arrays used by RMSprop.
    cdef DOUBLE_t *Gsyn0, *Gsyn1neg, *Gb0, *Gb1

    cdef np.uint32_t *index2freq = <np.uint32_t *>(np.PyArray_DATA(model.wv.index2freq))

    # Convert optimizer from string to one of the enum values.
    if opt_str == "sgd":
        optimizer = OptType.SGD
    elif opt_str == "rsgd":
        optimizer = OptType.RSGD
    elif opt_str == "wrsgd":
        optimizer = OptType.WeightedRSGD
    elif opt_str == "fullrsgd":
        optimizer = OptType.FullRSGD
    elif opt_str == "wfullrsgd":
        optimizer = OptType.WeightedFullRSGD
    elif opt_str == "rmsprop":
        optimizer = OptType.RMSprop
    else:
        print("Unrecognized optimizer type", opt_str)
        exit(-1)

    # Convert similarity function from string to one of the enum values.
    if sim_func_str == "dist-sq":
        sim_func = SimFuncType.DIST_SQ
    elif sim_func_str == "cosh-dist-sq":
        sim_func = SimFuncType.COSH_DIST_SQ
    elif "cosh-dist-pow-" in sim_func_str:
        sim_func = SimFuncType.COSH_DIST_POW_K
    elif sim_func_str == "cosh-dist":
        sim_func = SimFuncType.COSH_DIST
    elif sim_func_str == "log-dist":
        sim_func = SimFuncType.LOG_DIST
    elif sim_func_str == "log-dist-sq":
        sim_func = SimFuncType.LOG_DIST_SQ
    elif sim_func_str == "exp-dist":
        sim_func = SimFuncType.EXP_DIST
    else:
        print("Unrecognized similarity function type", sim_func_str)
        exit(-1)

    # num_projections only supported for wfullrsgd and fullrsgd
    if optimizer != OptType.WeightedFullRSGD and optimizer != OptType.FullRSGD:
        num_projections = -1

    if hs:
        syn1 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
        if with_bias:
            b0 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.b0))
            b1 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.b1))
        if optimizer == OptType.RMSprop:
            Gsyn0 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.Gsyn0))
            Gsyn1neg = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.Gsyn1neg))
            if with_bias:
                Gb0 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.Gb0))
                Gb1 = <DOUBLE_t *>(np.PyArray_DATA(model.trainables.Gb1))
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <DOUBLE_t *>np.PyArray_DATA(_work)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                for j in range(j, k):
                    if j == i:
                        continue
                    if hs:
                        printf("Hierarchical softmax is not supported for Poincare embeddings.\n")
                        exit(-1)
                    if negative:
                        if poincare == 1:
                            if is_nll == 0:
                                # Use a Poincare distance-based similarity between words for training.
                                next_random = fast_sentence_sg_poincare(negative, cum_table, cum_table_len, syn0, syn1neg, b0, b1, size, indexes[i], indexes[j], _alpha, _l2reg_coef, optimizer, sim_func, cosh_dist_pow, work, next_random, word_locks, index2freq, _compute_loss, with_bias, &batch_loss, &_epoch_training_loss)
                            else:
                                # Use a Poincare distance-based similarity between words for training and NLL loss.
                                next_random = fast_sentence_nll_poincare(negative, cum_table, cum_table_len, syn0, syn1neg, b0, b1, Gsyn0, Gsyn1neg, Gb0, Gb1, size, indexes[i], indexes[j], _alpha, _l2reg_coef, optimizer, sim_func, cosh_dist_pow, work, next_random, word_locks, index2freq, _compute_loss, with_bias, &num_projections, &batch_loss, &_epoch_training_loss)
                        else:
                            printf("Double embeddings are not supported for this embedding method. Try calling the float precision train_batch function instead.\n")
                            exit(-1)

    model.num_projections = num_projections
    model.running_training_loss = _epoch_training_loss
    if debug and random_int32(&next_random) % 100 == 0:
        print("dog norm {}".format(np.linalg.norm(model.wv["dog"])))

    return effective_words, batch_loss


def train_batch_cbow(model, sentences, alpha, _work, _neu1, compute_loss):
    cdef int hs = model.hs
    cdef int negative = model.negative
    cdef int sample = (model.vocabulary.sample != 0)
    cdef int cbow_mean = model.cbow_mean

    cdef int _compute_loss = (1 if compute_loss == True else 0)
    cdef REAL_t _epoch_training_loss = model.running_training_loss

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *word_locks = <REAL_t *>(np.PyArray_DATA(model.trainables.vectors_lockf))
    cdef REAL_t *work
    cdef REAL_t _alpha = alpha
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef np.uint32_t reduced_windows[MAX_SENTENCE_LEN]
    cdef int sentence_idx[MAX_SENTENCE_LEN + 1]
    cdef int window = model.window

    cdef int i, j, k
    cdef int effective_words = 0, effective_sentences = 0
    cdef int sent_idx, idx_start, idx_end

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    # For negative sampling
    cdef REAL_t *syn1neg
    cdef np.uint32_t *cum_table
    cdef unsigned long long cum_table_len
    # for sampling (negative and frequent-word downsampling)
    cdef unsigned long long next_random

    if hs:
        syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    if negative:
        syn1neg = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1neg))
        cum_table = <np.uint32_t *>(np.PyArray_DATA(model.vocabulary.cum_table))
        cum_table_len = len(model.vocabulary.cum_table)
    if negative or sample:
        next_random = (2**24) * model.random.randint(0, 2**24) + model.random.randint(0, 2**24)

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    # prepare C structures so we can go "full C" and release the Python GIL
    vlookup = model.wv.vocab
    sentence_idx[0] = 0  # indices of the first sentence always start at 0
    for sent in sentences:
        if not sent:
            continue  # ignore empty sentences; leave effective_sentences unchanged
        for token in sent:
            word = vlookup[token] if token in vlookup else None
            if word is None:
                continue  # leaving `effective_words` unchanged = shortening the sentence = expanding the window
            if sample and word.sample_int < random_int32(&next_random):
                continue
            indexes[effective_words] = word.index
            if hs:
                codelens[effective_words] = <int>len(word.code)
                codes[effective_words] = <np.uint8_t *>np.PyArray_DATA(word.code)
                points[effective_words] = <np.uint32_t *>np.PyArray_DATA(word.point)
            effective_words += 1
            if effective_words == MAX_SENTENCE_LEN:
                break  # TODO: log warning, tally overflow?

        # keep track of which words go into which sentence, so we don't train
        # across sentence boundaries.
        # indices of sentence number X are between <sentence_idx[X], sentence_idx[X])
        effective_sentences += 1
        sentence_idx[effective_sentences] = effective_words

        if effective_words == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?

    # precompute "reduced window" offsets in a single randint() call
    for i, item in enumerate(model.random.randint(0, window, effective_words)):
        reduced_windows[i] = item

    # release GIL & train on all sentences
    with nogil:
        for sent_idx in range(effective_sentences):
            idx_start = sentence_idx[sent_idx]
            idx_end = sentence_idx[sent_idx + 1]
            for i in range(idx_start, idx_end):
                j = i - window + reduced_windows[i]
                if j < idx_start:
                    j = idx_start
                k = i + window + 1 - reduced_windows[i]
                if k > idx_end:
                    k = idx_end
                if hs:
                    fast_sentence_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, _alpha, work, i, j, k, cbow_mean, word_locks, _compute_loss, &_epoch_training_loss)
                if negative:
                    next_random = fast_sentence_cbow_neg(negative, cum_table, cum_table_len, codelens, neu1, syn0, syn1neg, size, indexes, _alpha, work, i, j, k, cbow_mean, next_random, word_locks, _compute_loss, &_epoch_training_loss)

    model.running_training_loss = _epoch_training_loss
    return effective_words


# Score is only implemented for hierarchical softmax
def score_sentence_sg(model, sentence, _work):

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # should drop the
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0

    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            for j in range(j, k):
                if j == i or codelens[j] == 0:
                    continue
                score_pair_sg_hs(points[i], codes[i], codelens[i], syn0, syn1, size, indexes[j], work)

    return work[0]

cdef void score_pair_sg_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, const int codelen,
    REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t word2_index, REAL_t *work) nogil:

    cdef long long b
    cdef long long row1 = word2_index * size, row2, sgn
    cdef REAL_t f

    for b in range(codelen):
        row2 = word_point[b] * size
        f = our_dot(&size, &syn0[row1], &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f

def score_sentence_cbow(model, sentence, _work, _neu1):

    cdef int cbow_mean = model.cbow_mean

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *work
    cdef REAL_t *neu1
    cdef int size = model.wv.vector_size

    cdef int codelens[MAX_SENTENCE_LEN]
    cdef np.uint32_t indexes[MAX_SENTENCE_LEN]
    cdef int sentence_len
    cdef int window = model.window

    cdef int i, j, k
    cdef long result = 0

    # For hierarchical softmax
    cdef REAL_t *syn1
    cdef np.uint32_t *points[MAX_SENTENCE_LEN]
    cdef np.uint8_t *codes[MAX_SENTENCE_LEN]

    syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))

    # convert Python structures to primitive types, so we can release the GIL
    work = <REAL_t *>np.PyArray_DATA(_work)
    neu1 = <REAL_t *>np.PyArray_DATA(_neu1)

    vlookup = model.wv.vocab
    i = 0
    for token in sentence:
        word = vlookup[token] if token in vlookup else None
        if word is None:
            continue  # for score, should this be a default negative value?
        indexes[i] = word.index
        codelens[i] = <int>len(word.code)
        codes[i] = <np.uint8_t *>np.PyArray_DATA(word.code)
        points[i] = <np.uint32_t *>np.PyArray_DATA(word.point)
        result += 1
        i += 1
        if i == MAX_SENTENCE_LEN:
            break  # TODO: log warning, tally overflow?
    sentence_len = i

    # release GIL & train on the sentence
    work[0] = 0.0
    with nogil:
        for i in range(sentence_len):
            if codelens[i] == 0:
                continue
            j = i - window
            if j < 0:
                j = 0
            k = i + window + 1
            if k > sentence_len:
                k = sentence_len
            score_pair_cbow_hs(points[i], codes[i], codelens, neu1, syn0, syn1, size, indexes, work, i, j, k, cbow_mean)

    return work[0]

cdef void score_pair_cbow_hs(
    const np.uint32_t *word_point, const np.uint8_t *word_code, int codelens[MAX_SENTENCE_LEN],
    REAL_t *neu1, REAL_t *syn0, REAL_t *syn1, const int size,
    const np.uint32_t indexes[MAX_SENTENCE_LEN], REAL_t *work,
    int i, int j, int k, int cbow_mean) nogil:

    cdef long long a, b
    cdef long long row2
    cdef REAL_t f, g, count, inv_count, sgn
    cdef int m

    memset(neu1, 0, size * cython.sizeof(REAL_t))
    count = <REAL_t>0.0
    for m in range(j, k):
        if m == i or codelens[m] == 0:
            continue
        else:
            count += ONEF
            our_saxpy(&size, &ONEF, &syn0[indexes[m] * size], &ONE, neu1, &ONE)
    if count > (<REAL_t>0.5):
        inv_count = ONEF/count
    if cbow_mean:
        sscal(&size, &inv_count, neu1, &ONE)

    for b in range(codelens[i]):
        row2 = word_point[b] * size
        f = our_dot(&size, neu1, &ONE, &syn1[row2], &ONE)
        sgn = (-1)**word_code[b] # ch function: 0-> 1, 1 -> -1
        f *= sgn
        if f <= -MAX_EXP or f >= MAX_EXP:
            continue
        f = LOG_TABLE[<int>((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        work[0] += f


def init():
    """
    Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    into table EXP_TABLE.  Also calculate log(sigmoid(x)) into LOG_TABLE.

    """
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # Create vector with eps values.
    for i in range(MAX_EMB_SIZE):
        EPS_ARR[i] = EPS;

    # build the sigmoid table
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))
        LOG_TABLE[i] = <REAL_t>log( EXP_TABLE[i] )

    # build the trigonometric tables
    for i in range(COS_TABLE_SIZE):
        COS_TABLE[i] = <REAL_t>cos((i / <REAL_t>COS_TABLE_SIZE * 2 - 1) * MAX_COS)
        SIN_TABLE[i] = <REAL_t>sin((i / <REAL_t>COS_TABLE_SIZE * 2 - 1) * MAX_COS)

    # check whether sdot returns double or float
    d_res = dsdot(&size, x, &ONE, y, &ONE)
    p_res = <float *>&d_res
    if (abs(d_res - expected) < 0.0001):
        our_dot = our_dot_double
        our_saxpy = saxpy
        return 0  # double
    elif (abs(p_res[0] - expected) < 0.0001):
        our_dot = our_dot_float
        our_saxpy = saxpy
        return 1  # float
    else:
        # neither => use cython loops, no BLAS
        # actually, the BLAS is so messed up we'll probably have segfaulted above and never even reach here
        our_dot = our_dot_noblas
        our_saxpy = our_saxpy_noblas
        return 2

FAST_VERSION = init()  # initialize the module
MAX_WORDS_IN_BATCH = MAX_SENTENCE_LEN
