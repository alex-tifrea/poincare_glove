#!/usr/bin/env cython
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8

import cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np

# scipy <= 0.15
try:
    from scipy.linalg.blas import fblas
except ImportError:
    # in scipy > 0.15, fblas function has been removed
    import scipy.linalg.blas as fblas

from libc.stdio cimport printf, fopen, fclose, FILE, fflush, stdout, fread, fwrite, fseek, SEEK_SET
from libc.math cimport tanh, isnan, isinf, log, pow, fabs, acosh, sqrt, exp, cosh
from libc.stdlib cimport exit, malloc, free

cdef extern from "vfast_invsqrt.h":
    double fast_inv_sqrt(float number) nogil

REAL = np.float32

cdef scopy_ptr scopy=<scopy_ptr>PyCObject_AsVoidPtr(fblas.scopy._cpointer)  # y = x
cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(fblas.saxpy._cpointer)  # y += alpha * x
cdef sdot_ptr sdot=<sdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # float = dot(x, y)
cdef dsdot_ptr dsdot=<dsdot_ptr>PyCObject_AsVoidPtr(fblas.sdot._cpointer)  # double = dot(x, y)
cdef snrm2_ptr snrm2=<snrm2_ptr>PyCObject_AsVoidPtr(fblas.snrm2._cpointer)  # sqrt(x^2)
cdef sscal_ptr sscal=<sscal_ptr>PyCObject_AsVoidPtr(fblas.sscal._cpointer) # x = alpha * x (float)
cdef ssbmv_ptr ssbmv=<ssbmv_ptr>PyCObject_AsVoidPtr(fblas.ssbmv._cpointer) # y = alpha * A * x + beta * y (float)

DEF MAX_TANH_ARG = 15
DEF GRAD_CLAMP_THRESHOLD = 10
DEF TOTAL_NUM_PAIRS = 1409881283
DEF EPS = 0.00001
DEF BETA1 = 0.9
DEF BETA2 = 0.999
DEF SCALING_FACTOR_LR = 0.001

DEF EXP_TABLE_SIZE = 1000
DEF MAX_EXP = 6
DEF MAX_NN_NODES = 10000

cdef REAL_t X_MAX = <REAL_t>100.0
cdef REAL_t ALPHA = <REAL_t>0.75

cdef int ZERO = 0
cdef float ZEROF = <REAL_t>0.0
cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef REAL_t MINUS_ONEF = <REAL_t>-1.0

cdef enum DistFuncType:
    UNK_DIST_FUNC=0
    DIST=1  # use -dist as similarity during training; when used for mix-poincare it will use
            # the distance with sqrt to aggregate the small distances
    DIST_SQ=2  # use -dist^2 as similarity during training; when used for mix-poincare it will use
               # the distance w/o sqrt to aggregate the small embeddings
    COSH_DIST=3  # use -cosh(dist) as similarity function during training
    NN=4  # use -f_nn(dist) as similarity function during training, where nn is the scalar output of a neural network
    COSH_DIST_SQ=5  # use -cosh(dist)^2 as similarity function during training
    LOG_DIST_SQ=6  # use -log(dist^2 + 1) as similarity function during training
    COSH_DIST_POW_K=7  # use -cosh(dist)^k as similarity function during training

cdef enum EmbType:
    UNK_EMB=0
    VANILLA=1  # use dot product for training
    EUCLID=2  # use Euclidean distance for training
    POINCARE=3  # use Poincare distance for training
    MIX_POINCARE=4  # use mix-Poincare distance for training

cdef enum OptType:
    UNK_OPT=0
    ADAGRAD=1  # Euclidean ADAGRAD (like in vanilla GloVe)
    FullRSGD=2  # full (exact) Riemannian SGD that uses the closed form of the exponential map in the Poincare ball
    WeightedFullRSGD=3  # full (exact) RSGD where the lr is weighted by a function that depends on the frequency of the word whose embedding is being updated
    RADAGRAD=4  # Riemannian ADAGRAD
    MixRADAGRAD=5  # Riemanninan ADAGRAD on the carthesian product of small-dimensional hyperbolic spaces
    RAMSGRAD=6  # Riemannian AMSgrad

cdef enum PositiveNonlinearity:
    UNK_NONLIN=0
    RELU=1
    SIGMOID=2
    ID=3

cdef struct NNConfig:
    int num_nodes
    PositiveNonlinearity nonlinearity
    # weights[0*num_nodes] = biases; weights[1*num_nodes] = hidden layer; weights[2*num_nodes] = output layer
    REAL_t *weights
    REAL_t *output_bias  # scalar that is the output bias

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
cdef void clamp(float *v, int size, float _min, float _max) nogil:
    for i in range(size):
        v[i] = max(_min, min(v[i], _max))

cdef float clamp_scalar(float v, float _min, float _max) nogil:
    return max(_min, min(v, _max))


# Mitigate potential numerical errors
cdef inline int is_number(const REAL_t x) nogil:
    return 0 if isnan(x) or isinf(x) else 1

cdef float our_tanh(const float x) nogil:
    return tanh(min(max(x, -MAX_TANH_ARG), MAX_TANH_ARG))

cdef float vec_max(const int size, float *v) nogil:
    cdef int i
    cdef float max = -100000.0
    for i in range(size):
        if max < v[i]:
            max = v[i]
    return max

cdef float vec_min(const int size, float *v) nogil:
    cdef int i
    cdef float min = 100000.0
    for i in range(size):
        if min > v[i]:
            min = v[i]
    return min


# Will output 3 values:
# similarity    = dot product-based similarity between two vectors, v and w, defined as -dist_func(v \dot w)
# dS_dv         = derivative of the similarity wrt v
# dS_dw         = derivative of the similarity wrt w
cdef void vanilla_similarity(REAL_t *similarity, REAL_t *dS_dv, REAL_t *dS_dw, REAL_t *v, REAL_t *w,
                             const int size, const DistFuncType dist_func,
                             const NNConfig nn, REAL_t *grad_dS_dweights) nogil:

    cdef double dot_prod
    cdef REAL_t grad_dnn_dx

    if dist_func == DistFuncType.NN:
        # Similarity is -f(v \dot w).
        dot_prod = our_dot(&size, v, &ONE, w, &ONE)
        f_nn(dot_prod, nn, similarity, &grad_dnn_dx, grad_dS_dweights)

        # Account for the minus in front of the f(v \dot w).
        similarity[0] = -similarity[0]
        grad_dnn_dx = -grad_dnn_dx
        sscal(&size, &MINUS_ONEF, grad_dS_dweights, &ONE)
    else:
        similarity[0] = our_dot(&size, v, &ONE, w, &ONE)

    scopy(&size, w, &ONE, dS_dv, &ONE)
    scopy(&size, v, &ONE, dS_dw, &ONE)
    if dist_func == DistFuncType.NN:
        # Multiply by the gradient of the neural net wrt its input
        sscal(&size, &grad_dnn_dx, dS_dv, &ONE)
        sscal(&size, &grad_dnn_dx, dS_dw, &ONE)


# Will output 3 values:
# similarity    = Euclidean distance-based similarity between two vectors, v and w, defined as -dist_func(euclid_dist(v,w))
# dS_dv         = derivative of the similarity wrt v
# dS_dw         = derivative of the similarity wrt w
cdef void euclidean_similarity(REAL_t *similarity, REAL_t *dS_dv, REAL_t *dS_dw, REAL_t *v, REAL_t *w,
                              const int size, const DistFuncType dist_func) nogil:

    cdef REAL_t diff[size]

    # Compute Euclidean distance between vectors.
    # diff = v - w
    scopy(&size, v, &ONE, diff, &ONE)
    our_saxpy(&size, &MINUS_ONEF, w, &ONE, diff, &ONE)


    if dist_func == DistFuncType.DIST_SQ:
        # Similarity is -0.5 * dist(v, w).
        similarity[0] = -0.5 * our_dot(&size, diff, &ONE, diff, &ONE)

    if dist_func == DistFuncType.DIST_SQ:
        # dS_dv = -diff = w - v
        scopy(&size, diff, &ONE, dS_dv, &ONE)
        sscal(&size, &MINUS_ONEF, dS_dv, &ONE)
        # dS_dw = diff = v - w
        scopy(&size, diff, &ONE, dS_dw, &ONE)


cdef inline double relu(double x, REAL_t *grad) nogil:
    if x > 0.0:
        grad[0] = 1.0
        return x
    grad[0] = 0.0
    return 0.0


cdef inline double sigmoid(double x, REAL_t *grad) nogil:
    cdef double result

    if x >= MAX_EXP:
        if grad != NULL:
            grad[0] = 0.0
        return 1.0
    elif x <= -MAX_EXP:
        if grad != NULL:
            grad[0] = 0.0
        return 0.0
    result = EXP_TABLE[<int> x]
    if grad != NULL:
        grad[0] = result * (1 - result)

    return result


# Neural network used to learn the function through which we pass the distance to match it with the log-co-occ. counts.
cdef void f_nn(const double x, const NNConfig nn, REAL_t *result,
               REAL_t *grad_do_dx, REAL_t *grad_do_dweights) nogil:
    cdef int i
    cdef REAL_t a, alpha, grad_relu_dw1, grad_relu_dw2, pos_w1, pos_w2

    grad_do_dx[0] = 0.0
    result[0] = 0.0
    for i in range(nn.num_nodes):
        if nn.nonlinearity == PositiveNonlinearity.RELU:
            pos_w1 = relu(nn.weights[1*nn.num_nodes + i], &grad_relu_dw1)
            pos_w2 = relu(nn.weights[2*nn.num_nodes + i], &grad_relu_dw2)
        elif nn.nonlinearity == PositiveNonlinearity.SIGMOID:
            pos_w1 = sigmoid(nn.weights[1*nn.num_nodes + i], &grad_relu_dw1)
            pos_w2 = sigmoid(nn.weights[2*nn.num_nodes + i], &grad_relu_dw2)
        elif nn.nonlinearity == PositiveNonlinearity.ID:
            pos_w1 = nn.weights[1*nn.num_nodes + i]
            pos_w2 = nn.weights[2*nn.num_nodes + i]
            grad_relu_dw1 = 1.0
            grad_relu_dw2 = 1.0
        a = sigmoid(pos_w1 * x + nn.weights[0*nn.num_nodes + i], NULL)
        alpha = pos_w2 * a * (1 - a)

        result[0] = result[0] + pos_w2 * a
        grad_do_dx[0] = grad_do_dx[0] + pos_w1 * alpha
        grad_do_dweights[2*nn.num_nodes + i] = alpha  # grad wrt to bias1

        grad_do_dweights[0*nn.num_nodes + i] = alpha * x * grad_relu_dw1  # grad wrt hidden unit weight
        grad_do_dweights[1*nn.num_nodes + i] = a * grad_relu_dw2  # grad wrt output unit weight

    result[0] = result[0] + nn.output_bias[0]


# Will output 3 values:
# similarity    = Poincare similarity between two vectors in the carthesian product of hyperbolic spaces, v and w,
#                 defined as sqrt(sum_of_i poincare_dist(v_i, w_i)^2), where v_i and w_i are the ith pair of small-dimensional embeddings
# gradR_dS_dv   = Riemannian derivatives of the similarity wrt v (i.e. Euclidean derivative / conformal factor at v),
#                 for all the small v
# gradR_dS_dw   = Riemannian derivatives of the similarity wrt w (i.e. Euclidean derivative / conformal factor at w),
#                 for all the small w
cdef void mix_poincare_similarity(REAL_t *similarity, REAL_t *gradR_dS_dv, REAL_t *gradR_dS_dw, REAL_t *v, REAL_t *w,
                              const int size, const int num_embs, const DistFuncType dist_func,
                              const REAL_t cosh_dist_pow, int *num_projections) nogil:

    cdef int i, index, small_emb_size = size / num_embs
    cdef REAL_t dist_sum = 0.0, curr_dist_sq, coef
    cdef DistFuncType small_emb_dist_func = DistFuncType.DIST_SQ

    for i in range(0,num_embs):
        index = small_emb_size * i
        poincare_similarity(&curr_dist_sq, &gradR_dS_dv[index], &gradR_dS_dw[index], &v[index], &w[index],
                            small_emb_size, small_emb_dist_func, cosh_dist_pow, num_projections)

        # Accumulate sum of distances squared.
        dist_sum = dist_sum + (-curr_dist_sq)

    if dist_func == DistFuncType.DIST:
        similarity[0] = -sqrt(dist_sum)

        # Multiply each individual gradient by the contribution of the gradient of sqrt.
        coef = 1.0 / 2.0 / (-similarity[0])
        sscal(&size, &coef, gradR_dS_dv, &ONE)
        sscal(&size, &coef, gradR_dS_dw, &ONE)
    elif dist_func == DistFuncType.DIST_SQ:
        similarity[0] = -dist_sum
    elif dist_func == DistFuncType.COSH_DIST_SQ:
        coef = cosh(sqrt(dist_sum))
        similarity[0] = -coef * coef

        # Multiply each individual gradient by the contribution of the gradient of cosh(sqrt(dist))^2.
        coef = coef * fast_inv_sqrt(dist_sum / (coef * coef - 1))
        sscal(&size, &coef, gradR_dS_dv, &ONE)
        sscal(&size, &coef, gradR_dS_dw, &ONE)


# Will output 3 values:
# similarity    = Poincare similarity between two vectors in hyperbolic space, v and w, defined as -dist_func(poincare_dist(v,w))
# gradR_dS_dv   = Riemannian derivative of the similarity wrt v (i.e. Euclidean derivative / conformal factor at v)
# gradR_dS_dw   = Riemannian derivative of the similarity wrt w (i.e. Euclidean derivative / conformal factor at w)
cdef void poincare_similarity(REAL_t *similarity, REAL_t *gradR_dS_dv, REAL_t *gradR_dS_dw, REAL_t *v, REAL_t *w,
                              const int size, const DistFuncType dist_func,
                              const REAL_t cosh_dist_pow, int *num_projections) nogil:
    cdef int start
    cdef double dot_vv, dot_ww
    cdef REAL_t g, alpha_v, beta_w, coef, p_dist, dS_dg, factor, squared_dist
    cdef REAL_t threshold = 1-EPS, norm_factor, emb_norm
    cdef REAL_t diff[size]

    emb_norm = snrm2(&size, v, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        sscal(&size, &norm_factor, v, &ONE)
        dot_vv = threshold * threshold
        if num_projections != NULL:
            num_projections[0] = num_projections[0] + 1
    else:
        dot_vv = our_dot(&size, v, &ONE, v, &ONE)

    emb_norm = snrm2(&size, w, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        sscal(&size, &norm_factor, w, &ONE)
        dot_ww = threshold * threshold
        if num_projections != NULL:
            num_projections[0] = num_projections[0] + 1
    else:
        dot_ww = our_dot(&size, w, &ONE, w, &ONE)

    # Sanity check
    if dot_vv >= 1 or dot_ww >= 1 or isnan(dot_vv) or isnan(dot_ww):
        printf("[%d, %d] Cannot compute Poincare distance between points. Points need to be inside the unit ball, but their squared norm is %f and %f.\n", v, w, dot_vv, dot_ww)
        exit(-1)

    scopy(&size, v, &ONE, diff, &ONE)
    our_saxpy(&size, &MINUS_ONEF, w, &ONE, diff, &ONE)
    squared_dist = our_dot(&size, diff, &ONE, diff, &ONE)

    alpha_v = 1 - dot_vv
    beta_w = 1 - dot_ww
    g = 1 + 2 * squared_dist / (alpha_v * beta_w + EPS*EPS)

    if dist_func == DistFuncType.DIST_SQ:
        # Similarity is -poincare_dist(v, w)^2.
        p_dist = acosh(g)
        similarity[0] = -p_dist * p_dist
    elif dist_func == DistFuncType.DIST:
        # Similarity is -poincare_dist(v, w).
        p_dist = acosh(g)
        similarity[0] = -p_dist
    elif dist_func == DistFuncType.COSH_DIST:
        # Similarity is -cosh(poincare_dist(v, w)).
        p_dist = g
        similarity[0] = -p_dist
    elif dist_func == DistFuncType.COSH_DIST_SQ:
        # Similarity is -cosh(poincare_dist(v, w))^2.
        p_dist = g
        similarity[0] = -p_dist * p_dist
    elif dist_func == DistFuncType.COSH_DIST_POW_K:
        # Similarity is -cosh(poincare_dist(v, w))^k.
        p_dist = pow(g, cosh_dist_pow)
        similarity[0] = -p_dist
    elif dist_func == DistFuncType.LOG_DIST_SQ:
        # Similarity is -log(poincare_dist(v, w)^2 + 1).
        p_dist = acosh(g)
        similarity[0] = -log(p_dist * p_dist + 1)

    if isnan(similarity[0]):
        printf("[%f %f], [%f %f]\n", v[0], v[1], w[0], w[1])
        printf("g=%f dot_vv=%f dot_ww=%f squared_dist=%f\n", g, dot_vv, dot_ww, squared_dist)

    # Derivative of similarity wrt g.
    if dist_func == DistFuncType.DIST_SQ:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -2.0 * p_dist * fast_inv_sqrt(g * g - 1)
        else:
            # dS_dg is -2 for lim g->1
            dS_dg = -2.0
    elif dist_func == DistFuncType.DIST:
        dS_dg = -fast_inv_sqrt(g * g - 1)
    elif dist_func == DistFuncType.COSH_DIST:
        dS_dg = -1.0
    elif dist_func == DistFuncType.COSH_DIST_SQ:
        dS_dg = -2.0 * p_dist
    elif dist_func == DistFuncType.COSH_DIST_POW_K:
        dS_dg = -cosh_dist_pow * p_dist / g
    elif dist_func == DistFuncType.LOG_DIST_SQ:
        if fabs(g - 1.0) > 0.00001:
            dS_dg = -2.0 * p_dist / sqrt(g * g - 1) / (p_dist * p_dist + 1)
        else:
            # dS_dg is -2 for lim g->1
            dS_dg = -2.0
    else:
        printf("Unsupported distance function used.\n")
        exit(-1)

    # Compute derivative wrt v multiplied by the inverse of the conformal factor at v.
    factor = dS_dg * alpha_v / beta_w
    start = 1
    # Compute result[1] = coef1 * v + coef2 * w
    scopy(&size, v, &ONE, gradR_dS_dv, &ONE)
    coef = (1.0 + squared_dist / alpha_v) * factor
    sscal(&size, &coef, gradR_dS_dv, &ONE)
    coef = -factor
    saxpy(&size, &coef, w, &ONE, gradR_dS_dv, &ONE)
    clamp(gradR_dS_dv, size, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)

    # Compute derivative wrt w multiplied by the inverse of the conformal factor at w.
    factor = dS_dg * beta_w / alpha_v
    start = 1 + size
    # Compute result[1+size] = coef1 * w + coef2 * v
    scopy(&size, w, &ONE, gradR_dS_dw, &ONE)
    coef = (1.0 + squared_dist / beta_w) * factor
    sscal(&size, &coef, gradR_dS_dw, &ONE)
    coef = -factor
    saxpy(&size, &coef, v, &ONE, gradR_dS_dw, &ONE)
    clamp(gradR_dS_dw, size, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)


# Weight function used by glove to weigh the contribution of each co-occ pair.
cdef inline REAL_t weight_func(const REAL_t x) nogil:
    return 1.0 if x > X_MAX else pow(x / X_MAX, ALPHA)

# Update embedding using Euclidean AdaGrad.
cdef inline void update_embedding(const int size, REAL_t *emb, REAL_t lr, REAL_t *grad, REAL_t *gradsq) nogil:
    cdef int i

    for i in range(size):
        emb[i] = emb[i] - lr * grad[i] * fast_inv_sqrt(gradsq[i])
        gradsq[i] = gradsq[i] + grad[i] * grad[i]

# Update a Poincare embedding using the full RSGD:
#       emb = exp_emb(-lr * riemannian_grad)
# Parameters:
#       size : scalar
#           embedding size
#       emb : array
#           pointer to the beginning of the embedding vector
#       grad : array
#           Riemannian gradient of loss wrt emb
cdef inline void update_embedding_full_rsgd(const int size, REAL_t *emb, REAL_t lr, REAL_t *grad) nogil:
    cdef REAL_t emb_norm, v_norm, dot_emb_emb, dot_emb_v, dot_vv, alpha, gamma, denominator, coef, threshold = 1.0 - EPS, norm_factor

    v_norm = lr * snrm2(&size, grad, &ONE) + 1e-15
    dot_emb_v = -lr * our_dot(&size, emb, &ONE, grad, &ONE)
    dot_emb_emb = our_dot(&size, emb, &ONE, emb, &ONE)
    dot_vv = lr * lr * our_dot(&size, grad, &ONE, grad, &ONE)
    alpha = our_tanh(v_norm / (1 - dot_emb_emb)) / v_norm
    gamma = alpha * alpha * dot_vv
    denominator = 1 + 2 * alpha * dot_emb_v + dot_emb_emb * gamma + 1e-15
    coef = (1 + 2 * alpha * dot_emb_v + gamma) / denominator
    sscal(&size, &coef, emb, &ONE)

    coef = (1 - dot_emb_emb) * alpha * (-lr) / denominator
    our_saxpy(&size, &coef, grad, &ONE, emb, &ONE)

    # Project back onto the ball.
    emb_norm = snrm2(&size, emb, &ONE)
    if emb_norm > threshold:
        norm_factor = threshold / emb_norm
        sscal(&size, &norm_factor, emb, &ONE)

# Update embedding using RAMSGrad. In gradsq we store the accumulated squared gradients. We only have one scalar per
# embedding, so, for the ith embedding we only use gradsq[i*size], not the whole gradsq[i*size : (i+1)*size]
# Similarly, in mom we store the momentum (one per embedding) and in betas we store beta1 and beta2 in betas[0] and betas[1] respectively.
cdef inline void update_embedding_ramsgrad(const int size, REAL_t *emb, REAL_t lr, REAL_t *grad,
                                           REAL_t *gradsq, REAL_t *mom, REAL_t *betas) nogil:
    cdef REAL_t adapted_lr, coef, mom_norm, norm_factor, threshold = 1-EPS

    # Update accumulated betas.
    betas[0] = 0.0  # betas[0] * BETA1
    betas[1] = 0.0  # betas[1] * BETA2

    # Update momentum.
    coef = BETA1 / (1 - betas[0])
    sscal(&size, &coef, mom, &ONE)
    coef = (1 - BETA1) / (1 - betas[0])
    our_saxpy(&size, &coef, grad, &ONE, mom, &ONE)

    # Update the sum of the past squared gradients.
    gradsq[0] = max(gradsq[0],
                    (BETA2 * gradsq[0] + (1 - BETA2) * our_dot(&size, grad, &ONE, grad, &ONE) / size) / (1 - betas[1]))

    adapted_lr = lr * fast_inv_sqrt(gradsq[0])

    # Perform RSGD, using the exponential map.
    update_embedding_full_rsgd(size, emb, adapted_lr, mom)

    # Project momentum onto the ball.
    mom_norm = snrm2(&size, mom, &ONE)
    if mom_norm > threshold:
        norm_factor = threshold / mom_norm
        sscal(&size, &norm_factor, mom, &ONE)

# Update embedding using RAdaGrad. In gradsq we store the accumulated squared gradients. We only have one scalar per
# embedding, so, for the ith embedding we only use gradsq[i*size], not the whole gradsq[i*size : (i+1)*size]
cdef inline void update_embedding_radagrad(const int size, REAL_t *emb, REAL_t lr, REAL_t *grad, REAL_t *gradsq) nogil:
    cdef REAL_t adapted_lr = lr * fast_inv_sqrt(gradsq[0])

    # Update the sum of the past squared gradients.
    gradsq[0] = gradsq[0] + our_dot(&size, grad, &ONE, grad, &ONE) / size

    # Perform RSGD, using the exponential map.
    update_embedding_full_rsgd(size, emb, adapted_lr, grad)


# Update embeddings using RAdaGrad for the case when we use a Cartesian product of Poincare embeddings.
cdef inline void update_embedding_mix_radagrad(const int size, const int num_embs, REAL_t *emb, REAL_t lr, REAL_t *grad, REAL_t *gradsq) nogil:
    cdef REAL_t adapted_lr
    cdef int i, small_emb_size = size / num_embs, index

    clamp(grad, size, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)

    # Update each of the small embeddings separately.
    for i in range(0, num_embs):
        index = small_emb_size * i
        adapted_lr = lr * fast_inv_sqrt(gradsq[index])

        # Update the sum of the past squared gradients.
        gradsq[index] = gradsq[index] + our_dot(&small_emb_size, &grad[index], &ONE, &grad[index], &ONE) / small_emb_size

        # Perform RSGD, using the exponential map.
        update_embedding_full_rsgd(small_emb_size, &emb[index], adapted_lr, &grad[index])


cdef inline void get_similarity_and_gradients(const EmbType emb_type, const int size, const int num_embs, REAL_t *v, REAL_t *w,
                                              float *similarity, REAL_t *dS_dv, REAL_t *dS_dw,
                                              const DistFuncType dist_func, int *num_projections, const int cosh_dist_pow,
                                              const NNConfig nn, REAL_t *grad_dS_dweights) nogil:

    cdef int small_emb_size

    if emb_type == EmbType.VANILLA:
        vanilla_similarity(similarity, dS_dv, dS_dw, v, w, size, dist_func, nn, grad_dS_dweights)
    elif emb_type == EmbType.EUCLID:
        euclidean_similarity(similarity, dS_dv, dS_dw, v, w, size, dist_func)
    elif emb_type == EmbType.POINCARE or emb_type == EmbType.MIX_POINCARE:
        if num_embs > 0:
            small_emb_size = size / num_embs
            # In dS_dv and dS_dw we'll store the Riemannian gradient wrt v and w for all the small-dimensional embeddings.
            mix_poincare_similarity(similarity, dS_dv, dS_dw, v, w, size, num_embs, dist_func, 0, num_projections)
        else:
            # In dS_dv and dS_dw we'll compute the Riemannian gradient wrt v and w.
            poincare_similarity(similarity, dS_dv, dS_dw, v, w, size, dist_func, cosh_dist_pow, num_projections)
    else:
        printf("Unrecognized embedding type.")
        exit(-1)


# Process the list of triples concurrently, using `prange`.
cdef void fast_glove(
        const EmbType emb_type, const int use_glove_format, double (*coocc_func)(double) nogil, const DistFuncType dist_func, const NNConfig nn,
        const int use_scaling, REAL_t *scaling_factor, const char *coocc_file,
        REAL_t *syn0, REAL_t *syn1, REAL_t *b0, REAL_t *b1,
        REAL_t *gradsq_syn0, REAL_t *gradsq_syn1, REAL_t *gradsq_b0, REAL_t *gradsq_b1,
        REAL_t *mom_syn0, REAL_t *mom_syn1, REAL_t *mom_b0, REAL_t *mom_b1,
        REAL_t *betas_syn0, REAL_t *betas_syn1, REAL_t *betas_b0, REAL_t *betas_b1,
        const int num_workers, const int chunksize, const int size, const int num_embs, const int num_pairs, const int max_word_index, const int cosh_dist_pow,
        const OptType optimizer, const REAL_t lr, const int *index2freq, const int _compute_loss,
        const int with_bias, const int use_log_probs, REAL_t *_total_training_loss, int *_trained_pair_count,
        int *num_projections):

    cdef REAL_t diff, fdiff, coef, loss = 0.0, adaptive_lr, old_mean_bias, old_var_bias
    cdef float similarity, occ_count, ground_truth
    cdef REAL_t *grad0, *grad1, *syn0_copy, *syn1_copy, *grad_dS_dweights
    cdef int w1_index, w2_index, i, j, pair_count = 0
    cdef int row1, row2, threadid
    cdef FILE *fp
    cdef size_t read_size
    cdef int small_emb_size
    cdef REAL_t beta2 = 0.999, acc_beta2 = 1.0

    if num_embs > 0:
        small_emb_size = size / num_embs

    with nogil, parallel(num_threads=num_workers):

        # Thread-local variables.
        fp = fopen(coocc_file, "r")
        grad0 = <REAL_t *> malloc(cython.sizeof(REAL_t) * size)
        grad1 = <REAL_t *> malloc(cython.sizeof(REAL_t) * size)
        syn0_copy = <REAL_t *> malloc(cython.sizeof(REAL_t) * size)
        syn1_copy = <REAL_t *> malloc(cython.sizeof(REAL_t) * size)
        grad_dS_dweights = <REAL_t *> malloc(cython.sizeof(REAL_t) * nn.num_nodes * 3)
        similarity = 0.0
        w1_index = 0
        w2_index = 0
        occ_count = 0.0
        threadid = cython.parallel.threadid()

        for i in prange(num_pairs, schedule="static", chunksize=chunksize):
            read_size = read_triple(fp, use_glove_format, threadid, i, &w1_index, &w2_index, &occ_count)

            if w1_index >= max_word_index or w2_index >= max_word_index:
                continue

            if read_size <= 0:
                printf("Exiting because read operation returned %ld bytes\n", read_size)
                fflush(stdout)
                exit(-1)

            row1 = w1_index * size
            row2 = w2_index * size

            scopy(&size, &syn0[row1], &ONE, syn0_copy, &ONE)
            scopy(&size, &syn1[row2], &ONE, syn1_copy, &ONE)

            # Compute similarity and gradients of similarity wrt to the two vectors (i.e. syn0[w1], syn1[w2]), dS_dv and dS_dw.
            get_similarity_and_gradients(emb_type, size, num_embs, syn0_copy, syn1_copy, &similarity, grad0, grad1, dist_func,
                                         num_projections, cosh_dist_pow, nn, grad_dS_dweights)

            if use_log_probs == 1:
                ground_truth = coocc_func(occ_count / (index2freq[w1_index] * index2freq[w2_index]) + 0.01)
            else:
                ground_truth = coocc_func(occ_count)

            # Compute difference between log co-occ count and model.
            if with_bias == 1:
                diff = scaling_factor[0] * similarity + b0[w1_index] + b1[w2_index] - ground_truth
            else:
                # Add the global mean bias and divide by the global variance bias
                diff = scaling_factor[0] * (similarity - b0[0]) / b1[0] - ground_truth
                # TODO: gradient needs to be changed so that dS_dv accounts for 1/b1[0]
            fdiff = weight_func(occ_count) * diff

            if is_number(diff) == 0 or is_number(fdiff) == 0:  # or diff > 10000000:
                printf("Encountered invalid number. w1=%d w2=%d occ_count=%f occ_prob=%f; sim=%f ground_truth=%f; max_syn1=%f min_syn1=%f; syn0_norm=%f syn1_norm%f; b0=%f b1=%f; diff=%f\n",
                       w1_index, w2_index, occ_count, occ_count / (index2freq[w1_index] * index2freq[w2_index]),
                       similarity, ground_truth, vec_max(size, syn1_copy), vec_min(size, syn1_copy),
                       snrm2(&size, syn0_copy, &ONE), snrm2(&size, syn1_copy, &ONE),
                       (b0[w1_index] if with_bias == 1 else b0[0]), (b1[w2_index] if with_bias == 1 else b1[0]), diff)
                return

            if _compute_loss:
                # Accumulate loss.
                # Reduction variable.
                loss += 0.5 * fdiff * diff

            # Compute gradients. Multiply dl_dS (i.e. fdiff) with dS_dv and dS_dw respectively.
            coef = fdiff * scaling_factor[0]
            sscal(&size, &coef, grad0, &ONE)
            sscal(&size, &coef, grad1, &ONE)

            # Update embeddings.
            if optimizer == OptType.FullRSGD:
                update_embedding_full_rsgd(size, syn0_copy, lr, grad0)
                update_embedding_full_rsgd(size, syn1_copy, lr, grad1)
            elif optimizer == OptType.WeightedFullRSGD:
                adaptive_lr = lr * fast_inv_sqrt(index2freq[w1_index] / 100.0)
                update_embedding_full_rsgd(size, syn0_copy, adaptive_lr, grad0)
                adaptive_lr = lr * fast_inv_sqrt(index2freq[w2_index] / 100.0)
                update_embedding_full_rsgd(size, syn1_copy, adaptive_lr, grad1)
            elif optimizer == OptType.RAMSGRAD:
                # Riemannian AMSGrad update.
                update_embedding_ramsgrad(size, syn0_copy, lr, grad0, &gradsq_syn0[row1], &mom_syn0[row1], &betas_syn0[row1])
                update_embedding_ramsgrad(size, syn1_copy, lr, grad1, &gradsq_syn1[row2], &mom_syn1[row2], &betas_syn1[row2])
            elif optimizer == OptType.RADAGRAD:
                # Riemannian AdaGrad update.
                update_embedding_radagrad(size, syn0_copy, lr, grad0, &gradsq_syn0[row1])
                update_embedding_radagrad(size, syn1_copy, lr, grad1, &gradsq_syn1[row2])
            elif optimizer == OptType.MixRADAGRAD:
                # Riemannian AdaGrad update on carthesian product of small-dimensional spaces.
                update_embedding_mix_radagrad(size, num_embs, syn0_copy, lr, grad0, &gradsq_syn0[row1])
                update_embedding_mix_radagrad(size, num_embs, syn1_copy, lr, grad1, &gradsq_syn1[row2])
            elif optimizer == OptType.ADAGRAD:
                # AdaGrad updates.
                update_embedding(size, syn0_copy, lr, grad0, &gradsq_syn0[row1])
                update_embedding(size, syn1_copy, lr, grad1, &gradsq_syn1[row2])

            # Store new updated embeddings in the global storage.
            scopy(&size, syn0_copy, &ONE, &syn0[row1], &ONE)
            scopy(&size, syn1_copy, &ONE, &syn1[row2], &ONE)

            # Update biases.
            if with_bias == 1:
                # Always use AdaGrad to update the biases.
                coef = clamp_scalar(fdiff, -GRAD_CLAMP_THRESHOLD, GRAD_CLAMP_THRESHOLD)
                b0[w1_index] = b0[w1_index] - lr * fast_inv_sqrt(gradsq_b0[w1_index]) * coef
                b1[w2_index] = b1[w2_index] - lr * fast_inv_sqrt(gradsq_b1[w2_index]) * coef
                gradsq_b0[w1_index] = gradsq_b0[w1_index] + coef * coef * lr * lr
                gradsq_b1[w2_index] = gradsq_b1[w2_index] + coef * coef * lr * lr
            else:
                # Always use AdaGrad to update the global biases.
                old_mean_bias = b0[0]
                old_var_bias = b1[0]
                coef = lr * fdiff * (-1.0 / old_var_bias)
                b0[0] = b0[0] - fast_inv_sqrt(gradsq_b0[0]) * coef
                gradsq_b0[0] = gradsq_b0[0] + coef * coef
                coef = lr * fdiff * (-(similarity - old_mean_bias) / (old_var_bias * old_var_bias))
                b1[0] = b1[0] - fast_inv_sqrt(gradsq_b1[0]) * coef
                gradsq_b1[0] = gradsq_b1[0] + coef * coef

            # Update NN weights, if needed.
            if dist_func == DistFuncType.NN:
                # Multiply by dl_dnn
                coef = -lr * fdiff
                our_saxpy(&nn.num_nodes, &coef, &grad_dS_dweights[0*nn.num_nodes], &ONE, &nn.weights[0*nn.num_nodes], &ONE)
                our_saxpy(&nn.num_nodes, &coef, &grad_dS_dweights[1*nn.num_nodes], &ONE, &nn.weights[1*nn.num_nodes], &ONE)
                our_saxpy(&nn.num_nodes, &coef, &grad_dS_dweights[2*nn.num_nodes], &ONE, &nn.weights[2*nn.num_nodes], &ONE)
                nn.output_bias[0] = nn.output_bias[0] + coef * (-1.0)  # Account for the minus in S(v,w) = -f(dist(v,w))

            # Update scaling factor.
            if use_scaling == 1:
                scaling_factor[0] = scaling_factor[0] - SCALING_FACTOR_LR * fdiff * similarity

            # Reduction variable.
            pair_count += 1

        # Free dynamically allocated memory.
        free(grad0)
        free(grad1)

        fclose(fp)

    _trained_pair_count[0] = pair_count
    _total_training_loss[0] = loss


def train_glove_epoch(model):
    cdef int euclid = model.get_attr('euclid', 0)
    cdef int poincare = model.get_attr('poincare', 0)
    cdef int with_bias = (1 if model.with_bias == True else 0)
    cdef int use_log_probs= (1 if model.use_log_probs == True else 0)
    cdef int debug = (1 if model.get_attr('debug', False) else 0)
    cdef int use_glove_format = (1 if model.use_glove_format == True else 0)

    cdef int _compute_loss = (1 if model.compute_loss == True else 0)
    cdef int size = model.vector_size
    cdef int num_embs = model.num_embs
    cdef int num_pairs = model.num_pairs
    cdef int max_word_index = model.max_word_index
    cdef int num_workers = model.num_workers
    cdef int chunksize = model.chunksize
    cdef int cosh_dist_pow = 0

    cdef REAL_t lr = model.lr

    cdef bytes py_bytes = model.coocc_file.encode()
    cdef char *coocc_file_path = py_bytes

    cdef REAL_t *syn0 = <REAL_t *>(np.PyArray_DATA(model.wv.vectors))
    cdef REAL_t *syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.syn1))
    cdef REAL_t *b0 = NULL, *b1 = NULL  # biases
    cdef REAL_t *gradsq_syn0 = NULL, *gradsq_syn1 = NULL, *gradsq_b0 = NULL, *gradsq_b1 = NULL  # arrays used by AdaGrad
    cdef REAL_t *mom_syn0 = NULL, *mom_syn1 = NULL, *mom_b0 = NULL, *mom_b1 = NULL  # arrays used by AMSgrad
    cdef REAL_t *betas_syn0 = NULL, *betas_syn1 = NULL, *betas_b0 = NULL, *betas_b1 = NULL  # arrays used by AMSgrad

    cdef int *index2freq = <int*>(np.PyArray_DATA(model.wv.index2freq))

    cdef float epoch_loss = 0.0
    cdef int trained_pair_count = 0, num_projections = 0
    cdef int i

    cdef EmbType emb_type = EmbType.UNK_EMB

    cdef str opt_str = model.get_attr('optimizer', 'fullrsgd' if poincare == 1 else 'adagrad')
    cdef OptType optimizer = OptType.UNK_OPT

    cdef str dist_func_str = model.get_attr('dist_func', '')
    cdef DistFuncType dist_func = DistFuncType.UNK_DIST_FUNC
    cdef NNConfig nn

    cdef str coocc_func_str = model.get_attr('coocc_func', 'log')
    cdef double (*coocc_func)(double) nogil

    cdef REAL_t scaling_factor = model.get_attr('scaling_factor', 1.0)
    cdef int use_scaling = 1 if model.get_attr('use_scaling', False) else 0

    if use_scaling == 1:
        printf("Scaling LR is %f\n", SCALING_FACTOR_LR)

    # Convert co-occurrence function from string.
    if coocc_func_str == "log":
        coocc_func = log
    else:
        print("Unrecognized similarity function type", coocc_func_str)
        exit(-1)

    if euclid == 1:
        emb_type = EmbType.EUCLID
    elif poincare == 1:
        if num_embs > 0:
            emb_type = EmbType.MIX_POINCARE
        else:
            emb_type = EmbType.POINCARE
    else:
        emb_type = EmbType.VANILLA

    if emb_type == EmbType.VANILLA:
        # Convert distance function from string to one of the enum values.
        if dist_func_str == "nn":
            dist_func = DistFuncType.NN
            nn.num_nodes = model.nn_config.num_nodes
            nn.weights = <REAL_t *>(np.PyArray_DATA(model.trainables.nn_weights))
            nn.output_bias = <REAL_t*>(np.PyArray_DATA(model.trainables.nn_output_bias))
            if model.nn_config.nonlinearity == "relu":
                nn.nonlinearity = PositiveNonlinearity.RELU
            elif model.nn_config.nonlinearity == "sigmoid":
                nn.nonlinearity = PositiveNonlinearity.SIGMOID
            elif model.nn_config.nonlinearity == "id":
                nn.nonlinearity = PositiveNonlinearity.ID

    if emb_type == EmbType.EUCLID:
        # Convert distance function from string to one of the enum values.
        if dist_func_str == "dist":
            dist_func = DistFuncType.DIST
        elif dist_func_str == "dist-sq":
            dist_func = DistFuncType.DIST_SQ
        else:
            print("Unrecognized distance function type", dist_func_str, "for euclid")
            exit(-1)

    if emb_type == EmbType.POINCARE:
        # Convert distance function from string to one of the enum values.
        if dist_func_str == "dist-sq":
            dist_func = DistFuncType.DIST_SQ
        elif dist_func_str == "dist":
            dist_func = DistFuncType.DIST
        elif dist_func_str == "cosh-dist":
            dist_func = DistFuncType.COSH_DIST
        elif dist_func_str == "cosh-dist-sq":
            dist_func = DistFuncType.COSH_DIST_SQ
        elif dist_func_str == "log-dist-sq":
            dist_func = DistFuncType.LOG_DIST_SQ
        elif "cosh-dist-pow-" in dist_func_str:
            dist_func = DistFuncType.COSH_DIST_POW_K
            cosh_dist_pow = model.cosh_dist_pow
        else:
            print("Unrecognized distance function type", dist_func_str, "for poincare")
            exit(-1)

    if emb_type == EmbType.MIX_POINCARE:
        # Convert distance function from string to one of the enum values.
        if dist_func_str == "dist-sq":
            dist_func = DistFuncType.DIST_SQ
        elif dist_func_str == "dist":
            dist_func = DistFuncType.DIST
        elif dist_func_str == "cosh-dist-sq":
            dist_func = DistFuncType.COSH_DIST_SQ
        else:
            print("Unrecognized distance function type", dist_func_str, "for mix-poincare")
            exit(-1)

    # Convert optimizer from string to one of the enum values.
    if opt_str == "adagrad":
        optimizer = OptType.ADAGRAD
    elif opt_str == "radagrad":
        optimizer = OptType.RADAGRAD
    elif opt_str == "mixradagrad":
        optimizer = OptType.MixRADAGRAD
    elif opt_str == "fullrsgd":
        optimizer = OptType.FullRSGD
    elif opt_str == "wfullrsgd":
        optimizer = OptType.WeightedFullRSGD
    elif opt_str == "ramsgrad":
        optimizer = OptType.RAMSGRAD

    if with_bias:
        b0 = <REAL_t *>(np.PyArray_DATA(model.trainables.b0))
        b1 = <REAL_t *>(np.PyArray_DATA(model.trainables.b1))
    else:
        # NOTE: if we don't have bias, b0[0] will hold the global mean bias and b1[0] will hold the global variance bias.
        b0 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        b1 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        b0[0] = model.trainables.mean_bias
        b1[0] = model.trainables.var_bias


    # Extract arrays for accumulated squared gradients.
    gradsq_syn0 = <REAL_t *>(np.PyArray_DATA(model.trainables.gradsq_syn0))
    gradsq_syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.gradsq_syn1))
    if with_bias:
        gradsq_b0 = <REAL_t *>(np.PyArray_DATA(model.trainables.gradsq_b0))
        gradsq_b1 = <REAL_t *>(np.PyArray_DATA(model.trainables.gradsq_b1))
    else:
        gradsq_b0 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        gradsq_b1 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        gradsq_b0[0] = model.trainables.gradsq_mean_bias
        gradsq_b1[0] = model.trainables.gradsq_var_bias

    # Extract arrays for accumulated momentum.
    mom_syn0 = <REAL_t *>(np.PyArray_DATA(model.trainables.mom_syn0))
    mom_syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.mom_syn1))
    if with_bias:
        mom_b0 = <REAL_t *>(np.PyArray_DATA(model.trainables.mom_b0))
        mom_b1 = <REAL_t *>(np.PyArray_DATA(model.trainables.mom_b1))
    else:
        mom_b0 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        mom_b1 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        mom_b0[0] = model.trainables.mom_mean_bias
        mom_b1[0] = model.trainables.mom_var_bias

    # Extract arrays for accumulated betas for AMSgrad.
    betas_syn0 = <REAL_t *>(np.PyArray_DATA(model.trainables.betas_syn0))
    betas_syn1 = <REAL_t *>(np.PyArray_DATA(model.trainables.betas_syn1))
    if with_bias:
        betas_b0 = <REAL_t *>(np.PyArray_DATA(model.trainables.betas_b0))
        betas_b1 = <REAL_t *>(np.PyArray_DATA(model.trainables.betas_b1))
    else:
        betas_b0 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        betas_b1 = <REAL_t *> malloc(cython.sizeof(REAL_t) * 1)
        betas_b0[0] = model.trainables.betas_mean_bias
        betas_b1[0] = model.trainables.betas_var_bias

    fast_glove(emb_type, use_glove_format, coocc_func, dist_func, nn, use_scaling, &scaling_factor, coocc_file_path,
               syn0, syn1, b0, b1,
               gradsq_syn0, gradsq_syn1, gradsq_b0, gradsq_b1,
               mom_syn0, mom_syn1, mom_b0, mom_b1,
               betas_syn0, betas_syn1, betas_b0, betas_b1,
               num_workers, chunksize, size, num_embs, num_pairs, max_word_index, cosh_dist_pow, optimizer, lr, index2freq,
               _compute_loss, with_bias, use_log_probs, &epoch_loss, &trained_pair_count, &num_projections)

    model.num_projections = num_projections
    if use_scaling == 1:
        model.scaling_factor = scaling_factor

    return epoch_loss, trained_pair_count


# Write the co-occurrence data in binary format.
def write_all(input_text_file, use_glove_format=True, restrict_vocab=999999):

    cdef FILE *fout
    cdef int p[3]
    cdef size_t size
    cdef bytes py_bytes
    cdef double count_dbl = 0.0
    cdef char *output_bin_file

    basename = input_text_file.rsplit(".", 1)[0]
    print(basename)
    output = basename+"_vocab"+str(restrict_vocab)+".bin"
    print(output)

    py_bytes = output.encode()
    output_bin_file = py_bytes
    fout = fopen(output_bin_file, "wb")

    for line in open(input_text_file, "r"):
        w1, w2, count = [x for x in line.strip().split("\t")]
        w1 = int(w1)
        w2 = int(w2)
        if use_glove_format is False:
            count = int(count)

            if w1 > restrict_vocab or w2 > restrict_vocab:
                continue

            p[0], p[1], p[2] = w1, w2, count
            fwrite(&p, cython.sizeof(int), 3, fout)
        else:
            count = float(count)

            if w1 > restrict_vocab or w2 > restrict_vocab:
                continue

            p[0], p[1] = w1, w2
            count_dbl = count
            fwrite(&p, cython.sizeof(int), 2, fout)
            fwrite(&count_dbl, cython.sizeof(double), 1, fout)

    fclose(fout)


# Read only one (word1, word2, coocc_count) triple.
cdef inline size_t read_triple(FILE *fp, const int use_glove_format, const int threadid, const int index, int *a, int *b, float *c) nogil:
    cdef size_t size
    cdef int p[3]
    cdef double count = 0.0
    cdef long int offset

    if use_glove_format == 1:
        offset = index * (2 * cython.sizeof(int) + cython.sizeof(double))
        fseek(fp, offset, SEEK_SET)
        size = fread(&p, cython.sizeof(int), 2, fp) + fread(&count, cython.sizeof(double), 1, fp)

        # Vocab is indexed from 0; for co-occ we use 1-based indexing, for the Glove format.
        a[0] = p[0] - 1
        b[0] = p[1] - 1
        c[0] = count
    else:
        offset = index * (3 * cython.sizeof(int))
        fseek(fp, offset, SEEK_SET)
        size = fread(&p, cython.sizeof(int), 3, fp)
        a[0] = p[0]
        b[0] = p[1]
        c[0] = p[2]

    return size


# Convert co-occ. data from adjacency matrix to adjacency list format. `limited_vocab` can be used to use a restricted
# vocabulary.
def read_as_neighbor_lists(use_glove_format, filename, num_pairs, vocab_size, limited_vocab={}):
    cdef bytes py_bytes = filename.encode()
    cdef char *bin_file = py_bytes
    cdef FILE *fp = fopen(bin_file, "rb")
    cdef int p[3]
    cdef double count = 0.0
    cdef size_t size
    cdef int i

    # Maps node -> adjacency list
    graph_map = {}
    for i in range(vocab_size):
        graph_map[i+1] = set()

    pair_count = 0.0
    for i in range(num_pairs):
        # printf("%d\t%d\t%lf\n", p[0], p[1], count)
        if use_glove_format:
            size = fread(&p, cython.sizeof(int), 2, fp) + fread(&count, cython.sizeof(double), 1, fp)
        else:
            size = fread(&p, cython.sizeof(int), 3, fp)

        word1, word2 = p[0], p[1]
        if word1 in graph_map and word2 in graph_map:
            if len(limited_vocab) == 0 or (word1 in limited_vocab and word2 in limited_vocab):
                pair_count = pair_count + 1
                graph_map[word1].add(word2)
                graph_map[word2].add(word1)

    fclose(fp)
    print("Found {} pairs between the words from the limited vocabulary".format(pair_count))

    return graph_map


# Takes as input a file that contains co-occurence pairs. Only select the pairs for which both words have a frequency
# rank smaller than `restrict_vocab`.
def extract_restrict_vocab_pairs(in_file, out_file, restrict_vocab=50000):
    cdef bytes in_py_bytes = in_file.encode(), out_py_bytes = out_file.encode()
    cdef char *in_bin_file = in_py_bytes, *out_bin_file = out_py_bytes
    cdef FILE *fin = fopen(in_bin_file, "rb"), *fout = fopen(out_bin_file, "wb")
    cdef int p[3]
    cdef double count = 0.0
    cdef size_t size
    cdef int num_pairs = 0

    size = fread(&p, cython.sizeof(int), 2, fin) + fread(&count, cython.sizeof(double), 1, fin)
    while size > 0:
        # printf("%d\t%d\t%lf\n", p[0], p[1], count)
        size = fread(&p, cython.sizeof(int), 2, fin) + fread(&count, cython.sizeof(double), 1, fin)
        if p[0] > restrict_vocab or p[1] > restrict_vocab:
            continue

        fwrite(&p, cython.sizeof(int), 2, fout)
        fwrite(&count, cython.sizeof(double), 1, fout)

        num_pairs = num_pairs + 1

    fclose(fin)
    fclose(fout)

    return num_pairs


# Read all the co-occurrence pairs from the file passed as parameter. If `return_pairs` is True, also return a map,
# where "w1 w2" is the key (where w1 and w2 are the words in the pair) and the values are the co-occ. counts.
def read_all(use_glove_format, filename, return_pairs=False):
    cdef bytes py_bytes = filename.encode()
    cdef char *bin_file = py_bytes
    cdef FILE *fp = fopen(bin_file, "rb")
    cdef int p[3]
    cdef double count = 0.0, max_count
    cdef size_t size
    cdef int num_pairs = 0

    pairs_map = {}

    if use_glove_format:
        size = fread(&p, cython.sizeof(int), 2, fp) + fread(&count, cython.sizeof(double), 1, fp)
        max_count = count
    else:
        size = fread(&p, cython.sizeof(int), 3, fp)
        max_count = p[3]
    while size > 0:
        # printf("%d\t%d\t%lf\n", p[0], p[1], count)
        if use_glove_format:
            size = fread(&p, cython.sizeof(int), 2, fp) + fread(&count, cython.sizeof(double), 1, fp)
            if max_count < count:
                max_count = count
        else:
            size = fread(&p, cython.sizeof(int), 3, fp)
            if max_count < p[3]:
                max_count = p[3]

        if return_pairs:
            key = str(p[0]) + " " + str(p[1]) if p[0] <= p[1] else str(p[1]) + " " + str(p[0])
            if key in pairs_map and pairs_map[key] != count:
                raise RuntimeError("Co-occ matrix is not symmetric! coocc({}, {}) = {} and coocc({}, {}) = {}".format(
                    p[1], p[0], pairs_map[key], p[0], p[1], count
                ))

            if key not in pairs_map:
                pairs_map[key] = count

        num_pairs = num_pairs + 1

    fclose(fp)

    if return_pairs:
        return num_pairs, pairs_map, max_count
    else:
        return num_pairs


def init():
    global our_dot
    global our_saxpy

    cdef int i
    cdef float *x = [<float>10.0]
    cdef float *y = [<float>0.01]
    cdef float expected = <float>0.1
    cdef int size = 1
    cdef double d_res
    cdef float *p_res

    # Precompute function `sigmoid(x) = 1 / (1 + exp(-x))`, for x values discretized
    # into table EXP_TABLE.
    for i in range(EXP_TABLE_SIZE):
        EXP_TABLE[i] = <REAL_t>exp((i / <REAL_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
        EXP_TABLE[i] = <REAL_t>(EXP_TABLE[i] / (EXP_TABLE[i] + 1))

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

init()  # initialize the module
