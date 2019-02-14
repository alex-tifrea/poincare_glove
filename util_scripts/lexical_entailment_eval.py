import argparse
import gensim
from glove_code.src.glove import Glove
from gensim.models.keyedvectors import PoincareWordEmbeddingsKeyedVectors as pkv
from gensim.models.callbacks import LossLogger, LossSetter
import json
from nltk.corpus import wordnet as wn
import numpy as np
import os
import random
from scipy import stats
from scipy.linalg import block_diag
import sys


MODEL = None
ROOT = ".."

hyperlex_file = os.path.join(ROOT, "data/hyperlex-data/hyperlex-all.txt")
wbless_file = os.path.join(ROOT, "data/BLESS_datasets/weeds_bless.json")


def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)


def read_hyperlex_format(filename, model):
    wv = model.wv

    with open(filename, "r") as f:
        lines = [line.strip().split() for line in f.readlines()[1:]]
        result = []
        hyperlex_vocab = {}
        discarded_count = 0
        for line in lines:
            if line[0] not in wv.vocab or line[1] not in wv.vocab:
                discarded_count += 1
                continue
            result.append([line[0], line[1], wv.vocab[line[0]].index, wv.vocab[line[1]].index, float(line[5])])
            hyperlex_vocab[line[0]] = [line[0]]
            hyperlex_vocab[line[1]] = [line[1]]
        # print("Discarded {} pairs out of {}".format(discarded_count, len(lines)))
        return np.array(result), hyperlex_vocab


def read_wbless(filename):
    with open(filename, "r") as f:
        wbless_data = json.load(f)

    wbless_vocab = {}
    for w1, w2, _ in wbless_data:
        wbless_vocab[w1] = w1
        wbless_vocab[w2] = w2

    return wbless_data, wbless_vocab


def mix_poincare_moebius_add_mat(A, B, num_embs):
    small_emb_size = int(A.shape[1] / num_embs)
    result = np.empty_like(A)
    for i in range(num_embs):
        start = i * small_emb_size
        end = (i + 1) * small_emb_size
        result[:, start:end] = pkv.moebius_add_mat(A[:, start:end], B[:, start:end])
    return result


def mix_poincare_moebius_mul_mat(A, r, num_embs):
    small_emb_size = int(A.shape[1] / num_embs)
    result = np.empty_like(A)
    for i in range(num_embs):
        start = i * small_emb_size
        end = (i + 1) * small_emb_size
        result[:, start:end] = pkv.moebius_mul_mat(A[:, start:end], r)
    return result


def fisher_info_distance(v, w, num_embs):
    v = v.reshape(-1)
    w = w.reshape(-1)

    half_plane_dists = []
    small_emb_size = int(v.shape[0] / num_embs)

    if small_emb_size != 2:
        raise RuntimeError("Only implemented for Cartesian product of 2D spaces; Current small_emb_size is {}".format(
            small_emb_size))

    diff = v - w
    for i in range(num_embs):
        start = i * small_emb_size
        end = (i + 1) * small_emb_size
        half_plane_dists.append(
            np.arccosh(1 + np.dot(diff[start:end], diff[start:end]) / (2 * v[start + 1] * w[start + 1]))
        )
    return np.linalg.norm(half_plane_dists) * np.sqrt(2)


# Rotate the rows of X. Each row is split into smaller 2D subspaces. sin_cos_vector contains (sin a, cos a) pairs,
# where a is the angle with which we rotate that 2D portion of a row of X.
def rotate_mat(sin_cos_vector, X):
    # Create rotation matrix.
    cos_sin_blocks = [[[c, -s], [s, c]] for s, c in sin_cos_vector.reshape(-1, 2)]
    rotation_matrix = cos_sin_blocks[0]
    for i in range(1, len(cos_sin_blocks)):
        rotation_matrix = block_diag(rotation_matrix, cos_sin_blocks[i])

    return (np.matmul(rotation_matrix, X.T)).T


def poincare_ball2half_plane(A, num_embs):
    small_emb_size = int(A.shape[1] / num_embs)
    result = np.empty_like(A)
    for i in range(num_embs):
        start = i * small_emb_size
        x = A[:, start]
        y = A[:, start + 1]
        denominator = x * x + (1 - y) * (1 - y)
        result[:, start] = 2 * x / denominator
        result[:, start + 1] = (1 - x * x - y * y) / denominator
    return result


def get_gaussians(model, hyperlex_vocab, wbless_vocab, unsupervised=False, aggregate="w",
                  scaling_factor=1.0, words_to_use=400):

    wv = model.wv
    wordnet_selected_words_file = os.path.join(ROOT,
                                               "msc_tifreaa/glove_code/data/wordnet_topmost_and_bottommost_words.txt")
    with open(wordnet_selected_words_file, "r") as f:
        # Select words that are on the top-most/bottom-most levels in WordNet, that are included in the
        # vocabulary of the model AND that do not appear in any of the HyperLex pairs.
        top_level_word_idxs = np.array([wv.vocab[word].index for word in filter(
            lambda w: w in wv.vocab and w not in hyperlex_vocab and w not in wbless_vocab,
            f.readline().strip().split(" "))])
        bottom_level_word_idxs = np.array([wv.vocab[word].index for word in filter(
            lambda w: w in wv.vocab and w not in hyperlex_vocab and w not in wbless_vocab,
            f.readline().strip().split(" "))])

    if unsupervised:
        top_level_word_idxs = range(0, words_to_use)
        bottom_level_word_idxs = range(50000 - words_to_use, 50000)
    else:
        top_level_word_idxs = top_level_word_idxs[:words_to_use]
        bottom_level_word_idxs = bottom_level_word_idxs[-words_to_use:]

    print(len(top_level_word_idxs), "words from the TOP-most levels were selected")
    print(len(bottom_level_word_idxs), "words from the BOTTOM-most levels were selected")

    if aggregate == "w":
        vectors = wv.vectors
    elif aggregate == "c":
        vectors = model.trainables.syn1neg
    else:
        return None

    # Rescale ALL embeddings.
    rescaled_vectors = mix_poincare_moebius_mul_mat(vectors, scaling_factor, model.num_embs)

    # Compute EUCLIDEAN average of top/bottom-most levels.
    top_and_bottom_levels_avg = np.mean(
        rescaled_vectors[np.concatenate((top_level_word_idxs, bottom_level_word_idxs)), :],
        axis=0)

    # Recenter ALL embeddings.
    mean_mat = np.repeat(top_and_bottom_levels_avg.reshape(1, -1), rescaled_vectors.shape[0], axis=0)
    recentered_vectors = mix_poincare_moebius_add_mat(-mean_mat, rescaled_vectors, model.num_embs)

    # Compute EUCLIDEAN average of the recentered top-most levels.
    top_levels_avg = np.mean(recentered_vectors[top_level_word_idxs, :], axis=0).reshape(-1, 2)
    top_levels_avg_norm = (top_levels_avg / np.linalg.norm(top_levels_avg, axis=1)[:, None]).reshape(-1)

    # print(np.linalg.norm(top_levels_avg_norm.reshape(-1, 2), axis=1))
    # Rotate ALL embeddings.
    rotated_vectors = rotate_mat(top_levels_avg_norm, recentered_vectors)

    # Isometry to convert from Poincare ball model to half-plane model.
    half_plane_vectors = poincare_ball2half_plane(rotated_vectors, model.num_embs)
    # print("HP shape", half_plane_vectors.shape)

    # Convert half-plane points to gaussian parameters.
    gaussians = half_plane_vectors.reshape(-1, model.num_embs, 2)
    gaussians[:, :, 0] /= np.sqrt(2)
    print("Gaussians shape", gaussians.shape)

    return gaussians


def get_KL_score(w1_idx, w2_idx, gaussians, **kwargs):
    v1, v2 = gaussians[w1_idx].reshape(-1, 2), gaussians[w2_idx].reshape(-1, 2)
    agg_neg_kl = 0.0
    for i in range(v1.shape[0]):
        m1, s1 = v1[i]
        m2, s2 = v2[i]
        curr_kl = 1.0 / 2 * (2 * np.log(s2 / s1) + (s1 / s2) ** 2 + (m1 - m2) ** 2 / s2 ** 2 - 1)
        agg_neg_kl -= curr_kl
    return agg_neg_kl


def get_cos_sim_score(w1_idx, w2_idx, gaussians, **kwargs):
    v1, v2 = gaussians[w1_idx].reshape(-1, 2), gaussians[w2_idx].reshape(-1, 2)
    cos_sims = []
    for i in range(v1.shape[0]):
        m1, _ = v1[i]
        m2, _ = v2[i]
        cos_sims.append(np.dot(m1, m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-10))
    return np.sum(cos_sims)


def get_nickel_score(w1_idx, w2_idx, score_type, gaussians, model, alpha_L=1000, alpha_N=1000, nickel_threshold=2.0,
                     debug_data=None, sigma_factor=1.0, **kwargs):

    v1, v2 = gaussians[w1_idx].reshape(-1, 2), gaussians[w2_idx].reshape(-1, 2)
    log_cardinal_v1 = v1.shape[0] * np.log(2 * sigma_factor) + np.log(v1[:, 1]).sum()
    log_cardinal_v2 = v2.shape[0] * np.log(2 * sigma_factor) + np.log(v2[:, 1]).sum()
    fisher_dist = fisher_info_distance(gaussians[w1_idx], gaussians[w2_idx], model.wv.num_embs)

    if debug_data != None:
        debug_data.append((round(log_cardinal_v1, 2), round(log_cardinal_v2, 2), round(fisher_dist, 2)))

    if score_type == 'L':
        return -(1 + alpha_L * (log_cardinal_v1 - log_cardinal_v2)) * fisher_dist
    elif score_type == 'M':
        return -(1 + alpha_L * (log_cardinal_v1 - log_cardinal_v2)) * max(fisher_dist, nickel_threshold)
    elif score_type == 'N':
        return -(1 + alpha_N * (1.0 / log_cardinal_v2 - 1.0 / log_cardinal_v1)) * fisher_dist
    elif score_type == 'O':
        return -(1 + alpha_N * (1.0 / log_cardinal_v2 - 1.0 / log_cardinal_v1)) * max(fisher_dist, nickel_threshold)
    else:
        raise RuntimeError("Unknown score_type")


def get_is_a_score(w1_idx, w2_idx, score_type, gaussians, model, kiela_threshold=2.0, sigma_factor=1, **kwargs):

    v1, v2 = gaussians[w1_idx].reshape(-1, 2), gaussians[w2_idx].reshape(-1, 2)
    log_cardinal_v1 = v1.shape[0] * np.log(2 * sigma_factor) + np.log(v1[:, 1]).sum()
    log_cardinal_v2 = v2.shape[0] * np.log(2 * sigma_factor) + np.log(v2[:, 1]).sum()
    # cos_sim = 0.0
    cardinal_intersection = 1.0
    for i in range(v1.shape[0]):
        m1, s1 = v1[i]
        m2, s2 = v2[i]
        s1 *= sigma_factor
        s2 *= sigma_factor
        a, b = m1 - s1, m1 + s1
        c, d = m2 - s2, m2 + s2
        if d > b:
            if c > a:
                cardinal_intersection = cardinal_intersection * max(0, b - c)
            else:
                cardinal_intersection = cardinal_intersection * (b - a)
        else:
            if c > a:
                cardinal_intersection = cardinal_intersection * (d - c)
            else:
                cardinal_intersection = cardinal_intersection * max(0, d - a)

        # cos_sim += np.dot(m1, m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-10)

    fisher_dist = fisher_info_distance(gaussians[w1_idx], gaussians[w2_idx], model.wv.num_embs)

    cardinal_intersection = np.log(1e-10 + cardinal_intersection)

    if score_type == 'C':
        return -log_cardinal_v1
    elif score_type == 'D':
        return log_cardinal_v2
    elif score_type == 'E':
        return cardinal_intersection
    elif score_type == 'F':
        return cardinal_intersection - log_cardinal_v1
    elif score_type == 'G':
        return cardinal_intersection + log_cardinal_v2
    elif score_type == 'H':
        return cardinal_intersection + log_cardinal_v2 - log_cardinal_v1
    elif score_type == 'I':
        return log_cardinal_v2 - log_cardinal_v1
    elif score_type == 'J':
        return 1 - log_cardinal_v1 / log_cardinal_v2
    elif score_type == 'K':
        return 1 - log_cardinal_v1 / log_cardinal_v2 if fisher_dist < kiela_threshold else 0.0
    else:
        raise RuntimeError("Unknown score_type")


def get_hyperlex_score(model, gaussians, score_type, sigma_factor=1.0,
                       kiela_threshold=None, nickel_threshold=None, alpha_L=None, alpha_N=None):
    gold_scores = []
    model_scores = []

    if score_type == 'A':
        score_function = get_cos_sim_score
    elif score_type == 'B':
        score_function = get_KL_score
    elif score_type in ['L', 'M', 'N', 'O']:
        score_function = get_nickel_score
    else:
        score_function = get_is_a_score

    for _, _, w1_idx, w2_idx, gold_score in hyperlex_data:
        gold_scores.append(float(gold_score))
        model_scores.append(score_function(int(w1_idx), int(w2_idx), score_type=score_type, gaussians=gaussians,
                                           model=model, sigma_factor=sigma_factor,
                                           kiela_threshold=kiela_threshold, nickel_threshold=nickel_threshold,
                                           alpha_L=alpha_L, alpha_N=alpha_N))
    return stats.spearmanr(gold_scores, model_scores)[0]


def wbless_eval(model, gaussians, score_type, sigma_factor=1.0,
                kiela_threshold=None, nickel_threshold=None, alpha_L=None, alpha_N=None):
    correct_count, total_count = 0.0, 0.0

    if score_type == 'A':
        score_function = get_cos_sim_score
    elif score_type == 'B':
        score_function = get_KL_score
    elif score_type in ['L', 'M', 'N', 'O']:
        score_function = get_nickel_score
    else:
        score_function = get_is_a_score

    instances = []
    for w1, w2, label in wbless_data:
        if w1 not in model.wv.vocab or w2 not in model.wv.vocab:
            continue
        total_count += 1
        model_score = score_function(model.wv.vocab[w1].index, model.wv.vocab[w2].index, score_type=score_type,
                                     gaussians=gaussians, model=model, sigma_factor=sigma_factor,
                                     kiela_threshold=kiela_threshold, nickel_threshold=nickel_threshold,
                                     alpha_L=alpha_L, alpha_N=alpha_N)

        instances.append((w1, w2, label, model_score))

    threshold = np.mean(np.array([model_score for _, _, _, model_score in instances]))
    correct = list(filter(lambda instance: (instance[3] > threshold and instance[2] == 1) or (
            instance[3] <= threshold and instance[2] == 0), instances))

    return float(len(correct)) / len(instances), len(instances), len(wbless_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--restrict_vocab', type=int, default=200000,
                        help='Size of vocab. Only used for evaluating analogy.')
    parser.add_argument('--model_filename', type=str, default='', help='Path to saved model.')
    parser.add_argument('--unsupervised', dest='unsupervised', action='store_true',
                        help='Evaluate unsupervised; get words by frequency, instead of from WordNet, in order to'
                             'determine the centering/rotation.')
    parser.add_argument('--words_to_use', type=int, default=400,
                        help='Number of generic words and specific words to use in order to determine the '
                             'translation/rotation.')
    parser.add_argument('--agg_eval', dest='agg_eval', action='store_true',
                        help='Use w+c during evaluation, instead of just w. Only works for Poincare embeddings.')
    parser.set_defaults(unsupervised=False, agg_eval=False)
    args = parser.parse_args()

    MODEL = Glove.load(args.model_filename)
    ROOT = args.root

    hyperlex_data, hyperlex_vocab = read_hyperlex_format(hyperlex_file, MODEL)
    wbless_data, wbless_vocab = read_wbless(wbless_file)

    gaussians = get_gaussians(MODEL, hyperlex_vocab, wbless_vocab, unsupervised=args.unsupervised,
                              scaling_factor=1.0, words_to_use=args.words_to_use)

    if "DISTFUNCcosh-dist-sq" in args.model_filename:
        ALPHA_L = 1000
        ALPHA_N = 1000
        KIELA_THRESHOLD = 2.0
        NICKEL_THRESHOLD = 2.0
    elif "DISTFUNCdist-sq" in args.model_filename:
        ALPHA_L = 1000
        ALPHA_N = 1000
        KIELA_THRESHOLD = 4.0
        NICKEL_THRESHOLD = 4.0
    else:
        raise RuntimeError("Unsupported model type")

    hyperlex_scores = []
    wbless_scores = []
    for score_type in char_range('A', 'O'):
        hyperlex_scores.append(get_hyperlex_score(MODEL, gaussians, score_type=score_type, sigma_factor=1.0,
                               kiela_threshold=KIELA_THRESHOLD, nickel_threshold=NICKEL_THRESHOLD,
                               alpha_L=ALPHA_L, alpha_N=ALPHA_N))
        wbless_scores.append(wbless_eval(MODEL, gaussians, score_type=score_type, sigma_factor=1.0,
                             kiela_threshold=KIELA_THRESHOLD, nickel_threshold=NICKEL_THRESHOLD,
                             alpha_L=ALPHA_L, alpha_N=ALPHA_N)[0])

    print("HyperLex:")
    print("\t".join([str(round(x, 4)) for x in hyperlex_scores]))
    print("WBLESS:")
    print("\t".join([str(round(x, 4)) for x in wbless_scores]))
