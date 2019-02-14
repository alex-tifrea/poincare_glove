import argparse
import gensim
from glove_code.src.glove import Glove
from gensim.models.callbacks import LossLogger, LossSetter
import json
from nltk.corpus import wordnet as wn
import numpy as np
import os
import random
from scipy import stats
from sklearn.linear_model import RidgeCV
import sys

# TODO: maybe tune alpha too (see "things to talk about in next meeting)
# alphas = np.array(range(21)) * 0.1
alphas = [1.0]
trunc_thresholds = np.array(range(21)) * 0.08

MODEL = None

# OUTDATED
def hyperlex_unsupervised(model, root, is_debug, debug_file=None):
    print("Unsupervised HyperLex scores (full dataset):")
    max_score, max_alpha, max_thresh = 0, 0, 0
    for alpha in alphas:
        if isinstance(model, Glove):
            alpha = -alpha
        curr_max_score, curr_max_thresh = 0, 0
        for thresh in trunc_thresholds:
            spearman, _, _, _, _ = model.wv.evaluate_lexical_entailment(
                os.path.join(root, 'data/hyperlex-data', 'hyperlex-all.txt'),
                dummy4unknown=False,
                alpha=alpha,
                trunc_threshold=thresh
            )
            if is_debug:
                print("alpha={};\tthresh={}\t=>\t{}".format(alpha, thresh, spearman[0]))
                sys.stdout.flush()
            if spearman[0] > curr_max_score:
                curr_max_score = spearman[0]
                curr_max_thresh = thresh

                if curr_max_score > max_score:
                    max_score = curr_max_score
                    max_alpha = alpha
                    max_thresh = curr_max_thresh
        if is_debug:
            print("For alpha={}, max score is {}, for thresh={}".format(alpha, curr_max_score, curr_max_thresh))
            print()
            sys.stdout.flush()

    # Run best model on nouns and verbs splits.
    spearman_nouns, _, _, _, _ = model.wv.evaluate_lexical_entailment(
        os.path.join(root, 'data/hyperlex-data/nouns-verbs', 'hyperlex-nouns.txt'),
        dummy4unknown=False,
        alpha=max_alpha,
        trunc_threshold=max_thresh,
        debug_file=debug_file
    )
    spearman_verbs, _, _, _, _ = model.wv.evaluate_lexical_entailment(
        os.path.join(root, 'data/hyperlex-data/nouns-verbs', 'hyperlex-verbs.txt'),
        dummy4unknown=False,
        alpha=max_alpha,
        trunc_threshold=max_thresh
    )

    print("\t- MAX SCORE (alpha/threshold/corr_nouns/corr_verbs/corr_all): {:.2f} / {:.2f} / {:.4f} / {:.4f} / {:.4f}".format(
        max_alpha, max_thresh, spearman_nouns[0], spearman_verbs[0], max_score))


def hyperlex_supervised(model, root, split_type):
    print("Supervised HyperLex scores ({} split):".format(split_type))
    hyperlex_training_file = os.path.join(root, "data/hyperlex-data/splits/"+split_type+"/hyperlex_training_all_"+split_type+".txt")
    hyperlex_test_file = os.path.join(root, "data/hyperlex-data/splits/"+split_type+"/hyperlex_test_all_"+split_type+".txt")

    train_set = read_hyperlex_format(model, hyperlex_training_file)
    test_set = read_hyperlex_format(model, hyperlex_test_file)
    train_features, train_labels = extract_vector_features(model, train_set)
    test_features, test_labels = extract_vector_features(model, test_set)

    # Train and evaluate Ridge regression model.
    regression_model = RidgeCV(cv=3,
                               alphas=[100.0, 75.0, 60.0, 50.0, 40.0, 35.0, 25.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25, 0.1],
                               fit_intercept=True)
    regression_model.fit(train_features, train_labels)

    # print("\t- Training set: {:.4f}".format(eval_model(regression_model, train_features, train_labels)))
    print("\t- Test set: {:.4f}".format(eval_model(regression_model, test_features, test_labels)))


# OUTDATED
def wbless_eval(model, root):
    print("Lexical entailment accuracy on WBLESS:")
    wbless_file = os.path.join(root, "data/BLESS_datasets/weeds_bless.json")
    with open(wbless_file, "r") as f:
        data = json.load(f)
    correct_count, total_count = 0.0, 0.0
    threshold = 123
    alpha = 0.02

    correct = []
    incorrect = []
    for w1, w2, label in data:
        if w1 not in model.wv.vocab or w2 not in model.wv.vocab:
            continue
        total_count += 1
        model_score = 0
        distance = model.wv.distance(w1, w2)
        if distance < threshold:
            model_score = 1 - (model.wv.embedding_norm(w2) + alpha) / model.wv.embedding_norm(w1)
        if (model_score > 0 and label == 1) or (model_score <= 0 and label == 0):
            correct_count += 1
            correct.append((w1, w2, label, model_score))
        else:
            incorrect.append((w1, w2, label, model_score))

    print("\t- Accuracy: {:.4f}, from a total of {}/{} pairs".format(
        float(correct_count) / total_count, int(total_count), len(data)))


def wordnet_rank(model, root, restrict_vocab=10000):
    print("Average rank in WordNet noun closure:")
    wordnet_noun_filename = os.path.join(root, "data/wordnet_noun_closure.tsv")
    wordnet_pairs = read_wordnet_format(model, wordnet_noun_filename, restrict_vocab=restrict_vocab)

    hypo_vectors = model.wv.vectors[wordnet_pairs[:, 2].astype(int)]

    target_ranks = []
    for hypo_vector, hyper_idx in zip(hypo_vectors, wordnet_pairs[:, 3].astype(int)):
        dists = model.wv.distances(hypo_vector, model.wv.vectors[:restrict_vocab])
        ranks = stats.rankdata(dists)
        target_ranks.append(ranks[hyper_idx])
    print("\t- avg rank is {:.4f}".format(np.average(np.array(target_ranks))))


def wordnet_level_rank_vector_norm_correlation(model, root):
    print("Correlation between WordNet level rank and vector norms (for noun transitive closure):")
    wordnet_noun_filename = os.path.join(root, "data/wordnet_noun_closure.tsv")
    wordnet_pairs = read_wordnet_format(model, wordnet_noun_filename)

    all_words = np.concatenate((wordnet_pairs[:, 0], wordnet_pairs[:, 1]))
    wordnet_levels = np.array([wn.synset(word).max_depth() for word in all_words])
    # wordnet_children_depth = []
    # for word in all_words:
    #     children_depth = [w.max_depth() for w in wn.synset(word).hyponyms()]
    #     if children_depth == []:
    #         max_children_depth = 0
    #     else:
    #         max_children_depth = max(children_depth)
    #     wordnet_children_depth.append(max(wn.synset(word).max_depth(), max_children_depth))
    # wordnet_level_ranks = wordnet_levels / np.array(wordnet_children_depth)
    wordnet_level_ranks = wordnet_levels
    word_indexes = np.concatenate((wordnet_pairs[:, 2], wordnet_pairs[:, 3])).astype(int)
    target_vector_norms = np.linalg.norm(model.wv.vectors[word_indexes], axis=1)
    # context_vector_norms = np.linalg.norm(model.trainables.syn1neg[word_indexes], axis=1)

    print("\t- Spearman correlation {:.4f}".format(stats.spearmanr(wordnet_level_ranks, target_vector_norms)[0]))


def norms_distribution(model):
    print("Vector norm statistics:")
    norms = np.linalg.norm(model.wv.vectors, axis=1)
    print("\t- Target vectors: avg norm / stddev / min norm / max norm: {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
          np.average(norms), np.std(norms), np.min(norms), np.max(norms)))

    norms = np.linalg.norm(model.trainables.syn1neg, axis=1)
    print("\t- Context vectors: avg norm / stddev / min norm / max norm: {:.4f} / {:.4f} / {:.4f} / {:.4f}".format(
        np.average(norms), np.std(norms), np.min(norms), np.max(norms)))


def norm_freq_correlation(model):
    print("Correlation between vector norms and 1.0 / freq:")
    word_freq = np.array([model.wv.vocab[word].count for word in model.wv.index2word])
    vector_norms = np.linalg.norm(model.wv.vectors, axis=1)

    print("\t- Spearman correlation (word rank<10000 / word rank>10000 / all): {:.4f} / {:.4f} / {:.4f}".format(
        stats.spearmanr(1.0 / word_freq[:10000], vector_norms[:10000])[0],
        stats.spearmanr(1.0 / word_freq[10000:], vector_norms[10000:])[0],
        stats.spearmanr(1.0 / word_freq, vector_norms)[0]))


def avg_relative_contrast(model, restrict_vocab=50000, num_samples=100, rank_thresh=100):
    print("Relative contrast:")
    indexes = random.sample(range(rank_thresh), num_samples) + \
              random.sample(range(rank_thresh, min(restrict_vocab, len(MODEL.wv.vocab))), num_samples)
    rcs = []
    for idx in indexes:
        rcs.append(compute_relative_contrast(model, idx, restrict_vocab))

    print("\t- Average Relative Contrast (word rank<{}/ word rank>{} / all): {:.4f} / {:.4f} / {:.4f}".format(
        rank_thresh, rank_thresh,
        np.average(np.array(rcs[:num_samples])),
        np.average(np.array(rcs[num_samples:])),
        np.average(np.array(rcs))))


# =================== HELPERS ===================
def compute_relative_contrast(model, index, restrict_vocab):
    limited = np.delete(model.wv.vectors[:restrict_vocab], index, 0)
    dists = model.wv.distances(model.wv.vectors[index], limited)
    min_dist = np.min(dists)
    mean_dist = np.average(dists)
    return mean_dist / (min_dist + 1e-15)

def extract_vector_features(model, dataset, agg_function=(lambda x, y: x-y)):
    index0 = dataset[:, 2].astype(int)
    index1 = dataset[:, 3].astype(int)
    labels = dataset[:, 4].astype(float)

    # Use w (target vectors).
    features = np.array([agg_function(v1, v2) for v1, v2 in zip(model.wv.vectors[index0], model.wv.vectors[index1])])

    return features, labels

def eval_model(regression_model, features, labels):
    pred = regression_model.predict(features[:, :])
    return stats.spearmanr(pred, labels)[0]

def read_hyperlex_format(model, filename):
    with open(filename, "r") as f:
        lines = [line.strip().split() for line in f.readlines()[1:]]
        result = []
        discarded_count = 0
        for line in lines:
            if line[0] not in model.wv.vocab or line[1] not in model.wv.vocab:
                discarded_count += 1
                continue
            result.append([line[0], line[1], model.wv.vocab[line[0]].index, model.wv.vocab[line[1]].index, float(line[5])])
        return np.array(result)

def read_wordnet_format(model, filename, restrict_vocab=500000):
    with open(filename, "r") as f:
        lines = [line.strip().split() for line in f.readlines()[1:]]
        result = []
        discarded_count = 0
        for line in lines:
            hypo, hyper = line[0], line[1]
            hypo_word, hyper_word = hypo.split(".")[0], hyper.split(".")[0]
            if (hypo_word not in model.wv.vocab or model.wv.vocab[hypo_word].index > restrict_vocab) or \
                    (hyper_word not in model.wv.vocab or model.wv.vocab[hyper_word].index > restrict_vocab):
                discarded_count += 1
                continue
            result.append([hypo, hyper, model.wv.vocab[hypo_word].index, model.wv.vocab[hyper_word].index])
        print("Instances used: ", len(result))
        return np.array(result)


# model_fn = os.path.join(ROOT, "models/geometric_emb/w2v_levy_nll_5_100_A01_a0001_n5_w5_c100_poincare_OPTwfullrsgd_SIMcosh-dist-sq_burnin1")
# model_fn = os.path.join(args.root, "models/word2vec_baseline/w2v_levy_sg_5_100_A025_a0001_n5_w5_c100_cosine_OPTsgd")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--restrict_vocab', type=int, default=100000,
                        help='Size of vocab. Only used for evaluating analogy.')
    parser.add_argument('--model_filename', type=str, default='', help='Path to saved model.')
    parser.add_argument('--agg_eval', dest='agg_eval', action='store_true',
                        help='Use w+c during evaluation, instead of just w. Only works for Poincare embeddings.')
    parser.add_argument('--cosine_eval', dest='cosine_eval', action='store_true',
                        help='Use cosine distance during evaluation, instead of the Poincare distance.')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                        help='Run model in debug mode')
    parser.add_argument('--glove', dest='glove', action='store_true',
                        help='Use a Glove model.')
    parser.set_defaults(train=False, agg_eval=False, cosine_dist=False, is_debug=False, glove=False)
    args = parser.parse_args()

    if args.glove:
        MODEL = Glove.load(args.model_filename)
    else:
        MODEL = gensim.models.Word2Vec.load(args.model_filename)

    # hyperlex_unsupervised(MODEL, args.root, args.is_debug)
    # hyperlex_supervised(MODEL, args.root, "random")
    # hyperlex_supervised(MODEL, args.root, "lexical")

    # norm_freq_correlation(MODEL)
    # avg_relative_contrast(MODEL)
    # norms_distribution(MODEL)

    # wbless_eval(MODEL, args.root)

    # WordNet metrics.
    # wordnet_rank(MODEL, args.root)
    # wordnet_level_rank_vector_norm_correlation(MODEL, args.root)
