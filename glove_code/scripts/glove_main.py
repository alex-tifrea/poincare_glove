#!/usr/local/bin/python3

import argparse
import gensim
from gensim.models.callbacks import WordEmbCheckpointSaver
from glove_code.src.glove import Glove, NNConfig, InitializationConfig
from util_scripts.get_model_eval_and_stats import *
import logging
from nltk.corpus import brown
import numpy as np
from numpy import float32 as REAL
from numpy.linalg import norm
import os
import socket
import time

MODEL_FILENAME_PATTERN = "models/glove/{}/glove_ep{}_size{}_lr{}_vocab{}_{}"
INITIALIZATION_MODEL_FILENAME = {
    "100D": "data/pretrained_models/glove_pretrained_100d_levy_vocab50k_cosh-dist-sq_bias",
    "vanilla_100D": "data/pretrained_models/glove_vanilla_pretrained_100d_levy_vocab50k_bias",
}
if socket.gethostname() in ["armin", "grinder", "mark", "youagain", "dalabgpu"]:  # DALAB machines
    INITIALIZATION_MODEL_FILENAME = {
        "100D": "/media/hofmann-scratch/Octavian/alext/data/pretrained_models/glove_pretrained_100d_levy_vocab50k_cosh-dist-sq_bias",
        "vanilla_100D": "/media/hofmann-scratch/Octavian/alext/data/pretrained_models/glove_vanilla_pretrained_100d_levy_vocab50k_bias",
    }
elif "lo-" in socket.gethostname():  # Leonhard nodes
    INITIALIZATION_MODEL_FILENAME = {
        "100D": "/cluster/scratch/tifreaa/data/pretrained_models/glove_pretrained_100d_levy_vocab50k_cosh-dist-sq_bias",
        "50x2D": "/cluster/scratch/tifreaa/data/pretrained_models/glove_pretrained_50x2D_ep50_levy_vocab50k_cosh-dist-sq_bias",
        "vanilla_100D": "/cluster/scratch/tifreaa/data/pretrained_models/glove_vanilla_pretrained_100d_levy_vocab50k_bias",
    }


SEM_GOOGLE_SIZE = 8869
SYN_GOOGLE_SIZE = 10675
GOOGLE_SIZE = 19544
MSR_SIZE = 8000

# IMPORTANT!!!!!!!!!!! First one for each embedding type should be the default.
SUPPORTED_OPTIMIZERS = {
    "vanilla": ["adagrad"],
    "euclid": ["adagrad"],
    "poincare": ["radagrad", "fullrsgd", "wfullrsgd", "ramsgrad"],
    "mix-poincare": ["mixradagrad"],
}

# IMPORTANT!! First one is the default.
# This refers to the distance function used in during training.
SUPPORTED_DIST_FUNCTIONS = {
    "vanilla": ["dist", "nn"],
    "euclid": ["dist-sq", "dist"],
    "poincare": ["dist-sq", "dist", "cosh-dist", "cosh-dist-sq", "cosh-dist-pow-*", "log-dist-sq"],
    "mix-poincare": ["dist", "dist-sq", "cosh-dist-sq"],
}

SUPPORTED_COOCC_FUNCTIONS = ["log"]

logging.basicConfig(level=logging.INFO)


def precision(eval_result):
    return len(eval_result['correct']) / (len(eval_result['correct']) + len(eval_result['incorrect']))


def compute_poincare_aggregate(model, config):
    """
    Precompute the average between the target and the context vector, for Poincare embeddings.
    We take as average the mid point between w and c on the geodesic that connects the 2 points
    (see page 89 in Ungar book).
    """
    if config["similarity"] == "poincare":
        print("precomputing aggregated vectors w+c for Poincare embeddings")
        gamma_w_sq = 1 / (1 - np.sum(model.wv.vectors * model.wv.vectors, axis=1))
        gamma_c_sq = 1 / (1 - np.sum(model.trainables.syn1neg * model.trainables.syn1neg, axis=1))
        denominator = gamma_w_sq + gamma_c_sq - 1
        agg = (model.wv.vectors * (gamma_w_sq / denominator)[:, None] +
               model.trainables.syn1neg * (gamma_c_sq / denominator)[:, None])

        model.wv.vectors = model.wv.moebius_mul_mat(agg, 0.5)
    elif config["similarity"] == "mix-poincare":
        print("precomputing aggregated vectors w+c for MIX-Poincare embeddings")
        small_emb_size = int(model.vector_size / model.num_embs)
        for i in range(model.num_embs):
            start = i * small_emb_size
            end = (i + 1) * small_emb_size
            indexes = range(start, end)
            gamma_w_sq = 1 / (1 - np.sum(model.wv.vectors[:, indexes] * model.wv.vectors[:, indexes], axis=1))
            gamma_c_sq = 1 / (1 - np.sum(model.trainables.syn1neg[:, indexes] * model.trainables.syn1neg[:, indexes], axis=1))
            denominator = gamma_w_sq + gamma_c_sq - 1
            agg = (model.wv.vectors[:, indexes] * (gamma_w_sq / denominator)[:, None] +
                   model.trainables.syn1neg[:, indexes] * (gamma_c_sq / denominator)[:, None])

            model.wv.vectors[:, indexes] = model.wv.moebius_mul_mat(agg, 0.5)
    else:
        print("precomputing aggregated vectors w+c for Euclidean embeddings")
        model.wv.vectors = model.wv.vectors + model.trainables.syn1neg


def split_filename(basename):
    info = basename.split('_')
    return info


# Extract information about a model from the filename.
def parse_model_filename(model_filename):
    info_dict = {}
    basename = os.path.basename(model_filename)
    info = split_filename(basename)

    if "pairs" in basename:
        return None, basename

    info_dict["epochs"] = int(info[1][2:])
    info_dict["emb_size"] = int(info[2][4:])
    info_dict["lr"] = float(info[3][2:])
    info_dict["restrict_vocab"] = int(info[4][5:])
    info_dict["similarity"] = info[5]
    info_dict["with_bias"] = True if "_bias" in basename else False
    info_dict["init_near_border"] = True if "_border-init" in basename else False

    for s in info:
        if "OPT" in s:
            info_dict["optimizer"] = s[3:]
        elif "COOCCFUNC" in s:
            info_dict["coocc_func"] = s[9:]
        elif "DISTFUNC" in s:
            info_dict["dist_func"] = s[8:]
        elif "scale" in s:
            info_dict["use_scaling"] = True
        elif "NUMEMBS" in s:
            info_dict["num_embs"] = int(s[7:])
        elif "logprobs" in s:
            info_dict["use_log_probs"] = True

    if "optimizer" not in info_dict:
        info_dict["optimizer"] = SUPPORTED_OPTIMIZERS[info_dict["similarity"]][0]
    if "coocc_func" not in info_dict:
        info_dict["coocc_func"] = SUPPORTED_COOCC_FUNCTIONS[0]
    if "dist_func" not in info_dict:
        info_dict["dist_func"] = SUPPORTED_DIST_FUNCTIONS[info_dict["similarity"]][0]
    if "use_scaling" not in info_dict:
        info_dict["use_scaling"] = False
    if "use_log_probs" not in info_dict:
        info_dict["use_log_probs"] = False

    return info_dict, basename


# Class that produces output both to stdout and to an output file. Used during evaluation.
class Logger:
    def __init__(self, fout=None):
        self.fout = fout

    def log(self, log_str='', end='\n'):
        logging.info(log_str)
        if self.fout:
            if end == '':
                self.fout.write(log_str)
            else:
                self.fout.write(log_str + end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train a new model.')
    parser.add_argument('--eval', dest='train', action='store_false',
                        help='Eval an existing model.')
    parser.add_argument('--use_our_format', dest='use_glove_format', action='store_false',
                        help='Use our format for reading the vocabulary and the co-occ matrix, instead of the format '
                             'from the original GloVe code.')
    parser.add_argument('--coocc_file', type=str,
                        help='Filename which contains the coocc matrix in text format.')
    parser.add_argument('--vocab_file', type=str,
                        help='Filename which contains the vocabulary.')
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--euclid', type=int, default=0,
                        help='Whether it uses Euclidean distance to train the embeddings instead of dot product.')
    parser.add_argument('--poincare', type=int, default=0, help='Whether it uses Poincare embeddings or not.')
    parser.add_argument('--dist_func', type=str, default="",
                        help='Distance function used by Poincare model during training.')
    parser.add_argument('--num_embs', type=int, default=0,
                        help='The number of small-dimensional planes that will come into the carthesian product of'
                             'manifolds')
    parser.add_argument('--mix', dest='mix', action='store_true',
                        help='If true, use a carthesian product of small-dimensional embeddings.')
    parser.add_argument('--nn_config', type=str, default="",
                        help='Configuration of the NN used during training.')
    parser.add_argument('--coocc_func', type=str, default="",
                        help='Co-occurence function used during training.')
    parser.add_argument('--use_scaling', dest='use_scaling', action='store_true',
                        help='Use trainable scaling factor for Poincare GloVe')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--restrict_vocab', type=int, default=400000,
                        help='Only use the `restrict_vocab` most frequent words')
    parser.add_argument('--size', type=int, default=100, help='Embedding size')
    parser.add_argument('--optimizer', type=str, default='', help='What optimizer to use.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--bias', dest='with_bias', action='store_true', help='Use a model with biases.')
    parser.add_argument('--workers', type=int, default=3, help='Number of concurrent workers.')
    parser.add_argument('--chunksize', type=int, default=1000,
                        help='Number of `prange` iterations that each thread processes at a time')
    parser.add_argument('--model_filename', type=str, default='', help='Path to saved model.')
    parser.add_argument('--train_log_filename', type=str, default='', help='Path to the training log.')
    parser.add_argument('--cosadd', dest='cosadd', action='store_true',
                        help='Use 3COSADD when evaluating word analogy.')
    parser.add_argument('--cosmul', dest='cosmul', action='store_true',
                        help='Use 3COSMUL when evaluating word analogy.')
    parser.add_argument('--distadd', dest='distadd', action='store_true',
                        help='Use 3DISTADD when evaluating word analogy.')
    parser.add_argument('--hypcosadd', dest='hypcosadd', action='store_true',
                        help='Use 3COSADD with gyrocosine when evaluating word analogy.')
    parser.add_argument('--agg_eval', dest='agg_eval', action='store_true',
                        help='Use w+c during evaluation, instead of just w. Only works for Poincare embeddings.')
    parser.add_argument('--ctx_eval', dest='ctx_eval', action='store_true',
                        help='Use c during evaluation, instead of w.')
    parser.add_argument('--cosine_eval', dest='cosine_eval', action='store_true',
                        help='Use cosine distance during evaluation, instead of the Poincare distance.')
    parser.add_argument('--ckpt_emb', dest='ckpt_emb', action='store_true',
                        help='Store checkpoints during training with the value of the embedding for certain words')
    parser.add_argument('--init_near_border', dest='init_near_border', action='store_true',
                        help='If set, initialize embeddings near the Poincare ball border, instead of near the origin.')
    parser.add_argument('--init_pretrained', dest='init_pretrained', action='store_true',
                        help='If set, initialize embeddings from pretrained model.')
    parser.add_argument('--use_log_probs', dest='use_log_probs', action='store_true',
                        help='If set, use log-probabilities instead of log-counts during training GloVe.')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                        help='Run model in debug mode')
    parser.set_defaults(train=False, use_glove_format=True, mix=False, with_bias=False, use_scaling=False,
                        cosadd=False, cosmul=False, distadd=False, hypcosadd=False, cosine_eval=False,
                        agg_eval=False, ctx_eval=False, shift_origin=False, cosine_dist=False, ckpt_emb=False,
                        init_near_border=False, init_pretrained=False, use_log_probs=False, is_debug=False)
    args = parser.parse_args()

    if args.size > 4 and args.size % 4 != 0:
        raise RuntimeError("Choose an embedding size that is a multiple of 4 (it speeds up computation)")

    model = None
    if args.train:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        callbacks = []

        emb_type = None
        if args.poincare == 1:
            emb_type = 'poincare'
        elif args.euclid == 1:
            emb_type = 'euclid'
        else:
            emb_type = 'vanilla'

        if args.mix:
            emb_type = "mix-" + emb_type

        coocc_func = args.coocc_func
        if coocc_func == "":
            coocc_func = SUPPORTED_COOCC_FUNCTIONS[0]  # Set the default
        else:
            if coocc_func not in SUPPORTED_COOCC_FUNCTIONS:
                raise RuntimeError("Unsupported co-occurrence function {}".format(coocc_func))

        cosh_dist_pow = 0
        dist_func = args.dist_func
        if emb_type == "poincare" or emb_type == "mix-poincare" or emb_type == "euclid":
            if dist_func == "":
                dist_func = SUPPORTED_DIST_FUNCTIONS[emb_type][0]  # Set the default
            elif "cosh-dist-pow" in dist_func:
                cosh_dist_pow = int(dist_func.rsplit("-", 1)[1])
            else:
                if dist_func not in SUPPORTED_DIST_FUNCTIONS[emb_type]:
                    raise RuntimeError("Unsupported distance function {} for emb type {}".format(dist_func, emb_type))
        num_embs = 0
        if "mix-" in emb_type:
            if args.num_embs == 0:
                raise RuntimeError("Invalid number of small embeddings.")
            num_embs = args.num_embs

        if args.num_embs != 0 and "mix-" not in emb_type:
            raise RuntimeError("num_embs is not supported for this embedding type: {}".format(emb_type))

        nn_config = None
        if dist_func == "nn":
            if args.nn_config == "":
                raise RuntimeError("No NN configuration provided!")
            nn_config = NNConfig(args.nn_config)

        optimizer = args.optimizer
        if optimizer == "":
            optimizer = SUPPORTED_OPTIMIZERS[emb_type][0]  # Set the default
        else:
            if optimizer not in SUPPORTED_OPTIMIZERS[emb_type]:
                raise RuntimeError("Unsupported optimizer {} for embedding type {}".format(optimizer, emb_type))

        filename = MODEL_FILENAME_PATTERN.format(
            "glove_baseline" if emb_type == "vanilla" else "geometric_emb",
            args.epochs, args.size, str(args.lr), args.restrict_vocab, emb_type)
        filename = filename + "_OPT" + optimizer
        filename = filename + "_COOCCFUNC" + coocc_func
        if emb_type != "vanilla":
            filename = filename + "_DISTFUNC" + dist_func
        elif emb_type == "vanilla" and dist_func == "nn":
            filename = filename + "_DISTFUNCnn"
        if dist_func == "nn":
            filename = filename + "_NN" + args.nn_config
        if num_embs:
            filename = filename + "_NUMEMBS" + str(num_embs)
        if args.with_bias:
            filename = filename + "_bias"
        if args.use_scaling:
            if emb_type != "poincare" and emb_type != "mix-poincare":
                raise RuntimeError("Scaling is only supported for Poincare GloVe embeddings.")
            filename = filename + "_scale"
        if args.use_log_probs:
            filename = filename + "_logprobs"
        if args.init_near_border:
            filename = filename + "_border-init"
        initialization_config = None
        if args.init_pretrained:
            if emb_type == "poincare" and args.size == 100:
                pretrained_model_filename = INITIALIZATION_MODEL_FILENAME["100D"]
            elif emb_type == "mix-poincare" and args.size == 100 and num_embs == 50:
                pretrained_model_filename = INITIALIZATION_MODEL_FILENAME["50x2D"]
            elif emb_type == "vanilla" and args.size == 100:
                pretrained_model_filename = INITIALIZATION_MODEL_FILENAME["vanilla_100D"]
            else:
                raise RuntimeError("Undefined pretrained embedding for this setting.")
            print("Initializing embeddings from pretrained model", pretrained_model_filename)
            initialization_config = InitializationConfig(
                pretrained_model_filename=os.path.join(args.root, pretrained_model_filename)
            )
            filename = filename + "_INITpretrained"
        model_filename = os.path.join(args.root, filename)

        ckpt_word_list = None
        if args.ckpt_emb:
            with open(os.path.join(args.root, "msc_tifreaa/data/google_analogy_vocab.txt"), "r") as f:
                ckpt_word_list = [word.strip() for word in f.readlines()]
            ckpt_filename = "word_emb_checkpoints/emb_ckpt_" + os.path.basename(model_filename)
            ckpt_filename = os.path.join(args.root, ckpt_filename)
            callbacks.append(WordEmbCheckpointSaver(ckpt_filename=ckpt_filename))

        print("[Training] Train new model {} using {}".format(model_filename, optimizer.upper()), end="")
        if emb_type == "poincare":
            print(" and distance function {}".format(dist_func.upper()))
        else:
            print("")

        model = Glove(
            use_glove_format=args.use_glove_format,
            coocc_file=args.coocc_file,
            vocab_file=args.vocab_file,
            restrict_vocab=args.restrict_vocab,
            num_workers=args.workers,
            chunksize=args.chunksize,
            epochs=args.epochs,
            euclid=args.euclid,
            poincare=args.poincare,
            with_bias=args.with_bias,
            use_log_probs=args.use_log_probs,
            dist_func=dist_func,
            cosh_dist_pow=cosh_dist_pow,
            num_embs=num_embs,
            nn_config=nn_config,
            coocc_func=coocc_func,
            use_scaling=args.use_scaling,
            lr=args.lr,
            optimizer=optimizer,
            ckpt_word_list=ckpt_word_list,
            init_near_border=args.init_near_border,
            init_pretrained_config=initialization_config,
            callbacks=callbacks,
            vector_size=args.size,
            vector_dtype=REAL)

        if args.use_scaling:
            print("Final scaling factor is {}".format(model.scaling_factor))

        if optimizer == "wfullrsgd" or optimizer == "fullrsgd":
            logging.info("")
            logging.info("Number of projections back to the Poincare ball: {}".format(model.num_projections))

        # Cleanup model.
        model.cleanup()

        # Save model.
        print("Saving model to {}".format(model_filename))
        with open(model_filename, "wb") as f:
            model.save(f)
    else:
        model = Glove.load(args.model_filename)
        wv = model.wv
        wv.trainables = model.trainables

        # XXX: uncomment to evaluate the model with the scaled and projected pretrained embeddings used for initialization
        # wv.vectors = model.trainables.initialization_config.init_vectors

        directory = os.path.join(args.root, "eval_logs")
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Extract model info from the model filename.
        config, basename = parse_model_filename(args.model_filename)

        # Ugly fix. To ensure backward compatibilty with an earlier version that used a different convention for the filename.
        if config is None:
            config = {}
            config["similarity"] = "poincare" if ("poincare" in basename) else ("euclid" if ("euclid" in basename) else "vanilla")

        analogy_type = None
        if args.cosadd:
            analogy_type = "cosadd"
        elif args.cosmul:
            analogy_type = "cosmul"
        elif args.distadd:
            analogy_type = "distadd"
        elif args.hypcosadd:
            analogy_type = "hypcosadd"
        elif config["similarity"] == "poincare":
            if args.cosine_eval:
                analogy_type = "hyp_pt-eucl-cos-dist"
            else:
                analogy_type = "hyp_pt"
        elif config["similarity"] == "mix-poincare":
            if args.cosine_eval:
                analogy_type = "mix-hyp_pt-eucl-cos-dist"
            else:
                analogy_type = "mix-hyp_pt"
        else:
            analogy_type = "cosadd"  # The default for dot product and Euclidean embeddings is 3COSADD

        if config["similarity"] == "mix-poincare" and "num_embs" not in config:
            raise RuntimeError("Mix Poincare embeddings should have a valid number of small embeddings")

        if args.agg_eval:
            compute_poincare_aggregate(model, config)

        if args.ctx_eval:
            model.wv.vectors = model.trainables.syn1neg

        if args.shift_origin:
            left_offset = -np.average(model.wv.vectors, axis=0)
            # right_offset = -np.average(model.wv.vectors, axis=0)
            left_offset_mat = np.tile(left_offset.reshape(1, -1), (model.wv.vectors.shape[0], 1))
            # right_offset_mat = np.tile(right_offset.reshape(1, -1), (model.wv.vectors.shape[0], 1))

            model.wv.vectors = gensim.models.keyedvectors.PoincareWordEmbeddingsKeyedVectors.moebius_add_mat(
                left_offset_mat, model.wv.vectors)
            # model.wv.vectors = gensim.models.keyedvectors.PoincareWordEmbeddingsKeyedVectors.moebius_add_mat(
            #     model.wv.vectors, right_offset_mat)

        if config["similarity"] == "poincare" or config["similarity"] == "mix-poincare":
            if args.cosine_eval:
                model.wv.use_poincare_distance = False
                model.wv.init_sims()
            else:
                model.wv.use_poincare_distance = True

        # Create name for file that will store the logs.
        eval_log_filename = "eval_logs/eval_" + basename.split("_", 1)[1] + "_" + analogy_type + \
                            ("_agg" if args.agg_eval else ("_ctx" if args.ctx_eval else ""))
        # eval_log_filename = eval_log_filename + ("_cosdist" if config["similarity"] == "poincare" and args.cosine_eval else "")
        eval_log_filename = os.path.join(args.root, eval_log_filename)
        feval = None
        if args.restrict_vocab != 0:
            feval = open(eval_log_filename, "w+")
            logger = Logger(feval)
        else:
            # Don't save the output to file if we are not running the word analogy benchmarks.
            logger = Logger()

        if len(config) > 1:
            logger.log('MODEL: (Epochs, {}), (Emb size, {}), (LR, {}), (Optimizer, {}), (With bias, {}), (Similarity, {}), (Dist. func. {}), (Scaling, {}), (Use Log-Probs, {}), (Restrict vocab, {})'.format(
                config["epochs"], config["emb_size"], config["lr"], config["optimizer"].upper(), "yes" if config["with_bias"] else "no",
                config["similarity"], config["dist_func"].upper(), config["use_scaling"], config["use_log_probs"],
                config["restrict_vocab"]
            ))

        if args.restrict_vocab != 0:
            logger.log('EVALUATION: (Analogy type, {}), (Vectors used, {})'.format(
                analogy_type, ("W+C" if args.agg_eval else ("C" if args.ctx_eval else "W"))))
        else:
            logger.log()

        sim_debug_file = None
        if args.is_debug:
            sim_debug_file = os.path.join(args.root, "eval_logs/debug_similarity.csv")
        hyperlex_debug_file = None
        if args.is_debug:
            hyperlex_debug_file = os.path.join(args.root, "eval_logs/debug_hyperlex.csv")

        # logger.log("========= Various statistics =========")
        # norms_distribution(model)
        # wordnet_level_rank_vector_norm_correlation(model, args.root)

        logger.log("========= Similarity evaluation =========")
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'rare_word.txt'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("Stanford Rare World: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'wordsim353.tsv'),
            dummy4unknown=False,
            debug_file=sim_debug_file,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("WordSim353: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'simlex999.txt'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("SimLex999: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'mturk771.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("MTurk771: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'simverb3500.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("SimVerb3500: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'men_dataset.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("MEN: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'MC-30.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("MC: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'RG-65.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("RG: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'YP-130.tsv'),
            dummy4unknown=False,
            restrict_vocab=args.restrict_vocab
        )
        logger.log("YP: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))

        if args.restrict_vocab != 0:
            logger.log("=========== Analogy evaluation ==========")
            most_similar = None
            if analogy_type == "cosadd" or analogy_type == "hypcosadd":
                most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy
            elif analogy_type == "cosmul":
                most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.batch_most_similar_cosmul_analogy
            elif analogy_type == "hyp_pt" or analogy_type == "hyp_pt-eucl-cos-dist":
                most_similar = gensim.models.keyedvectors.PoincareWordEmbeddingsKeyedVectors.batch_most_similar_hyperbolic_analogy
            elif analogy_type == "mix-hyp_pt" or analogy_type == "mix-hyp_pt-eucl-cos-dist":
                most_similar = gensim.models.keyedvectors.MixPoincareWordEmbeddingsKeyedVectors.batch_most_similar_mix_hyperbolic_analogy
            elif analogy_type == "distadd":
                most_similar = gensim.models.keyedvectors.PoincareWordEmbeddingsKeyedVectors.batch_most_similar_3distadd_analogy
            else:
                raise RuntimeError("Unknown analogy type.")

            print(config["similarity"], analogy_type)
            if (config["similarity"] == "mix-poincare" or config["similarity"] == 'poincare') and (analogy_type == "cosadd" or analogy_type == "cosmul"):
                model.wv.vectors_norm = None
                gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.init_sims(model.wv)
            if config["similarity"] == "poincare" and args.cosine_eval:
                model.wv.vectors_norm = None
                gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.init_sims(model.wv)

            start = time.time()
            analogy_eval = wv.accuracy(
                    os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'questions-words.txt'),
                    restrict_vocab=args.restrict_vocab,
                    most_similar=most_similar,
                    debug=args.is_debug)
            # Now, instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers.
            logger.log("Semantic Google: {} {} {:.2f} {:.4f} {:.4f}".format(analogy_eval[-3]['correct'][0],
                                                                    analogy_eval[-3]['correct'][0] + analogy_eval[-3]['incorrect'][0],
                                                                    analogy_eval[-3]['t_argmax'][0],
                                                                    analogy_eval[-3]['correct'][0] / (analogy_eval[-3]['correct'][0] + analogy_eval[-3]['incorrect'][0]),
                                                                    analogy_eval[-3]['correct'][0] / SEM_GOOGLE_SIZE))
            logger.log("Syntactic Google: {} {} {:.2f} {:.4f} {:.4f}".format(analogy_eval[-2]['correct'][0],
                                                                    analogy_eval[-2]['correct'][0] + analogy_eval[-2]['incorrect'][0],
                                                                    analogy_eval[-2]['t_argmax'][0],
                                                                    analogy_eval[-2]['correct'][0] / (analogy_eval[-2]['correct'][0] + analogy_eval[-2]['incorrect'][0]),
                                                                    analogy_eval[-2]['correct'][0] / SYN_GOOGLE_SIZE))
            logger.log("Google: {} {} {:.2f} {:.4f} {:.4f}".format(analogy_eval[-1]['correct'][0],
                                                    analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                                    analogy_eval[-1]['t_argmax'][0],
                                                    analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                                    analogy_eval[-1]['correct'][0] / GOOGLE_SIZE))

            if not args.is_debug:
                analogy_eval = wv.accuracy(
                        os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data/', 'msr_word_relationship.processed'),
                        restrict_vocab=args.restrict_vocab,
                        most_similar=most_similar)
                # Now, instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers.
                logger.log("Microsoft: {} {} {:.2f} {:.4f} {:.4f}".format(analogy_eval[-1]['correct'][0],
                                                           analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                                           analogy_eval[-1]['t_argmax'][0],
                                                           analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                                           analogy_eval[-1]['correct'][0] / MSR_SIZE))
            logging.info("")
            logging.info("Analogy task took {} seconds to perform.".format(time.time() - start))
        if feval:
            feval.close()
