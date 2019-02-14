#!/usr/local/bin/python3

import argparse
import gensim
from gensim.models.callbacks import LossLogger, LossSetter, VectorNormLogger, WordEmbCheckpointSaver
from gensim.models.word2vec import InitializationConfig
from util_scripts.get_model_eval_and_stats import *
import logging
from nltk.corpus import brown
import numpy as np
from numpy.linalg import norm
import os
import time

WIKI_PATH = '/media/hofmann-scratch/other-data/Wikipedia/WikipediaPlainText/textFromAllWikipedia2014Feb.txt_one_doc_per_line'
LEVY_PATH = 'data/levy_dataset/levy_wikipedia_dataset'
TEXT8_FILE = "data/text8/text8"
MODEL_FILENAME_PATTERN = "models/{}/w2v_{}_{}_{}_{}_A{}_a{}_n{}_w{}_c{}_{}"
INITIALIZATION_MODEL_FILENAME_PATTERN = "data/pretrained_models/word2vec_pretrained_{}_{}"

# IMPORTANT!!!!!!!!!!! First one for each embedding type should be the default.
SUPPORTED_OPTIMIZERS = {
    "cosine": ["sgd", "wsgd", "rmsprop"],
    "euclid": ["sgd"],
    "poincare": ["rsgd", "wrsgd", "fullrsgd", "wfullrsgd", "rmsprop"],
}

# IMPORTANT!! First one is the default.
# This refers to the similarity function used in the NLL loss during training.
SUPPORTED_SIM_FUNCTIONS = [
    "dist-sq", "cosh-dist-sq", "cosh-dist-pow-*", "cosh-dist", "log-dist", "log-dist-sq", "exp-dist"
]

GOOGLE_SIZE = 19544
MSR_SIZE = 8000

logging.basicConfig(level=logging.INFO)

def precision(eval_result):
    return len(eval_result['correct']) / (len(eval_result['correct']) + len(eval_result['incorrect']))


def get_sentences(dataset_name):
    if dataset_name == 'brown':
        data = brown.sents()
    elif dataset_name == 'text8':
        data = gensim.models.word2vec.Text8Corpus(os.path.join(args.root, TEXT8_FILE))
    elif dataset_name == 'wiki':
        data = gensim.models.word2vec.LineSentence(WIKI_PATH)
    elif dataset_name == 'levy':
        data = gensim.models.word2vec.LineSentence(os.path.join(args.root, LEVY_PATH))
    else:
        raise RuntimeError('Invalid dataset name')
    return data


def split_filename(basename):
    info = basename.split('_')

    # Ugly fix. Insert default values for backward compatibility.
    if len(info) < 9:
        info.insert(5, "A025")
        info.insert(6, "a0001")
        info.insert(7, "n5")
        info.insert(8, "w5")
        info[9] = "c" + str(info[9])
    if info[6][0] != 'a':
        info.insert(6, "a0001")

    return info


def compute_poincare_aggregate(model):
    """
    Precompute the average between the target and the context vector, for Poincare embeddings.
    We take as average the mid point between w and c on the geodesic that connects the 2 points
    (see page 89 in Ungar book).
    """
    if model.poincare and getattr(model.wv, 'agg_vectors', None) is None:
        print("precomputing aggregated vectors w+c for Poincare embeddings")
        gamma_w_sq = 1 / (1 - np.sum(model.wv.vectors * model.wv.vectors, axis=1))
        gamma_c_sq = 1 / (1 - np.sum(model.trainables.syn1neg * model.trainables.syn1neg, axis=1))
        denominator = gamma_w_sq + gamma_c_sq - 1
        agg = (model.wv.vectors * (gamma_w_sq / denominator)[:, None] +
               model.trainables.syn1neg * (gamma_c_sq / denominator)[:, None])

        model.wv.vectors = model.wv.moebius_mul_mat(agg, 0.5)


def parse_model_filename(model_filename):
    info_dict = {}
    basename = os.path.basename(model_filename)
    info = split_filename(basename)
    info_dict["dataset"] = info[1]
    info_dict["w2v_model"] = info[2]
    info_dict["epochs"] = int(info[3])
    info_dict["emb_size"] = int(info[4])
    info_dict["alpha"] = float("0." + info[5][1:])
    info_dict["min_alpha"] = float("0." + info[6][1:])
    info_dict["negative"] = int(info[7][1:])
    info_dict["window"] = int(info[8][1:])
    info_dict["min_count"] = int(info[9][1:])
    info_dict["similarity"] = info[10]
    info_dict["l2reg_coef"] = float(info[11][1:]) if len(info) > 11 and info[11][0] == "l" else 0.0
    info_dict["with_bias"] = True if "_bias" in basename else False
    info_dict["init_near_border"] = True if "_border-init" in basename else False
    info_dict["normalized"] = True if "_norm" in basename else False

    burnin_info = list(filter(lambda x: x.startswith("burnin"), info))
    info_dict["burnin_epochs"] = int(burnin_info[0][6:]) if len(burnin_info) == 1 else 0

    for s in info:
        if "OPT" in s:
            info_dict["optimizer"] = s[3:]
        elif "INIT" in s:
            info_dict["init_config"] = s[4:]
        elif "SIM" in s:
            info_dict["sim_func"] = s[3:]
    if "optimizer" not in info_dict:
        info_dict["optimizer"] = SUPPORTED_OPTIMIZERS[info_dict["similarity"]][0]
    if "init_config" not in info_dict:
        info_dict["init_config"] = None
    if "sim_func" not in info_dict:
        info_dict["sim_func"] = SUPPORTED_SIM_FUNCTIONS[0]

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
    parser.add_argument('--ds', type=str, default='brown',
                        help='Dataset on which to train the model.')
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train a new model.')
    parser.add_argument('--eval', dest='train', action='store_false',
                        help='Eval an existing model.')
    parser.add_argument('--restrict_vocab', type=int, default=100000,
                        help='Size of vocab. Only used for evaluating analogy.')
    parser.add_argument('--cosadd', dest='cosadd', action='store_true',
                        help='Use 3COSADD when evaluating word analogy.')
    parser.add_argument('--cosmul', dest='cosmul', action='store_true',
                        help='Use 3COSMUL when evaluating word analogy.')
    parser.add_argument('--sg', type=int, default=1, help='Choose W2V model. 1 = skip-gram; 0 = CBOW')
    parser.add_argument('--alpha', type=float, default=0.025, help='Initial learning rate')
    parser.add_argument('--min_alpha', type=float, default=0.0001, help='Min learning rate')
    parser.add_argument('--l2reg', type=float, default=0.0, help='The coefficient of L2 regularization')
    parser.add_argument('--optimizer', type=str, default='', help='What optimizer to use.')
    parser.add_argument('--init_config', type=str, default='',
                        help='The initialization configuration to use. Should be in the format '
                             '<euclid2hyp_method><scaling_factor> e.g. exp0.1 or id0.01')
    parser.add_argument('--sim_func', type=str, default="", help='Similarity function used by Poincare model.')
    parser.add_argument('--size', type=int, default=100, help='Embedding size')
    parser.add_argument('--negative', type=int, default=5, help='Number of negative samples that are considered')
    parser.add_argument('--window', type=int, default=5, help='Sliding window size')
    parser.add_argument('--nll', dest='is_nll', action='store_true', help='Use NLL loss instead of NegSampling')
    parser.add_argument('--burnin_epochs', type=int, default=0, help='Number of burn-in epochs, before training')
    parser.add_argument('--normalized', dest='normalized', action='store_true',
                        help='Normalize word vectors to unit norm after each update.')
    parser.add_argument('--euclid', type=int, default=0, help='Whether it uses Euclidean distance for training or not.')
    parser.add_argument('--poincare', type=int, default=0, help='Whether it uses Poincare embeddings or not.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Ignores all words with total frequency lower than this.')
    parser.add_argument('--model_filename', type=str, default='', help='Path to saved model.')
    parser.add_argument('--train_log_filename', type=str, default='', help='Path to the training log.')
    parser.add_argument('--workers', type=int, default=3, help='Number of concurrent workers.')
    parser.add_argument('--bias', dest='with_bias', action='store_true', help='Use a model with biases.')
    parser.add_argument('--init_near_border', dest='init_near_border', action='store_true',
                        help='If set, initialize embeddings near the Poincare ball border, instead of near the origin.')
    parser.add_argument('--agg_eval', dest='agg_eval', action='store_true',
                        help='Use w+c during evaluation, instead of just w. Only works for Poincare embeddings.')
    parser.add_argument('--ctx_eval', dest='ctx_eval', action='store_true',
                        help='Use c during evaluation, instead of w.')
    parser.add_argument('--shift_origin', dest='shift_origin', action='store_true',
                        help='Shift the origin of the points before evaluation.')
    parser.add_argument('--cosine_eval', dest='cosine_eval', action='store_true',
                        help='Use cosine distance during evaluation, instead of the Poincare distance.')
    parser.add_argument('--ckpt_emb', dest='ckpt_emb', action='store_true',
                        help='Store checkpoints during training with the value of the embedding for certain words')
    parser.add_argument('--debug', dest='is_debug', action='store_true',
                        help='Run model in debug mode')
    parser.set_defaults(train=False, cosadd=False, cosmul=False, is_nll=False, normalized=False,
                        with_bias=False, init_near_border=False, agg_eval=False, ctx_eval=False, shift_origin=False,
                        cosine_dist=False, ckpt_emb=False, is_debug=False)
    args = parser.parse_args()

    if args.size > 4 and args.size % 4 != 0:
        raise RuntimeError("Choose an embedding size that is a multiple of 4 (it speeds up computation)")

    model = None
    if args.train:
        callbacks = [
            LossSetter(),
            LossLogger(log_file=args.train_log_filename),
            VectorNormLogger(log_file=args.train_log_filename)]
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        if args.model_filename:
            # Continue training an existing model. All hyperparameters will be extracted from the filename. The
            # number of additional epochs for which we want to train needs to be provided as parameter. The new
            # model will be stored in a file with the same name, but with a different number of epochs in the name.

            # Extract model info from the model filename.
            config, basename = parse_model_filename(args.model_filename)

            print("[Training] Continue training model {} for an additional {} epochs".format(basename, args.epochs))

            sentences = get_sentences(config["dataset"])

            # Load and train model.
            model = gensim.models.Word2Vec.load(args.model_filename)
            if abs(model.min_alpha_yet_reached - args.min_alpha) < 0.0001:
                print("[WARNING] The learning rate will be constant throughout training!")
            model.train(
                sentences, total_examples=model.corpus_count, epochs=args.epochs,
                start_alpha=args.alpha, end_alpha=args.min_alpha, compute_loss=True,
                callbacks=callbacks)

            # Update filename with the new number of epochs.
            filename = MODEL_FILENAME_PATTERN.format(
                "word2vec_baseline" if config["similarity"] == "cosine" else "geometric_emb",
                config["dataset"], config["w2v_model"], (config["epochs"] + args.epochs), config["emb_size"], str(args.alpha)[2:], str(args.min_alpha)[2:], config["negative"],
                config["window"], config["min_count"], config["similarity"])
            if config["l2reg_coef"]:
                filename = filename + "_l" + str(config["l2reg_coef"])
            if config["with_bias"]:
                filename = filename + "_bias"
            if config["init_near_border"]:
                filename = filename + "_border-init"
            if config["normalized"]:
                filename = filename + "_norm"
            if config["burnin_epochs"] != 0:
                filename = filename + "_burnin" + str(config["burnin_epochs"])
            filename += "_cont"
            new_model_filename = os.path.join(args.root, filename)
        else:
            sentences = get_sentences(args.ds)

            emb_type = None
            if args.euclid == 1:
                emb_type = 'euclid'
            elif args.poincare == 1:
                emb_type = 'poincare'
            else:
                emb_type = 'cosine'

            optimizer = args.optimizer
            if optimizer == "":
                optimizer = SUPPORTED_OPTIMIZERS[emb_type][0]  # Set the default
            else:
                if optimizer not in SUPPORTED_OPTIMIZERS[emb_type]:
                    raise RuntimeError("Unsupported optimizer {} for embedding type {}".format(optimizer, emb_type))

            sim_func = args.sim_func
            cosh_dist_pow = 0.0
            if sim_func != "" and emb_type != "poincare":
                raise RuntimeError("Choosing a different similarity function is only supported for poincare embeddings")
            if emb_type == "poincare":
                if sim_func == "":
                    sim_func = SUPPORTED_SIM_FUNCTIONS[0]  # Set the default
                elif "cosh-dist-pow" in sim_func:
                    cosh_dist_pow = int(sim_func.rsplit("-", 1)[1])
                else:
                    if sim_func not in SUPPORTED_SIM_FUNCTIONS:
                        raise RuntimeError("Unsupported similarity function {}".format(sim_func))

            model_type = None
            if args.sg == 1:
                if args.is_nll:
                    model_type = "nll"
                else:
                    model_type = "sg"
            else:
                model_type = "cbow"

            # Check if folders exist and create them otherwise.
            directory = os.path.join(args.root, "eval_logs")
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.join(args.root, "models")
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.join(args.root, "word_emb_checkpoints")
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.join(args.root, "models/word2vec_baseline")
            if not os.path.exists(directory):
                os.makedirs(directory)
            directory = os.path.join(args.root, "models/geometric_emb")
            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = MODEL_FILENAME_PATTERN.format(
                "word2vec_baseline" if args.euclid == 0 and args.poincare == 0 else "geometric_emb",
                args.ds, model_type, args.epochs, args.size, str(args.alpha)[2:],
                str(args.min_alpha)[2:], args.negative, args.window, args.min_count, emb_type)
            filename = filename + "_OPT" + optimizer
            if emb_type == "poincare":
                filename = filename + "_SIM" + sim_func
            initialization_config = None
            if args.init_config:
                pretrained_model_filename = INITIALIZATION_MODEL_FILENAME_PATTERN.format(args.ds, args.size)
                print("Initializing embeddings from pretrained model", pretrained_model_filename)
                initialization_config = InitializationConfig(
                    pretrained_model_filename=os.path.join(args.root, pretrained_model_filename),
                    config_str=args.init_config
                )
                filename = filename + "_INIT" + args.init_config
            if args.l2reg:
                filename = filename + "_l" + str(args.l2reg)
            if args.with_bias:
                filename = filename + "_bias"
            if args.init_near_border:
                filename = filename + "_border-init"
            if args.normalized:
                filename = filename + "_norm"
            if args.burnin_epochs != 0:
                filename = filename + "_burnin" + str(args.burnin_epochs)
            new_model_filename = os.path.join(args.root, filename)

            ckpt_word_list = None
            if args.ckpt_emb:
                with open(os.path.join(args.root, "data/google_analogy_vocab.txt"), "r") as f:
                    ckpt_word_list = [word.strip() for word in f.readlines()]
                ckpt_filename = "word_emb_checkpoints/emb_ckpt_" + os.path.basename(new_model_filename).split("_", 1)[1]
                ckpt_filename = os.path.join(args.root, ckpt_filename)
                callbacks.append(WordEmbCheckpointSaver(ckpt_filename=ckpt_filename))

            print("[Training] Train new model {} using {}".format(new_model_filename, optimizer.upper()), end="")
            if emb_type == "poincare":
                print(" and similarity function {}".format(sim_func.upper()))
            else:
                print("")

            # The first input to Word2Vec is a list of lists of strings. Each item in
            # the top-level list is a list of the words and special punctuation
            # (e.g. . or "). One such item corresponds to one sentence.
            model = gensim.models.Word2Vec(
                sentences,
                sg=args.sg,
                is_nll=args.is_nll,
                normalized=args.normalized,
                burnin_epochs=args.burnin_epochs,
                euclid=args.euclid,
                poincare=args.poincare,
                size=args.size,
                alpha=args.alpha,
                min_alpha=args.min_alpha,
                l2reg_coef=args.l2reg,
                optimizer=optimizer,
                sim_func=sim_func,
                cosh_dist_pow=cosh_dist_pow,
                negative=args.negative,
                window=args.window,
                min_count=args.min_count,
                iter=args.epochs,
                workers=args.workers,
                compute_loss=True,
                with_bias=args.with_bias,
                init_near_border=args.init_near_border,
                initialization_config=initialization_config,
                ckpt_word_list=ckpt_word_list,
                debug=args.is_debug,
                callbacks=callbacks)

        # Save model.
        with open(new_model_filename, "wb") as f:
            model.save(f)

        # Sanity check the norm of some random words if the word embeddings need to be normalized.
        if args.normalized:
            logging.info("")
            logging.info("Sanity check-normalized word embeddings (word: norm):")
            words = ["dog", "man", "king", "usa", "something"]
            for w in words:
                logging.info("\t {}: {}".format(w, norm(model.wv[w])))
        if args.with_bias:
            logging.info("")
            logging.info("Sanity check-sum of: input biases {}; output biases {}".format(
                np.sum(model.trainables.b0), np.sum(model.trainables.b1)))

        if optimizer == "wfullrsgd" or optimizer == "fullrsgd":
            logging.info("")
            logging.info("Number of projections back to the Poincare ball: {}".format(model.num_projections))
    else:
        model = gensim.models.Word2Vec.load(args.model_filename)
        wv = model.wv
        wv.trainables = model.trainables

        # XXX: uncomment to evaluate the model with the scaled and projected pretrained embeddings used for initialization
        # wv.vectors = model.trainables.initialization_config.init_vectors

        directory = os.path.join(args.root, "eval_logs")
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Extract model info from the model filename.
        basename = os.path.basename(args.model_filename)
        config, basename = parse_model_filename(args.model_filename)

        analogy_type = None
        if args.cosadd:
            analogy_type = "cosadd"
        elif args.cosmul:
            analogy_type = "cosmul"
        elif config["similarity"] == "poincare":
            analogy_type = "hyp_pt"
        else:
            analogy_type = "cosadd"  # The default for dot product and Euclidean embeddings is 3COSADD

        if config["similarity"] == "poincare" and args.agg_eval:
            compute_poincare_aggregate(model)

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

        if config["similarity"] == "poincare":
            if args.cosine_eval:
                model.wv.use_poincare_distance = False
            else:
                model.wv.use_poincare_distance = True

        # Create name for file that will store the logs.
        eval_log_filename = "eval_logs/eval_" + basename.split("_", 1)[1] + "_" + analogy_type + \
                            ("_agg" if config["similarity"] == "poincare" and args.agg_eval else ("_ctx" if args.ctx_eval else ""))
        eval_log_filename = eval_log_filename + ("_cosdist" if config["similarity"] == "poincare" and args.cosine_eval else "")
        eval_log_filename = os.path.join(args.root, eval_log_filename)
        feval = None
        if args.restrict_vocab != 0:
            feval = open(eval_log_filename, "w+")
            logger = Logger(feval)
        else:
            # Don't save the output to file if we are not running the word analogy benchmarks.
            logger = Logger()

        l2reg_coef = str(config["l2reg_coef"])
        logger.log('MODEL: (Dataset, {}), (W2V model, {}), (Epochs, {}), (Emb size, {}), (Alpha, {}), (Min alpha, {}), (Negative, {}), (Window, {}), (Similarity, {}), (Optimizer, {}), (Sim. func, {}), (Init config, {}), (With Bias, {}), (Normalized, {}), (L2Reg Coeff, {}), (Burn-in, {})'.format(
            config["dataset"], config["w2v_model"], config["epochs"], config["emb_size"], config["alpha"],
            config["min_alpha"], config["negative"], config["window"], config["similarity"],
            config["optimizer"].upper(), config["sim_func"].upper(), config["init_config"],
            "yes" if config["with_bias"] else "no", "yes" if config["normalized"] else "no",
            config["l2reg_coef"], "yes" if config["burnin_epochs"] else "no"
        ))

        if args.restrict_vocab != 0:
            logger.log('EVALUATION: (Analogy type, {}), (Restrict vocab, {}), (Vectors used, {}), (Distance, {})'.format(
                analogy_type, args.restrict_vocab, ("W+C" if args.agg_eval else ("C" if args.ctx_eval else "W")),
                ("cosine" if config["similarity"] != "poincare" or args.cosine_eval else "Poincare")))
        else:
            logger.log()

        sim_debug_file = None
        if args.is_debug:
            sim_debug_file = os.path.join(args.root, "eval_logs/debug_similarity.csv")

        logger.log("========= Various statistics =========")
        norms_distribution(model)
        wordnet_level_rank_vector_norm_correlation(model, args.root)

        logger.log("========= Similarity evaluation =========")
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'rare_word.txt'),
            dummy4unknown=False
        )
        logger.log("Stanford Rare World: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'wordsim353.tsv'),
            dummy4unknown=False,
            debug_file=sim_debug_file
        )
        logger.log("WordSim353: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))
        pearson, spearman, ratio = wv.evaluate_word_pairs(
            os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'simlex999.txt'),
            dummy4unknown=False
        )
        logger.log("SimLex999: {:.4f} {:.4f} {:.4f}".format(pearson[0], spearman[0], ratio))

        if args.restrict_vocab != 0:
            logger.log("=========== Analogy evaluation ==========")
            most_similar = None
            if analogy_type == "cosadd":
                most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy
            elif analogy_type == "cosmul":
                most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.batch_most_similar_cosmul_analogy
            elif analogy_type == "hyp_pt":
                most_similar = gensim.models.keyedvectors.PoincareWordEmbeddingsKeyedVectors.batch_most_similar_hyperbolic_analogy
            else:
                raise RuntimeError("Unknown analogy type.")
            start = time.time()
            analogy_eval = wv.accuracy(
                    os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data', 'questions-words.txt'),
                    restrict_vocab=args.restrict_vocab,
                    most_similar=most_similar,
                    debug=args.is_debug)
            # Now, instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers.
            logger.log("Google: {} {} {} {}".format(analogy_eval[-1]['correct'][0],
                                                    analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                                    analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                                    analogy_eval[-1]['correct'][0] / GOOGLE_SIZE))

            if not args.is_debug:
                analogy_eval = wv.accuracy(
                        os.path.join(args.root, 'msc_tifreaa/gensim/test/test_data/', 'msr_word_relationship.processed'),
                        restrict_vocab=args.restrict_vocab,
                        most_similar=most_similar)
                # Now, instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers.
                logger.log("Microsoft: {} {} {} {}".format(analogy_eval[-1]['correct'][0],
                                                           analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                                           analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                                           analogy_eval[-1]['correct'][0] / MSR_SIZE))
            logging.info("")
            logging.info("Analogy task took {} seconds to perform.".format(time.time() - start))
        if feval:
            feval.close()
