from gensim.models.word2vec import WordEmbeddingCheckpoints
from gensim.utils import SaveLoad
from gensim.models.keyedvectors import VanillaWordEmbeddingsKeyedVectors, PoincareWordEmbeddingsKeyedVectors, Vocab, \
    MixPoincareWordEmbeddingsKeyedVectors
from glove_code.src.glove_inner import read_all
from glove_code.src.utils import is_number
import logging
from math import log
from numpy import array, newaxis, random, empty, zeros, ones, float32 as REAL, dot, sqrt, std, mean, uint32, copy
from numpy.linalg import norm
import threading
from timeit import default_timer

logger = logging.getLogger(__name__)

X_MAX = 100.0
ALPHA = 0.75
MAX_ABS_VALUE_EMB_INIT = 0.001

vocab_lock = threading.Lock()


try:
    from glove_code.src.glove_inner import train_glove_epoch
except ImportError:
    logger.warning("COULD NOT IMPORT CYTHON FUNCTION")
    # TODO: update according to the code in glove_inner.pyx, if you intend to use this.
    # XXX: this is not needed unless the cython version cannot be imported. If the python function is used instead of
    # the cython one, then training would be extremely(!) slow.
    def train_glove_epoch(model, batch_data, compute_loss):
        batch_loss = 0.0
        pair_tally = 0
        for data in batch_data:
            word1_index, word2_index, occ_count = data
            # Compute difference between log co-occ count and model.
            if model.with_bias:
                diff = dot(model.wv.syn0[word1_index], model.syn1[word2_index]) + model.b0[word1_index] + model.b1[word2_index] - log(occ_count)
            else:
                diff = dot(model.wv.syn0[word1_index], model.syn1[word2_index]) - log(occ_count)
            fdiff = (1.0 if occ_count > X_MAX else pow(occ_count / X_MAX, ALPHA)) * diff

            if not is_number(diff) or not is_number(fdiff):
                logger.warning("NaN or inf encountered in {}, {}".format(diff, fdiff))
                continue

            if compute_loss:
                # Accumulate loss.
                batch_loss += 0.5 * fdiff * diff

            # AdaGrad updates.
            fdiff *= model.lr  # for ease in calculating gradient
            grad0 = fdiff * model.syn1[word2_index]
            grad1 = fdiff * model.wv.syn0[word1_index]

            # Update embeddings.
            model.wv.syn0[word1_index] -= 1.0 / sqrt(model.gradsq_syn0[word1_index]) * grad0
            model.syn1[word2_index] -= 1.0 / sqrt(model.gradsq_syn1[word2_index]) * grad1
            model.gradsq_syn0[word1_index] += norm(grad0)**2
            model.gradsq_syn1[word2_index] += norm(grad1)**2

            if model.with_bias:
                # Update biases.
                model.b0[word1_index] -= 1.0 / sqrt(model.gradsq_b0[word1_index]) * fdiff
                model.b1[word2_index] -= 1.0 / sqrt(model.gradsq_b1[word2_index]) * fdiff
                model.gradsq_b0[word1_index] += fdiff * fdiff
                model.gradsq_b1[word2_index] += fdiff * fdiff

            pair_tally += 1

        return pair_tally, batch_loss


class Glove(SaveLoad):
    def __init__(self, use_glove_format, coocc_file, vocab_file, restrict_vocab, num_workers=5, chunksize=100, epochs=5,
                 optimizer=None, lr=0.05, vector_size=100, vector_dtype=REAL, poincare=0, euclid=1, dist_func=None,
                 cosh_dist_pow=0, num_embs=0, nn_config=None, coocc_func="log", use_scaling=False, seed=1,
                 compute_loss=True, with_bias=False, use_log_probs=False, ckpt_word_list=None, init_near_border=False,
                 init_pretrained_config=None, callbacks=None):

        self.use_glove_format = use_glove_format
        self.coocc_file = coocc_file
        self.num_workers = num_workers
        self.chunksize = chunksize
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = lr
        self.vector_size = vector_size
        self.vector_dtype = vector_dtype
        self.compute_loss = compute_loss
        self.with_bias = with_bias
        self.use_log_probs = use_log_probs

        self.total_train_time = 0
        self.epoch_training_loss = 0.0
        self.trained_pair_count = 0
        self.num_projections = 0
        self.init_near_border = init_near_border
        self.init_pretrained_config = init_pretrained_config

        self.callbacks = callbacks
        self.finished_training = False

        self.poincare = poincare
        self.euclid = euclid
        self.dist_func = dist_func
        self.cosh_dist_pow = cosh_dist_pow
        self.num_embs = num_embs
        self.nn_config = nn_config
        self.coocc_func = coocc_func
        self.use_scaling = use_scaling
        if self.use_scaling:
            self.scaling_factor = 1.0 + (random.rand() * 2 * 0.01 - 0.01)
            print("Initial scaling factor is {}".format(self.scaling_factor))
        self.emb_type = "vanilla"
        if self.poincare == 1:
            self.emb_type = "poincare"
        elif self.euclid == 1:
            self.emb_type = "euclid"

        self.max_word_index = restrict_vocab
        self.vocab_size = 0

        if self.emb_type == "poincare":
            if num_embs > 0:
                self.wv = MixPoincareWordEmbeddingsKeyedVectors(vector_size, num_embs=num_embs, vector_dtype=vector_dtype,
                                                                init_near_border=init_near_border,
                                                                init_pretrained_config=init_pretrained_config)
            else:
                self.wv = PoincareWordEmbeddingsKeyedVectors(vector_size, vector_dtype=vector_dtype,
                                                             init_near_border=init_near_border,
                                                             init_pretrained_config=init_pretrained_config)
        else:
            self.wv = VanillaWordEmbeddingsKeyedVectors(vector_size, vector_dtype=vector_dtype,
                                                        init_pretrained_config=init_pretrained_config)
        self.load_vocab(vocab_file, restrict_vocab)

        self.num_pairs = 0
        self.inspect_training_corpus()
        self.trainables = GloveTrainables(self.vocab_size, self.vector_size, self.vector_dtype, seed, self.with_bias,
                                          self.nn_config)
        self.wv.trainables = self.trainables

        if ckpt_word_list:
            self.word_checkpoints = WordEmbeddingCheckpoints(ckpt_word_list, self)

        # Initialize embeddings.
        self.trainables.init_embeddings(self.wv)

        # Train embeddings.
        self.train()

    def _log_epoch_end(self, cur_epoch, elapsed):
        _, rw_spearman_corr, _ = self.wv.evaluate_word_pairs('../msc_tifreaa/gensim/test/test_data/rare_word.txt')
        _, wordsim_spearman, _ = self.wv.evaluate_word_pairs('../msc_tifreaa/gensim/test/test_data/wordsim353.tsv')
        _, simlex_spearman, _ = self.wv.evaluate_word_pairs('../msc_tifreaa/gensim/test/test_data/simlex999.txt')
        _, mturk_spearman, _ = self.wv.evaluate_word_pairs('../msc_tifreaa/gensim/test/test_data/mturk771.tsv')
        _, simverb_spearman, _ = self.wv.evaluate_word_pairs('../msc_tifreaa/gensim/test/test_data/simverb3500.tsv')

        # Compute embedding norms for top 10 most frequent words and least frequent words (words outside top 1000)
        target_norms = array([norm(self.wv.word_vec(w)) for w in self.wv.index2entity[:self.vocab_size]])
        context_norms = array([norm(self.trainables.syn1neg[idx]) for idx in range(self.vocab_size)])
        top10_avg_norm_target, top10_avg_norm_context = mean(target_norms[:10]), mean(context_norms[:10])
        not_top1000_avg_norm_target, not_top1000_avg_norm_context = mean(target_norms[1000:]), mean(context_norms[1000:])

        print_string = "EPOCH - {:d} : training on {:d} pairs took {:.1f}s, {:.0f} pairs/s, epoch loss {:f}\n\t- Epoch {:d} Similarity: rareword {:.4f}, wordsim {:.4f}, simlex {:.4f}, mturk {:.4f}, simverb {:.4f}, top10_avg_norm {:.4f} / {:.4f}, last_avg_norm {:.4f} / {:.4f}, norm_stddev {:.4f}/{:.4f}".format(
            cur_epoch + 1, self.trained_pair_count, elapsed, self.trained_pair_count / elapsed,
            float(self.epoch_training_loss), cur_epoch + 1,
            rw_spearman_corr[0], wordsim_spearman[0], simlex_spearman[0], mturk_spearman[0], simverb_spearman[0],
            top10_avg_norm_target, top10_avg_norm_context,
            not_top1000_avg_norm_target, not_top1000_avg_norm_context,
            std(target_norms), std(context_norms))
        if not self.with_bias:
            print_string += ", mean bias {:.4f}, variance bias {:.4f}".format(
                self.trainables.mean_bias, self.trainables.var_bias
            )
        if self.use_scaling:
            print_string += ", scaling_factor {:.4f}".format(self.scaling_factor)
        print(print_string)
        logger.info(print_string)

        if (cur_epoch + 1) % 5 == 0:
            self.prepare_emb_for_eval()
            # XXX: we currently compute analogy using 3COSADD and cosine distance because it is a lot faster than using
            # Moebius parallel transport, and we don't want to slow down training.
            # Note that this only happens for logging the progress during training. For evaluating an already trained
            # model, any of the supported analogy functions can be chosen.
            old_value = getattr(self.wv, "use_poincare_distance") if hasattr(self.wv, "use_poincare_distance") else False
            self.wv.use_poincare_distance = False
            google_analogy_eval = self.wv.accuracy(
                'gensim/test/test_data/questions-words.txt',
                restrict_vocab=400000,
                most_similar=VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy,
                verbose=False)
            msr_analogy_eval = self.wv.accuracy(
                '../msc_tifreaa/gensim/test/test_data/msr_word_relationship.processed',
                restrict_vocab=400000,
                most_similar=VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy,
                verbose=False)
            print_string = "\t- Epoch {:d} Analogy: Google {:.4f}, MSR {:.4f}".format(
                cur_epoch + 1, google_analogy_eval[-1]['correct'][0] / 19544,
                msr_analogy_eval[-1]['correct'][0] / 8000)
            print(print_string)
            logger.info(print_string)
            self.wv.use_poincare_distance = old_value

            # Reset vector_norms so that they are computed again next time when we want to evaluate analogy.
            self.wv.vectors_norm = None

    def _log_train_end(self, total_elapsed):
        logger.info(
            "training on %i pairs took %.1fs, %.0f pairs/s",
            self.trained_pair_count, total_elapsed, self.trained_pair_count / total_elapsed
        )

    def _train_epoch(self, cur_epoch):
        """Train one epoch."""
        start = default_timer() - 0.00001
        self.epoch_training_loss, self.trained_pair_count = train_glove_epoch(self)
        elapsed = default_timer() - start

        # We need a lock here, because the self.wv.vocab will be changed during the similarity evaluation, which means
        # that other threads that might be using vocab (e.g. ckpt_worker) may run into race conditions.
        vocab_lock.acquire()
        self._log_epoch_end(cur_epoch, elapsed)
        vocab_lock.release()

    def train(self):
        start = default_timer() - 0.00001

        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Start worker that will save model checkpoints.
        if hasattr(self, "word_checkpoints"):
            print("Starting model checkpoint thread")
            ckpt_worker = threading.Thread(target=self.run_model_ckpt_job)
            ckpt_worker.daemon = True
            ckpt_worker.start()

        for cur_epoch in range(self.epochs):
            self._train_epoch(cur_epoch)

        # Log overall time
        total_elapsed = default_timer() - start

        self.finished_training = True
        if hasattr(self, "word_checkpoints"):
            ckpt_worker.join()
            print("Model checkpoint thread finished working")

        self._log_train_end(total_elapsed)

        for callback in self.callbacks:
            callback.on_train_end(self)

    def load_vocab(self, vocab_file, restrict_vocab):
        # Read vocab.
        with open(vocab_file, "r") as f:
            self.vocab_size = 0
            self.wv.index2freq = []
            all_lines = f.readlines()[:restrict_vocab] if restrict_vocab > 0 else f.readlines()
            for index, line in enumerate(all_lines):
                if self.use_glove_format:
                    word, count = line.strip().split(" ")  # vocab is indexed from 0; for co-occ we use 1-based indexing
                    index = index
                else:
                    index, word, count = line.strip().split("\t")
                    index = int(index) - 1  # indexing starts at 1 in the file; for co-occ we use 0-based indexing
                self.wv.index2word.append(word)
                self.wv.vocab[word] = Vocab(index=index, count=int(count))
                self.wv.index2freq.append(count)
                self.vocab_size += 1

        self.wv.index2freq = array(self.wv.index2freq).astype(uint32)
        self.wv.vector_size = self.vector_size
        self.wv.vector_dtype = self.vector_dtype

        # Unused members from VanillaWordEmbeddingsKeyedVectors.
        self.wv.vectors_norm = None

        print("Loaded vocabulary with {} words".format(self.vocab_size))

    def inspect_training_corpus(self):
        self.num_pairs = read_all(self.use_glove_format, self.coocc_file)
        print("Finished first traversal of corpus. Detected a total of {} pairs".format(self.num_pairs))

    def run_model_ckpt_job(self):
        ckpt_delay, next_ckpt = 0.5, 1.0
        start = default_timer() - 0.00001
        while not self.finished_training:
            elapsed = default_timer() - start
            if elapsed >= next_ckpt:
                vocab_lock.acquire()
                self.word_checkpoints.add_checkpoints()
                vocab_lock.release()
                next_ckpt = elapsed + ckpt_delay

    def prepare_emb_for_eval(self):
        if self.emb_type == "poincare":
            pass
            # TODO: need to change keyedvectors so that they use agg_vectors for logging the training of Poincare embeddings
            # Note that this method is only invoked when logging the progress during training, and not when running the
            # evaluation of an already trained model.
            # self.wv.agg_vectors = self.wv.moebius_mul_mat(agg, 0.5)
        else:
            self.wv.vectors_norm = self.wv.vectors + self.trainables.syn1
            self.wv.vectors_norm = (self.wv.vectors_norm / sqrt((self.wv.vectors_norm ** 2).sum(-1))[..., newaxis]).astype(self.vector_dtype)

    def cleanup(self):
        # Remove references to auxiliary variables that are memory heavy.
        self.trainables.gradsq_syn0, self.trainables.gradsq_syn1 = None, None
        self.trainables.gradsq_b0, self.trainables.gradsq_b1 = None, None

    def get_attr(self, attr_name, default=0):
        return getattr(self, attr_name) if hasattr(self, attr_name) else default


class GloveTrainables(SaveLoad):
    def __init__(self, vocab_size, vector_size, vector_dtype, seed, with_bias, nn_config):
        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.vector_dtype = vector_dtype
        self.with_bias = with_bias
        self.seed = seed

        # Word embeddings and biases.
        self.syn1 = empty((vocab_size, vector_size), dtype=vector_dtype)
        if with_bias:
            self.b0 = empty((vocab_size,), dtype=vector_dtype)
            self.b1 = empty((vocab_size,), dtype=vector_dtype)
        else:
            # Even if we don't have one bias per word embedding, use a global mean and variance bias that "normalizes"
            # the model distribution and the log co-occ count distribution and brings them to the same domain.
            self.mean_bias = 0.0
            self.var_bias = 0.0

        # Arrays for accumulating the sum of the squared gradients during training, for AdaGrad.
        # Initialize with 1.0, so that the initial value of eta is equal to initial learning rate.
        self.gradsq_syn0 = ones((vocab_size, vector_size), dtype=vector_dtype)
        self.gradsq_syn1 = ones((vocab_size, vector_size), dtype=vector_dtype)
        if with_bias:
            self.gradsq_b0 = ones((vocab_size,), dtype=vector_dtype)
            self.gradsq_b1 = ones((vocab_size,), dtype=vector_dtype)
        else:
            self.gradsq_mean_bias = 1.0
            self.gradsq_var_bias = 1.0

        # Arrays for accumulating momentum for AMSgrad.
        self.mom_syn0 = zeros((vocab_size, vector_size), dtype=vector_dtype)
        self.mom_syn1 = zeros((vocab_size, vector_size), dtype=vector_dtype)
        if with_bias:
            self.mom_b0 = zeros((vocab_size,), dtype=vector_dtype)
            self.mom_b1 = zeros((vocab_size,), dtype=vector_dtype)
        else:
            self.mom_mean_bias = 0.0
            self.mom_var_bias = 0.0

        # Arrays for keeping track of coefficients for AMSgrad.
        self.betas_syn0 = ones((vocab_size, vector_size), dtype=vector_dtype)
        self.betas_syn1 = ones((vocab_size, vector_size), dtype=vector_dtype)
        if with_bias:
            self.betas_b0 = ones((vocab_size,), dtype=vector_dtype)
            self.betas_b1 = ones((vocab_size,), dtype=vector_dtype)
        else:
            self.betas_mean_bias = 1.0
            self.betas_var_bias = 1.0

        # For compatibility with the evaluation code for SGNS, we also add syn1neg.
        self.syn1neg = self.syn1

        self.nn_weights = None
        if nn_config:
            self.nn_weights = empty((3, nn_config.num_nodes), dtype=vector_dtype)
            self.nn_output_bias = empty((1, 1), dtype=vector_dtype)

    def init_embeddings(self, wv):
        wv.vectors = empty((self.vocab_size, self.vector_size), dtype=self.vector_dtype)
        start = 0

        if wv.init_pretrained_config:
            # Initialize from pretrained model.
            for i in range(min(wv.init_pretrained_config.vocab_size, self.vocab_size)):
                wv.vectors[i] = copy(wv.init_pretrained_config.init_vectors[i])
                self.syn1[i] = copy(wv.init_pretrained_config.init_syn1[i])
                self.b0[i] = copy(wv.init_pretrained_config.init_b0[i])
                self.b1[i] = copy(wv.init_pretrained_config.init_b1[i])

            # Set start to be the vocab size of the pretrained model. This will add new rows to the data structures,
            # for the words in the vocabulary that do not appear in the pretrained model.
            start = wv.init_pretrained_config.vocab_size

        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in range(start, self.vocab_size):
            if isinstance(wv, PoincareWordEmbeddingsKeyedVectors) and wv.init_near_border:
                wv.vectors[i] = self.seeded_vector(
                    wv.index2word[i] + str(self.seed),
                    wv.vector_size,
                    max_abs_value=0.001).astype(self.vector_dtype)
                vector_norm = random.uniform(low=0.99, high=0.999)
                wv.vectors[i] = wv.vectors[i] / norm(wv.vectors[i]) * vector_norm

                self.syn1[i] = self.seeded_vector(
                    wv.index2word[i] + str(self.seed) * 42,
                    self.vector_size,
                    max_abs_value=0.001).astype(self.vector_dtype)
                vector_norm = random.uniform(low=0.99, high=0.999)
                self.syn1[i] = self.syn1[i] / norm(self.syn1[i]) * vector_norm
            else:
                wv.vectors[i] = self.seeded_vector(
                    wv.index2word[i] + str(self.seed),
                    self.vector_size,
                    max_abs_value=MAX_ABS_VALUE_EMB_INIT).astype(self.vector_dtype)
                self.syn1[i] = self.seeded_vector(
                    wv.index2word[i] + str(self.seed) * 42,
                    self.vector_size,
                    max_abs_value=MAX_ABS_VALUE_EMB_INIT).astype(self.vector_dtype)

            if self.with_bias:
                # Initialize biases.
                self.b0[i] = random.random() * 2 * MAX_ABS_VALUE_EMB_INIT - MAX_ABS_VALUE_EMB_INIT
                self.b1[i] = random.random() * 2 * MAX_ABS_VALUE_EMB_INIT - MAX_ABS_VALUE_EMB_INIT

        if not self.with_bias:
            self.mean_bias = random.random() * 2 * MAX_ABS_VALUE_EMB_INIT - MAX_ABS_VALUE_EMB_INIT
            self.var_bias = random.random() * 2 * MAX_ABS_VALUE_EMB_INIT - MAX_ABS_VALUE_EMB_INIT

        if self.nn_weights is not None:
            # Init biases with both negative and positive small values.
            self.nn_weights[0] = self.seeded_vector(
                "nn_biases" + str(self.seed),
                len(self.nn_weights[0]), max_abs_value=0.00001).astype(self.vector_dtype)
            # Init weights only with positive values.
            for i in range(1, 3):
                self.nn_weights[i] = abs(self.seeded_vector(
                        "nn_weights" + str(i) + str(self.seed),
                        len(self.nn_weights[0]), max_abs_value=0.00001).astype(self.vector_dtype))
            # Init output bias.
            self.nn_output_bias = self.seeded_vector("nn_output_bias" + str(self.seed), 1, max_abs_value=0.0001).astype(
                        self.vector_dtype)

    @staticmethod
    def seeded_vector(seed_string, vector_size, max_abs_value=None):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(hash(seed_string) & 0xffffffff)
        if max_abs_value is None:
            return (once.rand(vector_size) - 0.5) / vector_size
        else:
            return once.rand(vector_size) * 2 * max_abs_value - max_abs_value


class InitializationConfig:
    """
    Class that stores information about the way we want to initialize the embeddings from pretrained embeddings.

    """

    def __init__(self, pretrained_model_filename):
        model = Glove.load(pretrained_model_filename)
        self.vocab_size = len(model.wv.vocab)

        self.init_vectors = model.wv.vectors
        self.init_syn1= model.trainables.syn1
        self.init_b0 = model.trainables.b0
        self.init_b1 = model.trainables.b1


class NNConfig(SaveLoad):
    def __init__(self, config_str):
        config_info = config_str.split("-")
        self.num_nodes = int(config_info[0])
        self.nonlinearity = config_info[1]
