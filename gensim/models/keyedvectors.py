#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Shiva Manne <manneshiva@gmail.com>
# Copyright (C) 2018 RaRe Technologies s.r.o.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Word vector storage and similarity look-ups.
Common code independent of the way the vectors are trained(Word2Vec, FastText, WordRank, VarEmbed etc)

The word vectors are considered read-only in this class.

Initialize the vectors by training e.g. Word2Vec::

>>> model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
>>> word_vectors = model.wv

Persist the word vectors to disk with::

>>> word_vectors.save(fname)
>>> word_vectors = KeyedVectors.load(fname)

The vectors can also be instantiated from an existing file on disk
in the original Google's word2vec C format as a KeyedVectors instance::

  >>> from gensim.models import KeyedVectors
  >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
  >>> word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.bin', binary=True)  # C binary format

You can perform various syntactic/semantic NLP word tasks with the vectors. Some of them
are already built-in::

  >>> word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.50882536), ...]

  >>> word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
  [('queen', 0.71382287), ...]

  >>> word_vectors.doesnt_match("breakfast cereal dinner lunch".split())
  'cereal'

  >>> word_vectors.similarity('woman', 'man')
  0.73723527

Correlation with human opinion on word similarity::

  >>> word_vectors.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
  0.51, 0.62, 0.13

And on analogies::

  >>> word_vectors.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))

and so on.

"""
from __future__ import division  # py3 "true division"

import logging

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty  # noqa:F401

# If pyemd C extension is available, import it.
# If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
try:
    from pyemd import emd
    PYEMD_EXT = True
except ImportError:
    PYEMD_EXT = False

from numpy import dot, zeros, float32 as REAL, float64 as DOUBLE, empty, memmap as np_memmap, \
    double, array, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, average, prod, argmax, divide as np_divide, tanh, arctanh, arccosh, cos, log
from numpy.linalg import norm
import numpy as np
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.corpora.dictionary import Dictionary
from six import string_types, integer_types
from six.moves import xrange, zip
from scipy import sparse, stats
from gensim.utils import deprecated
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash

logger = logging.getLogger(__name__)

EPS = 1e-10

class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class BaseKeyedVectors(utils.SaveLoad):

    def __init__(self, vector_size, vector_dtype=REAL):
        self.vectors = []
        self.vocab = {}
        self.vector_size = vector_size
        self.vector_dtype = vector_dtype
        self.index2entity = []

    def save(self, fname_or_handle, **kwargs):
        super(BaseKeyedVectors, self).save(fname_or_handle, **kwargs)

    @classmethod
    def load(cls, fname_or_handle, **kwargs):
        return super(BaseKeyedVectors, cls).load(fname_or_handle, **kwargs)

    def similarity(self, entity1, entity2):
        """Compute cosine similarity between entities, specified by string tag.
        """
        raise NotImplementedError()

    def most_similar(self, **kwargs):
        """Find the top-N most similar entities.
        Possibly have `positive` and `negative` list of entities in `**kwargs`.
        """
        return NotImplementedError()

    def distance(self, entity1, entity2):
        """Compute distance between vectors of two input entities, specified by string tag.
        """
        raise NotImplementedError()

    def distances(self, entity1, other_entities=()):
        """Compute distances from given entity (string tag) to all entities in `other_entity`.
        If `other_entities` is empty, return distance between `entity1` and all entities in vocab.
        """
        raise NotImplementedError()

    def embedding_norm(self, word):
        """Compute the norm of the target embedding for a given word
        """
        raise NotImplementedError()

    def get_vector(self, entity):
        """Accept a single entity as input, specified by string tag.
        Returns the entity's representations in vector space, as a 1D numpy array.
        """
        if entity in self.vocab:
            result = self.vectors[self.vocab[entity].index]
            result.setflags(write=False)
            return result
        else:
            raise KeyError("'%s' not in vocabulary" % entity)

    def __getitem__(self, entities):
        """
        Accept a single entity (string tag) or list of entities as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if isinstance(entities, string_types):
            # allow calls like trained_model['office'], as a shorthand for trained_model[['office']]
            return self.get_vector(entities)

        return vstack([self.get_vector(entity) for entity in entities])

    def __contains__(self, entity):
        return entity in self.vocab

    def most_similar_to_given(self, entity1, entities_list):
        """Return the entity from entities_list most similar to entity1."""
        return entities_list[argmax([self.similarity(entity1, entity) for entity in entities_list])]

    def closer_than(self, entity1, entity2):
        """Returns all entities that are closer to `entity1` than `entity2` is to `entity1`."""
        all_distances = self.distances(entity1)
        e1_index = self.vocab[entity1].index
        e2_index = self.vocab[entity2].index
        closer_node_indices = np.where(all_distances < all_distances[e2_index])[0]
        return [self.index2entity[index] for index in closer_node_indices if index != e1_index]

    def rank(self, entity1, entity2):
        """Rank of the distance of `entity2` from `entity1`, in relation to distances of all entities from `entity1`."""
        return len(self.closer_than(entity1, entity2)) + 1


class WordEmbeddingsKeyedVectors(BaseKeyedVectors):
    """Class containing common methods for operations over word vectors."""

    def __init__(self, vector_size, vector_dtype=REAL, init_pretrained_config=None):
        super(WordEmbeddingsKeyedVectors, self).__init__(vector_size=vector_size, vector_dtype=vector_dtype)
        self.vectors_norm = None
        self.index2word = []
        self.index2freq = []
        self.init_pretrained_config = init_pretrained_config

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self instead")
    def wv(self):
        return self

    @property
    def index2entity(self):
        return self.index2word

    @index2entity.setter
    def index2entity(self, value):
        self.index2word = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors instead")
    def syn0(self):
        return self.vectors

    @syn0.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors instead")
    def syn0(self, value):
        self.vectors = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_norm instead")
    def syn0norm(self):
        return self.vectors_norm

    @syn0norm.setter
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_norm instead")
    def syn0norm(self, value):
        self.vectors_norm = value

    def __contains__(self, word):
        return word in self.vocab

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.*2vec.*2VecKeyedVectors.load` which supports
        operations on trained word vectors like `most_similar`.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_norm'])
        super(WordEmbeddingsKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        If `use_norm` is True, returns the normalized word vector.

        Examples
        --------
        >>> trained_model['office']
        array([ -1.40128313e-02, ...])

        """
        if word in self.vocab:
            if use_norm:
                result = self.vectors_norm[self.vocab[word].index]
            else:
                result = self.vectors[self.vocab[word].index]

            result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def get_vector(self, word):
        return self.word_vec(word)

    def words_closer_than(self, w1, w2):
        """
        Returns all words that are closer to `w1` than `w2` is to `w1`.

        Parameters
        ----------
        w1 : str
            Input word.
        w2 : str
            Input word.

        Returns
        -------
        list (str)
            List of words that are closer to `w1` than `w2` is to `w1`.

        Examples
        --------
        >>> model.words_closer_than('carnivore', 'mammal')
        ['dog', 'canine']

        """
        return super(WordEmbeddingsKeyedVectors, self).closer_than(w1, w2)

    def similar_by_word(self, word, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words.

        Parameters
        ----------
        word : str
            Word
        topn : int
            Number of top-N similar words to return. If topn is False, similar_by_word returns
            the vector of similarity scores.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        Example::

          >>> trained_model.similar_by_word('graph')
          [('user', 0.9999163150787354), ...]

        """
        return self.most_similar(positive=[word], topn=topn, restrict_vocab=restrict_vocab)

    def similar_by_vector(self, vector, topn=10, restrict_vocab=None):
        """
        Find the top-N most similar words by vector.

        Parameters
        ----------
        vector : numpy.array
            vector from which similarities are to be computed.
            expected shape (dim,)
        topn : int
            Number of top-N similar words to return. If topn is False, similar_by_vector returns
            the vector of similarity scores.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        return self.most_similar(positive=[vector], topn=topn, restrict_vocab=restrict_vocab)

    def similarity_matrix(self, dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100, dtype=REAL):
        """Constructs a term similarity matrix for computing Soft Cosine Measure.

        Constructs a a sparse term similarity matrix in the :class:`scipy.sparse.csc_matrix` format for computing
        Soft Cosine Measure between documents.

        Parameters
        ----------
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`
            A dictionary that specifies a mapping between words and the indices of rows and columns
            of the resulting term similarity matrix.
        tfidf : :class:`gensim.models.tfidfmodel.TfidfModel`, optional
            A model that specifies the relative importance of the terms in the dictionary. The rows
            of the term similarity matrix will be build in an increasing order of importance of terms,
            or in the order of term identifiers if None.
        threshold : float, optional
            Only pairs of words whose embeddings are more similar than `threshold` are considered
            when building the sparse term similarity matrix.
        exponent : float, optional
            The exponent applied to the similarity between two word embeddings when building the term similarity matrix.
        nonzero_limit : int, optional
            The maximum number of non-zero elements outside the diagonal in a single row or column
            of the term similarity matrix. Setting `nonzero_limit` to a constant ensures that the
            time complexity of computing the Soft Cosine Measure will be linear in the document
            length rather than quadratic.
        dtype : numpy.dtype, optional
            Data-type of the term similarity matrix.

        Returns
        -------
        :class:`scipy.sparse.csc_matrix`
            Term similarity matrix.

        See Also
        --------
        :func:`gensim.matutils.softcossim`
            The Soft Cosine Measure.
        :class:`gensim.similarities.docsim.SoftCosineSimilarity`
            A class for performing corpus-based similarity queries with Soft Cosine Measure.


        Notes
        -----
        The constructed matrix corresponds to the matrix Mrel defined in section 2.1 of
        `Delphine Charlet and Geraldine Damnati, "SimBow at SemEval-2017 Task 3: Soft-Cosine Semantic Similarity
        between Questions for Community Question Answering", 2017
        <http://www.aclweb.org/anthology/S/S17/S17-2051.pdf>`__.

        """
        logger.info("constructing a term similarity matrix")
        matrix_order = len(dictionary)
        matrix_nonzero = [1] * matrix_order
        matrix = sparse.identity(matrix_order, dtype=dtype, format="dok")
        num_skipped = 0
        # Decide the order of rows.
        if tfidf is None:
            word_indices = range(matrix_order)
        else:
            assert max(tfidf.idfs) < matrix_order
            word_indices = [
                index for index, _ in sorted(tfidf.idfs.items(), key=lambda x: x[1], reverse=True)
            ]

        # Traverse rows.
        for row_number, w1_index in enumerate(word_indices):
            if row_number % 1000 == 0:
                logger.info(
                    "PROGRESS: at %.02f%% rows (%d / %d, %d skipped, %.06f%% density)",
                    100.0 * (row_number + 1) / matrix_order, row_number + 1, matrix_order,
                    num_skipped, 100.0 * matrix.getnnz() / matrix_order**2)
            w1 = dictionary[w1_index]
            if w1 not in self.vocab:
                num_skipped += 1
                continue  # A word from the dictionary is not present in the word2vec model.
            # Traverse upper triangle columns.
            if matrix_order <= nonzero_limit + 1:  # Traverse all columns.
                columns = (
                    (w2_index, self.similarity(w1, dictionary[w2_index]))
                    for w2_index in range(w1_index + 1, matrix_order)
                    if w1_index != w2_index and dictionary[w2_index] in self.vocab)
            else:  # Traverse only columns corresponding to the embeddings closest to w1.
                num_nonzero = matrix_nonzero[w1_index] - 1
                columns = (
                    (dictionary.token2id[w2], similarity)
                    for _, (w2, similarity)
                    in zip(
                        range(nonzero_limit - num_nonzero),
                        self.most_similar(positive=[w1], topn=nonzero_limit - num_nonzero)
                    )
                    if w2 in dictionary.token2id
                )
                columns = sorted(columns, key=lambda x: x[0])

            for w2_index, similarity in columns:
                # Ensure that we don't exceed `nonzero_limit` by mirroring the upper triangle.
                if similarity > threshold and matrix_nonzero[w2_index] <= nonzero_limit:
                    element = similarity**exponent
                    matrix[w1_index, w2_index] = element
                    matrix_nonzero[w1_index] += 1
                    matrix[w2_index, w1_index] = element
                    matrix_nonzero[w2_index] += 1
        logger.info(
            "constructed a term similarity matrix with %0.6f %% nonzero elements",
            100.0 * matrix.getnnz() / matrix_order**2
        )
        return matrix.tocsc()

    def wmdistance(self, document1, document2):
        """
        Compute the Word Mover's Distance between two documents. When using this
        code, please consider citing the following papers:

        .. Ofir Pele and Michael Werman, "A linear time histogram metric for improved SIFT matching".
        .. Ofir Pele and Michael Werman, "Fast and robust earth mover's distances".
        .. Matt Kusner et al. "From Word Embeddings To Document Distances".

        Note that if one of the documents have no words that exist in the
        Word2Vec vocab, `float('inf')` (i.e. infinity) will be returned.

        This method only works if `pyemd` is installed (can be installed via pip, but requires a C compiler).

        Example:
            >>> # Train word2vec model.
            >>> model = Word2Vec(sentences)

            >>> # Some sentences to test.
            >>> sentence_obama = 'Obama speaks to the media in Illinois'.lower().split()
            >>> sentence_president = 'The president greets the press in Chicago'.lower().split()

            >>> # Remove their stopwords.
            >>> from nltk.corpus import stopwords
            >>> stopwords = nltk.corpus.stopwords.words('english')
            >>> sentence_obama = [w for w in sentence_obama if w not in stopwords]
            >>> sentence_president = [w for w in sentence_president if w not in stopwords]

            >>> # Compute WMD.
            >>> distance = model.wmdistance(sentence_obama, sentence_president)
        """

        if not PYEMD_EXT:
            raise ImportError("Please install pyemd Python package to compute WMD.")

        # Remove out-of-vocabulary words.
        len_pre_oov1 = len(document1)
        len_pre_oov2 = len(document2)
        document1 = [token for token in document1 if token in self]
        document2 = [token for token in document2 if token in self]
        diff1 = len_pre_oov1 - len(document1)
        diff2 = len_pre_oov2 - len(document2)
        if diff1 > 0 or diff2 > 0:
            logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

        if len(document1) == 0 or len(document2) == 0:
            logger.info(
                "At least one of the documents had no words that werein the vocabulary. "
                "Aborting (returning inf)."
            )
            return float('inf')

        dictionary = Dictionary(documents=[document1, document2])
        vocab_len = len(dictionary)

        if vocab_len == 1:
            # Both documents are composed by a single unique token
            return 0.0

        # Sets for faster look-up.
        docset1 = set(document1)
        docset2 = set(document2)

        # Compute distance matrix.
        distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
        for i, t1 in dictionary.items():
            for j, t2 in dictionary.items():
                if t1 not in docset1 or t2 not in docset2:
                    continue
                # Compute Euclidean distance between word vectors.
                distance_matrix[i, j] = sqrt(np_sum((self[t1] - self[t2])**2))

        if np_sum(distance_matrix) == 0.0:
            # `emd` gets stuck if the distance matrix contains only zeros.
            logger.info('The distance matrix is all zeros. Aborting (returning inf).')
            return float('inf')

        def nbow(document):
            d = zeros(vocab_len, dtype=double)
            nbow = dictionary.doc2bow(document)  # Word frequencies.
            doc_len = len(document)
            for idx, freq in nbow:
                d[idx] = freq / float(doc_len)  # Normalized word frequencies.
            return d

        # Compute nBOW representation of documents.
        d1 = nbow(document1)
        d2 = nbow(document2)

        # Compute WMD.
        return emd(d1, d2, distance_matrix)

    @staticmethod
    def log_accuracy(section):
        # Instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers
        idx = argmax(section["correct"])
        correct, incorrect = section["correct"][idx], section["incorrect"][idx]
        if correct + incorrect > 0:
            print("{}: {:.1f}% ({}/{}) for t={:.2f}".format(
                section['section'], 100.0 * correct / (correct + incorrect), correct, correct + incorrect, idx*0.1
            ))

    def eval_accuracy_for_batch(self, batch, section, most_similar, restrict_vocab, case_insensitive, debug=False):
        if len(batch) == 0:
            return

        batch_arr = np.array(batch).reshape((-1, 4))

        A = batch_arr[:, 0]
        B = batch_arr[:, 1]
        C = batch_arr[:, 2]
        expected = [self.index2word[i].upper() if case_insensitive else self.index2word[i] for i in batch_arr[:, 3]]

        # find the most likely prediction, ignoring OOV words and input words
        results = most_similar(self, positive=[B, C], negative=A, restrict_vocab=restrict_vocab, debug=debug)

        for result in results:
            # correct, incorrect = [], []
            correct, incorrect = 0, 0
            predicted = [word.upper() for word in result[0]] if case_insensitive else result[0]
            for i, info in enumerate(zip(expected, predicted, A, B, C)):
                exp, pred, a, b, c = info
                # Instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers
                if pred == exp:
                    correct += 1
                else:
                    incorrect += 1
            section["correct"].append(correct)
            section["incorrect"].append(incorrect)

        batch.clear()

    def accuracy(self, questions, restrict_vocab=30000, most_similar=None, case_insensitive=True, debug=False, verbose=True):
        """
        Compute accuracy of the model. `questions` is a filename where lines are
        4-tuples of words, split into sections by ": SECTION NAME" lines.
        See questions-words.txt in
        https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
        for an example.

        The accuracy is reported (=printed to log and returned as a list) for each
        section separately, plus there's one aggregate summary at the end.

        Use `restrict_vocab` to ignore all questions containing a word not in the first `restrict_vocab`
        words (default 30,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        In case `case_insensitive` is True, the first `restrict_vocab` words are taken first, and then
        case normalization is performed.

        Use `case_insensitive` to convert all words in questions and vocab to their uppercase form before
        evaluating the accuracy (default True). Useful in case of case-mismatch between training tokens
        and question words. In case of multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        This method corresponds to the `compute-accuracy` script of the original C word2vec.

        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        if not most_similar:
            most_similar = VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy

        sections, section = [], None
        batch = []

        original_vocab = self.vocab
        self.vocab = ok_vocab
        for line_no, line in enumerate(utils.smart_open(questions)):
            line = utils.to_unicode(line)
            if line.startswith(': '):
                # Evaluate previous section.
                self.eval_accuracy_for_batch(batch=batch, section=section, most_similar=most_similar,
                                             restrict_vocab=restrict_vocab, case_insensitive=case_insensitive,
                                             debug=debug)

                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    if verbose:
                        self.log_accuracy(section)
                    # Only evaluate one section when running in debug mode.
                    if debug:
                        return sections
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.upper() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except ValueError:
                    logger.info("skipping invalid line #%i in %s", line_no, questions)
                    continue
                if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                    logger.debug("skipping line #%i with OOV words: %s", line_no, line.strip())
                    continue
                batch.append([self.vocab[w].index for w in [a, b, c, expected]])

        # Evaluate last section.
        self.eval_accuracy_for_batch(batch=batch, section=section, most_similar=most_similar,
                                     restrict_vocab=restrict_vocab, case_insensitive=case_insensitive, debug=debug)

        self.vocab = original_vocab
        if section:
            # store the last section, too
            sections.append(section)
            if verbose:
                self.log_accuracy(section)

        # Instead of adding the correct/incorrect words to a list, I am just counting the number of correct/incorrect answers
        if len(sections) > 1:
            sem = {
                'section': 'semantic',
                'correct': np.array([np.array(section["correct"])
                                     for section in filter(lambda s: not s['section'].startswith('gram'), sections)]).sum(axis=0),
                'incorrect': np.array([np.array(section["incorrect"])
                                       for section in filter(lambda s: not s['section'].startswith('gram'), sections)]).sum(axis=0),
            }
            syn = {
                'section': 'syntactic',
                'correct': np.array([np.array(section["correct"])
                                     for section in filter(lambda s: s['section'].startswith('gram'), sections)]).sum(axis=0),
                'incorrect': np.array([np.array(section["incorrect"])
                                       for section in filter(lambda s: s['section'].startswith('gram'), sections)]).sum(axis=0),
            }
        total = {
            'section': 'total',
            'correct': np.array([np.array(section["correct"]) for section in sections]).sum(axis=0),
            'incorrect': np.array([np.array(section["incorrect"]) for section in sections]).sum(axis=0),
        }

        if len(sections) > 1:
            if verbose:
                self.log_accuracy(sem)
                self.log_accuracy(syn)
            idx = argmax(sem["correct"])
            sem["correct"] = [sem["correct"][idx]]
            sem["incorrect"] = [sem["incorrect"][idx]]
            sem["t_argmax"] = [idx * 0.1]
            sections.append(sem)
            idx = argmax(syn["correct"])
            syn["correct"] = [syn["correct"][idx]]
            syn["incorrect"] = [syn["incorrect"][idx]]
            syn["t_argmax"] = [idx * 0.1]
            sections.append(syn)

        if verbose:
            self.log_accuracy(total)

        idx = argmax(total["correct"])
        total["correct"] = [total["correct"][idx]]
        total["incorrect"] = [total["incorrect"][idx]]
        total["t_argmax"] = [idx * 0.1]
        sections.append(total)
        return sections

    @staticmethod
    def log_evaluate_word_pairs(pearson, spearman, oov, pairs):
        logger.debug('Pearson correlation coefficient against %s: %.4f', pairs, pearson[0])
        logger.debug('Spearman rank-order correlation coefficient against %s: %.4f', pairs, spearman[0])
        logger.debug('Pairs with unknown words ratio: %.1f%%', oov)

    def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000,
                            case_insensitive=True, dummy4unknown=False, debug_file=None):
        """
        Compute correlation of the model with human similarity judgments. `pairs` is a filename of a dataset where
        lines are 3-tuples, each consisting of a word pair and a similarity value, separated by `delimiter`.
        An example dataset is included in Gensim (test/test_data/wordsim353.tsv). More datasets can be found at
        http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html or https://www.cl.cam.ac.uk/~fh295/simlex.html.

        The model is evaluated using Pearson correlation coefficient and Spearman rank-order correlation coefficient
        between the similarities from the dataset and the similarities produced by the model itself.
        The results are printed to log and returned as a triple (pearson, spearman, ratio of pairs with unknown words).

        Use `restrict_vocab` to ignore all word pairs containing a word not in the first `restrict_vocab`
        words (default 300,000). This may be meaningful if you've sorted the vocabulary by descending frequency.
        If `case_insensitive` is True, the first `restrict_vocab` words are taken, and then case normalization
        is performed.

        Use `case_insensitive` to convert all words in the pairs and vocab to their uppercase form before
        evaluating the model (default True). Useful when you expect case-mismatch between training tokens
        and words pairs in the dataset. If there are multiple case variants of a single word, the vector for the first
        occurrence (also the most frequent if vocabulary is sorted) is taken.

        Use `dummy4unknown=True` to produce zero-valued similarities for pairs with out-of-vocabulary words.
        Otherwise (default False), these pairs are skipped entirely.
        """
        ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
        ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

        similarity_gold = []
        similarity_model = []
        oov = 0

        original_vocab = self.vocab
        self.vocab = ok_vocab
        if debug_file:
            f = open(debug_file, "w")
            f.write("Word1,Word2,Gold standard(0-10),Model similarity (-hyp_dist^2)\n")

        for line_no, line in enumerate(utils.smart_open(pairs)):
            line = utils.to_unicode(line)
            if line.startswith('#'):
                # May be a comment
                continue
            else:
                try:
                    if case_insensitive:
                        a, b, sim = [word.upper() for word in line.split(delimiter)]
                    else:
                        a, b, sim = [word for word in line.split(delimiter)]
                    sim = float(sim)
                except (ValueError, TypeError):
                    logger.info('Skipping invalid line #%d in %s', line_no, pairs)
                    continue
                if a not in ok_vocab or b not in ok_vocab:
                    oov += 1
                    if dummy4unknown:
                        logger.debug('Zero similarity for line #%d with OOV words: %s', line_no, line.strip())
                        similarity_model.append(0.0)
                        similarity_gold.append(sim)
                        continue
                    else:
                        logger.debug('Skipping line #%d with OOV words: %s', line_no, line.strip())
                        continue
                similarity_gold.append(sim)  # Similarity from the dataset
                model_sim = self.similarity(a, b)
                similarity_model.append(model_sim)  # Similarity from the model
                if debug_file:
                    f.write(a.lower() + "," + b.lower() + "," + str(sim) + "," + str(model_sim) + "\n")

        if debug_file:
            f.close()
        self.vocab = original_vocab
        spearman = stats.spearmanr(similarity_gold, similarity_model)
        pearson = stats.pearsonr(similarity_gold, similarity_model)
        if dummy4unknown:
            oov_ratio = float(oov) / len(similarity_gold) * 100
        else:
            oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

        logger.debug('Pearson correlation coefficient against %s: %f with p-value %f', pairs, pearson[0], pearson[1])
        logger.debug(
            'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
            pairs, spearman[0], spearman[1]
        )
        logger.debug('Pairs with unknown words: %d', oov)
        self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
        return pearson, spearman, oov_ratio


    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            print("init_sims from WordEmbeddings")
            logger.info("precomputing L2-norms of word weight vectors; replace={}".format(replace))
            dtype = REAL
            if hasattr(self, 'vector_dtype'):
                dtype = self.vector_dtype
            if replace:
                for i in xrange(self.vectors.shape[0]):
                    self.vectors[i, :] /= sqrt((self.vectors[i, :] ** 2).sum(-1))
                self.vectors_norm = self.vectors
            else:
                self.vectors_norm = (self.vectors / sqrt((self.vectors ** 2).sum(-1))[..., newaxis]).astype(dtype)


class PoincareWordEmbeddingsKeyedVectors(WordEmbeddingsKeyedVectors):
    """
    Class used for word embeddings on the Poincare ball which use the Poincare geodesic distance for the similarity
    metric (instead of the cosine similarity).
    """

    def __init__(self, vector_size, vector_dtype=REAL, trainables=None, init_near_border=False,
                 init_pretrained_config=False):
        super(PoincareWordEmbeddingsKeyedVectors, self).__init__(vector_size=vector_size, vector_dtype=vector_dtype)

        # If True, use Poincare distance to measure similarity between words. Otherwise, use cosine distance.
        self.use_poincare_distance = True
        self.trainables = trainables
        self.init_near_border = init_near_border
        self.init_pretrained_config = init_pretrained_config

    def batch_most_similar_hyperbolic_analogy(self, positive=None, negative=None, restrict_vocab=None, debug=False):
        """
        Solve an analogy task. The result should be similar to the positive words and unlike the negative word.

        This method computes the similarity (according to the formula defined for the hyperbolic space) between
        the parallel transport of the input vector and the vectors for each word in the model and selects the word
        that is closest to the position of the parallel transported vector.

        Parameters
        ----------
        positive : list of two numpy.array
            List of two 2D numpy arrays. Each of them contains positive instances. The number of rows is equal to the
            number of questions in a batch.
        negative : numpy.array
            2D array that contains on each row the embedding of the negative word from that question. The number of
            rows is equal to the number of questions in a batch.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        batch_size = len(negative)

        # XXX: before calling this method, #accuracy is setting self.vocab to be only the restricted vocab.
        # So here, self.vocab is actually self.vocab[:restricted]

        if not self.use_poincare_distance:
            self.init_sims()

        # Retrieve embeddings.
        pos_emb = [
            self.vectors[positive[0]],
            self.vectors[positive[1]]
        ]
        neg_emb = self.vectors[negative]

        # Compute the parallel transport of the positive vector in the analogy question (i.e. c) using the new formula
        parallel_transp1 = self.moebius_add_mat(
            pos_emb[1],
            self.gyr_mat(pos_emb[1], -neg_emb, self.moebius_add_mat(-neg_emb, pos_emb[0])))  # batch_size x vector_size

        # Compute the parallel transport of the other positive vector (i.e. b) so the alternative formulation of the
        # analogy question.
        parallel_transp2 = self.moebius_add_mat(
            pos_emb[0],
            self.gyr_mat(pos_emb[0], -neg_emb, self.moebius_add_mat(-neg_emb, pos_emb[1])))  # batch_size x vector_size

        # Compute the gyrolinear combination between the two parallel
        # transported points.
        t = 0.3
        aux = self.moebius_add_mat(-parallel_transp1, parallel_transp2)

        results = []
        lin_comb_point = self.moebius_add_mat(parallel_transp1, self.moebius_mul_mat(aux, t))

        # Compute similarity between parallel transported input and all words in the vocabulary.
        if self.use_poincare_distance:
            limited = self.vectors if restrict_vocab is None else self.vectors[:restrict_vocab]  # vocab_size * vector_size
            # NOTE!!! This is not actually the distance, but cosh(distance) (so only the argument of arccosh in the
            # Poincare distance formula). However, cosh(x) is monotonous (for positive x) which means that we will get
            # the same argmax in the end.
            dists = self.cosh_distances_mat(lin_comb_point, limited)  # batch_size * vocab_size
        else:
            # Get normalized vectors, if we use cosine distance.
            limited_norm = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]  # vocab_size * vector_size
            # Normalize parallel transported vector.
            lin_comb_point = lin_comb_point / norm(lin_comb_point, axis=1)[:, None]
            dists = -dot(lin_comb_point, limited_norm.T)  # batch_size * vocab_size

        max_float = np.finfo(dists.dtype).max
        batch_size_range = np.arange(batch_size)
        x = np.concatenate((batch_size_range, batch_size_range, batch_size_range))
        y = np.concatenate((positive[0], positive[1], negative))
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)

        best = []
        if debug:
            for i in batch_size_range:
                top_ids = matutils.argsort(dists[i], topn=10)
                curr_best = [(self.index2word[idx],
                              self.distances(limited[idx], np.array([neg_emb[i], pos_emb[0][i], pos_emb[1][i]]))[0])
                             for idx in top_ids]
                best.append(curr_best)

        best_ids = np.argmin(dists, axis=1)
        result = (
            [self.index2word[i] for i in best_ids],
            dists[batch_size_range, best_ids].astype(np.float32),
            best)
        results.append(result)
        return results

    def batch_most_similar_3distadd_analogy(self, positive=None, negative=None, restrict_vocab=None, debug=False):
        """
        Solve an analogy task. The result should be similar to the positive words and unlike the negative word.

        Implements 3DISTADD. This replaces the cosine similarities in the 3COSADD formula with -dist.

        Parameters
        ----------
        positive : list of two numpy.array
            List of two 2D numpy arrays. Each of them contains positive instances. The number of rows is equal to the
            number of questions in a batch.
        negative : numpy.array
            2D array that contains on each row the embedding of the negative word from that question. The number of
            rows is equal to the number of questions in a batch.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        batch_size = len(negative)

        # XXX: before calling this method, #accuracy is setting self.vocab to be only the restricted vocab.
        # So here, self.vocab is actually self.vocab[:restricted]

        if not self.use_poincare_distance:
            self.init_sims()

        # Retrieve embeddings.
        pos_emb = [
            self.vectors[positive[0]],
            self.vectors[positive[1]]
        ]
        neg_emb = self.vectors[negative]

        results = []
        # Compute similarity between parallel transported input and all words in the vocabulary.
        if self.use_poincare_distance:
            limited = self.vectors if restrict_vocab is None else self.vectors[:restrict_vocab]  # vocab_size * vector_size
            # NOTE!!! This is not actually the distance, but cosh(distance) (so only the argument of arccosh in the
            # Poincare distance formula). However, cosh(x) is monotonous (for positive x) which means that we will get
            # the same argmax in the end.
            if isinstance(self, MixPoincareWordEmbeddingsKeyedVectors):
                dists = (self.mix_distances_mat(pos_emb[0], limited) + self.mix_distances_mat(pos_emb[1], limited) -
                         self.mix_distances_mat(neg_emb, limited))  # batch_size * vocab_size
            else:
                dists = (self.cosh_distances_mat(pos_emb[0], limited) + self.cosh_distances_mat(pos_emb[1], limited) -
                         self.cosh_distances_mat(neg_emb, limited))  # batch_size * vocab_size
        else:
            # Get normalized vectors, if we use cosine distance.
            limited_norm = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]  # vocab_size * vector_size
            # Normalize parallel transported vector.
            pos_emb[0] = pos_emb[0] / norm(pos_emb[0], axis=1)[:, None]
            pos_emb[1] = pos_emb[1] / norm(pos_emb[1], axis=1)[:, None]
            neg_emb = neg_emb / norm(neg_emb, axis=1)[:, None]
            if isinstance(self, MixPoincareWordEmbeddingsKeyedVectors):
                dists = (self.mix_distances_mat(pos_emb[0], limited_norm) + self.mix_distances_mat(pos_emb[1], limited_norm) -
                         self.mix_distances_mat(neg_emb, limited_norm))  # batch_size * vocab_size
            else:
                dists = -(dot(pos_emb[0], limited_norm.T) + dot(pos_emb[1], limited_norm.T) - dot(neg_emb, limited_norm.T))  # batch_size * vocab_size

        max_float = np.finfo(dists.dtype).max
        batch_size_range = np.arange(batch_size)
        x = np.concatenate((batch_size_range, batch_size_range, batch_size_range))
        y = np.concatenate((positive[0], positive[1], negative))
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)

        best = []
        if debug:
            for i in batch_size_range:
                top_ids = matutils.argsort(dists[i], topn=10)
                curr_best = [(self.index2word[idx],
                              self.distances(limited[idx], np.array([neg_emb[i], pos_emb[0][i], pos_emb[1][i]]))[0])
                             for idx in top_ids]
                best.append(curr_best)

        best_ids = np.argmin(dists, axis=1)
        result = (
            [self.index2word[i] for i in best_ids],
            dists[batch_size_range, best_ids].astype(np.float32),
            best)
        results.append(result)
        return results

    def cosh_distances_mat(self, vectors, other_vectors=None):
        """
        Returns the argument of the arccosh function in the Poincare distance formula. Since arccosh(x) is a monotonous
        function for x >= 1, this is enough to create a ranking and select the closest point to another reference point.

        Parameters
        ----------
        vectors: numpy.array
            Vectors from which distances are to be computed.
        other_vectors: numpy.array
            For each vector in `other_vectors` distance from each vector in `vectors` is computed.
            If None or empty, all words in vocab are considered (including the vectors in `vectors`).

        Returns
        -------
        np.array
            Returns a numpy.array that contains the distance between each row in `vectors`
            and each row in `other_vectors`
        """
        if other_vectors is None:
            other_vectors = self.vectors

        dot_ww = (other_vectors * other_vectors).sum(axis=1)  # vocab_size * 1
        beta_w = 1.0 / (1 - dot_ww)  # vocab_size * 1
        dot_vv = (vectors * vectors).sum(axis=1)  # batch_size * 1
        alpha_v = 1.0 / (1 - dot_vv)  # batch_size * 1
        dot_vw = dot(vectors, other_vectors.T)  # batch_size * vocab_size

        cosh_dists = 1 + (-2 * dot_vw + dot_ww + dot_vv[:, None]) * alpha_v[:, None] * beta_w  # batch_size * vocab
        return cosh_dists

    def distances(self, word_or_vector, other_vectors=None):
        """
        Compute the cosh of the Poincare distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : str or numpy.array
            Word or vector from which distances are to be computed.

        other_vectors: numpy.array or None
            For each vector in `other_vectors` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`,
            in the same order as `other_words`.

        Notes
        -----
        Raises KeyError if either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector)
        else:
            input_vector = word_or_vector
        if other_vectors is None:
            other_vectors = self.vectors

        if self.use_poincare_distance:
            return self.cosh_distances_mat(np.array([input_vector]), other_vectors)
        else:
            return 1 - VanillaWordEmbeddingsKeyedVectors.cosine_similarities(input_vector, other_vectors)

    def distance(self, word_or_vector1, word_or_vector2):
        """
        Compute distance between two words or vectors inside the Poincare ball.

        Example
        --------
        >>> trained_model.distance('woman', 'man')

        """
        v1 = self.word_vec(word_or_vector1) if isinstance(word_or_vector1, string_types) else word_or_vector1
        v2 = self.word_vec(word_or_vector2) if isinstance(word_or_vector2, string_types) else word_or_vector2
        if self.use_poincare_distance:
            diff = v1 - v2
            dist = arccosh(1 + 2 * dot(diff, diff) / (1 - dot(v1, v1) + EPS) / (1 - dot(v2, v2) + EPS))
            return dist
        else:
            return 1 - dot(matutils.unitvec(v1), matutils.unitvec(v2))

    def similarity(self, w1, w2):
        """
        Compute similarity between two words based on the Poincare distance between them.

        Example
        --------
        >>> trained_model.similarity('woman', 'man')

        """
        if self.use_poincare_distance:
            return -self.distance(w1, w2)**2
            # return -self.distance(w1, w2)**2 * norm(self[w1] - self[w2])**2
            # return -norm(self[w1] - self[w2])**2
        else:
            return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def embedding_norm(self, word_or_vector):
        """
        Compute embedding Poincare norm for a given word.

        Parameters
        ----------

        w : string
            word

        """
        v = self[word_or_vector] if isinstance(word_or_vector, string_types) else word_or_vector
        return arccosh(1 + 2 * dot(v, v) / (1 - dot(v, v)))

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            print("init_sims from PoincareWordEmbeddings")
            logger.info("precomputing L2-norms of word weight vectors; replace={}".format(replace))
            dtype = REAL
            if hasattr(self, 'vector_dtype'):
                dtype = self.vector_dtype
            self.vectors_norm = np.empty_like(self.vectors, dtype=dtype)
            # XXX: uncomment this line to compute gyrocosine
            # norms = self.embedding_norms_mat(self.vectors)
            norms = norm(self.vectors, axis=1)
            self.vectors_norm = (self.vectors / norms[:, None]).astype(dtype)

    @staticmethod
    def moebius_add_mat(A, B):
        """
        Return the result of the Moebius addition of the rows of matrix A with the rows of B.

        Parameters
        ----------
        A : numpy.array
            matrix, first argument of addition
        B : numpy.array
            matrix, second argument of addition

        Returns
        -------
        :obj: `numpy.array`
            matrix; Result of Moebius addition of the rows of matrix A with the rows of B

        """
        dot_aa = np.sum(A*A, axis=1)
        dot_bb = np.sum(B*B, axis=1)
        dot_ab = np.sum(A*B, axis=1)
        denominator = 1 + 2 * dot_ab + dot_aa * dot_bb
        coef_a = (1 + 2 * dot_ab + dot_bb) / denominator
        coef_b = (1 - dot_aa) / denominator

        return A * coef_a[:, None] + B * coef_b[:, None]

    @staticmethod
    def moebius_add(a, b):
        """
        Return the result of the Moebius addition of the two vectors, a + b

        Parameters
        ----------
        a : numpy.array
            vector, first argument of addition
        b : numpy.array
            vector, second argument of addition

        Returns
        -------
        :obj: `numpy.array`
            Result of Moebius addition a + b

        """
        dot_aa = dot(a, a)
        dot_bb = dot(b, b)
        dot_ab = dot(a, b)
        return ((1 + 2 * dot_ab + dot_bb) * a + (1 - dot_aa) * b) / (1 + 2 * dot_ab + dot_aa * dot_bb)

    @staticmethod
    def moebius_mul_mat(A, r):
        """
        Return the result of the Moebius scalar multiplication of vector v with scalar r

        Parameters
        ----------
        A : numpy.array (2D matrix)
        r : scalar

        Returns
        -------
        :obj: `numpy.array`
            Result of Moebius scalar multiplication between r and each of the rows of A

        """
        norm_v = norm(A, axis=1)
        return A * (tanh(r * arctanh(norm_v)) / (norm_v + 1e-10))[:, None]

    @staticmethod
    def moebius_mul(v, r):
        """
        Return the result of the Moebius scalar multiplication of vector v with scalar r

        Parameters
        ----------
        v : numpy.array (1D vector)
        r : scalar

        Returns
        -------
        :obj: `numpy.array`
            Result of Moebius scalar multiplication r * v

        """
        norm_v = norm(v)
        return tanh(r * arctanh(norm_v)) / norm_v * v

    @staticmethod
    def embedding_norms_mat(vectors):
        """
        Compute embedding Poincare norm for a set of vectors.

        Parameters
        ----------

        vectors : matrix
            np.array

        """
        dot_vv = (vectors * vectors).sum(axis=1)
        return arccosh(1 + 2 * dot_vv / (1 - dot_vv))

    @staticmethod
    def gyr(u, v, x):
        """
        Return the result of gyr[u, v](x).
        u : numpy.array (1D vector)
        v : numpy.array (1D vector)
        x : numpy.array (1D vector)

        Returns
        -------
        :obj: `numpy.array`
            Result of gyr[u, v](x)
        """
        a = PoincareWordEmbeddingsKeyedVectors.moebius_add(u, v)
        b = PoincareWordEmbeddingsKeyedVectors.moebius_add(u, PoincareWordEmbeddingsKeyedVectors.moebius_add(v, x))
        return PoincareWordEmbeddingsKeyedVectors.moebius_add(-a, b)

    @staticmethod
    def gyr_mat(u, v, x):
        """
        Return the result of gyr[u, v](x).
        u : numpy.array (2D matrix)
        v : numpy.array (2D matrix)
        x : numpy.array (2D matrix)

        Returns
        -------
        :obj: `numpy.array` (2D matrix)
            Result of gyr[u, v](x)
        """
        dot_uu = (u * u).sum(axis=1)  # batch_size x 1
        dot_vv = (v * v).sum(axis=1)  # batch_size x 1
        dot_uv = (u * v).sum(axis=1)  # batch_size x 1
        dot_ux = (u * x).sum(axis=1)  # batch_size x 1
        dot_vx = (v * x).sum(axis=1)  # batch_size x 1

        A = -dot_ux * dot_vv + dot_vx + 2 * dot_uv * dot_vx
        B = -dot_vx * dot_uu - dot_ux
        D = 1 + 2 * dot_uv + dot_uu * dot_vv

        coef_u = 2 * A / D
        coef_v = 2 * B / D
        return x + u * coef_u[:, None] + v * coef_v[:, None]

    @staticmethod
    def exp_map_mat(V, X):
        """
        Return the result of the exponential map applied from the tangent plane at point x, on the vector v that belongs
        to the tangent plane

        Parameters
        ----------
        V : numpy.array
            matrix, the rows are vectors that belong in the tangent plane at x
        X : numpy.array
            matrix, the rows are points on the manifold, where the tangent plane is considered

        Returns
        -------
        :obj: `numpy.array`
            Result of the exponential map on each of the rows of the output matrix

        """
        norm_v = np.linalg.norm(V, axis=1)
        dot_xx = np.sum(X*X, axis=1)
        coef = tanh(1.0/dot_xx * norm_v) / norm_v
        second_term = V * coef[:, None]
        return PoincareWordEmbeddingsKeyedVectors.moebius_add_mat(X, second_term)

    @staticmethod
    def log_map_mat(V, X):
        """
        Return the result of the logarithmic map. The resulting point belongs to the tangent plane at point x.
        Both x and v are points on the manifold

        Parameters
        ----------
        V : numpy.array
            matrix, the rows are vectors that belong to the manifold
        X : numpy.array
            matrix, the rows are points on the manifold, where the tangent plane is considered

        Returns
        -------
        :obj: `numpy.array`
            Result of the logarithmic map on each of the rows of the output matrix

        """
        add_result = PoincareWordEmbeddingsKeyedVectors.moebius_add_mat(-X, V)
        norm_add_result = np.linalg.norm(add_result, axis=1)
        dot_xx = np.sum(X*X, axis=1)
        coef = dot_xx * arctanh(norm_add_result) / norm_add_result
        return add_result * coef[:, None]


class MixPoincareWordEmbeddingsKeyedVectors(PoincareWordEmbeddingsKeyedVectors):
    def __init__(self, vector_size, num_embs, vector_dtype=REAL, trainables=None, init_near_border=False,
                 init_pretrained_config=False):
        super(MixPoincareWordEmbeddingsKeyedVectors, self).__init__(
            vector_size=vector_size, vector_dtype=vector_dtype, trainables=trainables,
            init_near_border=init_near_border, init_pretrained_config=init_pretrained_config)

        self.num_embs = num_embs
        self.small_emb_size = int(vector_size / num_embs)

    def batch_most_similar_mix_hyperbolic_analogy(self, positive=None, negative=None, restrict_vocab=None, debug=False):
        """
        Solve an analogy task. The result should be similar to the positive words and unlike the negative word.

        This method computes the similarity (according to the formula defined for the hyperbolic space) between
        the parallel transport of the input vector and the vectors for each word in the model and selects the word
        that is closest to the position of the parallel transported vector.

        Parameters
        ----------
        positive : list of two numpy.array
            List of two 2D numpy arrays. Each of them contains positive instances. The number of rows is equal to the
            number of questions in a batch.
        negative : numpy.array
            2D array that contains on each row the embedding of the negative word from that question. The number of
            rows is equal to the number of questions in a batch.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        batch_size = len(negative)

        # XXX: before calling this method, #accuracy is setting self.vocab to be only the restricted vocab.
        # So here, self.vocab is actually self.vocab[:restricted]

        if not self.use_poincare_distance:
            self.init_sims()

        # Retrieve embeddings.
        pos_emb = [
            self.vectors[positive[0]],
            self.vectors[positive[1]]
        ]
        neg_emb = self.vectors[negative]

        parallel_transp1 = empty((batch_size, self.vector_size), dtype=self.vector_dtype)
        parallel_transp2 = empty((batch_size, self.vector_size), dtype=self.vector_dtype)
        aux = empty((batch_size, self.vector_size), dtype=self.vector_dtype)
        lin_comb_point = empty((batch_size, self.vector_size), dtype=self.vector_dtype)
        small_emb_size = int(self.vector_size / self.num_embs)

        # Compute gyro-parallel transport in each of the small dimensional spaces.
        for i in range(self.num_embs):
            # Compute the parallel transport of the positive vector in the analogy question (i.e. c) using the new
            # formula
            start = small_emb_size * i
            end = small_emb_size * (i+1)
            parallel_transp1[:, start:end] = self.moebius_add_mat(
                pos_emb[1][:, start:end],
                self.gyr_mat(pos_emb[1][:, start:end],
                             -neg_emb[:, start:end],
                             self.moebius_add_mat(-neg_emb[:, start:end],
                                                  pos_emb[0][:, start:end])))  # batch_size x vector_size

            # Compute the parallel transport of the other positive vector (i.e. b) so the alternative formulation of the
            # analogy question.
            parallel_transp2[:, start:end] = self.moebius_add_mat(
                pos_emb[0][:, start:end],
                self.gyr_mat(pos_emb[0][:, start:end], -neg_emb[:, start:end],
                             self.moebius_add_mat(-neg_emb[:, start:end],
                                                  pos_emb[1][:, start:end])))  # batch_size x vector_size

            aux[:, start:end] = self.moebius_add_mat(-parallel_transp1[:, start:end], parallel_transp2[:, start:end])

        # Compute the gyrolinear combination between the two parallel
        # transported points.
        t = 0.3
        results = []

        for i in range(self.num_embs):
            start = small_emb_size * i
            end = small_emb_size * (i+1)
            lin_comb_point[:, start:end] = self.moebius_add_mat(parallel_transp1[:, start:end],
                                                                self.moebius_mul_mat(aux[:, start:end], t))

        # Compute similarity between parallel transported input and all words in the vocabulary.
        if self.use_poincare_distance:
            limited = self.vectors if restrict_vocab is None else self.vectors[:restrict_vocab]  # vocab_size * vector_size
        else:
            # Get normalized vectors, if we use cosine distance.
            limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]  # vocab_size * vector_size
            # Normalize parallel transported vector.
            for i in range(self.num_embs):
                start = small_emb_size * i
                end = small_emb_size * (i+1)
                lin_comb_point[:, start:end] = lin_comb_point[:, start:end] / (norm(lin_comb_point[:, start:end], axis=1)[:, None] + 1e-5)

        dists = self.mix_distances_mat(lin_comb_point, limited)  # batch_size * vocab_size

        max_float = np.finfo(dists.dtype).max
        batch_size_range = np.arange(batch_size)
        x = np.concatenate((batch_size_range, batch_size_range, batch_size_range))
        y = np.concatenate((positive[0], positive[1], negative))
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)
        dists[x, y] = max_float  # batch_size * (vocab_size - 3)

        best = []
        if debug:
            for i in batch_size_range:
                top_ids = matutils.argsort(dists[i], topn=10)
                curr_best = [(self.index2word[idx],
                              self.distances(limited[idx], np.array([neg_emb[i], pos_emb[0][i], pos_emb[1][i]]))[0])
                             for idx in top_ids]
                best.append(curr_best)

        best_ids = np.argmin(dists, axis=1)
        result = (
            [self.index2word[i] for i in best_ids],
            dists[batch_size_range, best_ids].astype(np.float32),
            best)
        results.append(result)
        return results

    def mix_distances_mat(self, vectors, other_vectors=None):
        """
        Return distance in the product of hyperbolic spaces, between the rows of `vectors` and the rows of
        `other_vectors`.

        Parameters
        ----------
        vectors: numpy.array
            Vectors from which distances are to be computed.
        other_vectors: numpy.array
            For each vector in `other_vectors` distance from each vector in `vectors` is computed.
            If None or empty, all words in vocab are considered (including the vectors in `vectors`).

        Returns
        -------
        np.array
            Returns a numpy.array that contains the distance between each row in `vectors`
            and each row in `other_vectors`
        """

        dists = zeros((vectors.shape[0], other_vectors.shape[0]), dtype=self.vector_dtype)
        small_emb_size = int(self.vector_size / self.num_embs)

        if self.use_poincare_distance == True:
            for i in range(self.num_embs):
                start = small_emb_size * i
                end = small_emb_size * (i+1)
                curr_dists = np.arccosh(
                    self.cosh_distances_mat(vectors[:, start:end], other_vectors[:, start:end]))
                dists += curr_dists * curr_dists
            dists = np.sqrt(dists)
        else:
            # The vectors need to be normalized!!!
            for i in range(self.num_embs):
                start = small_emb_size * i
                end = small_emb_size * (i+1)
                curr_dists = -dot(vectors[:, start:end], other_vectors[:, start:end].T)
                dists += curr_dists

        return dists

    def distances(self, word_or_vector, other_vectors=None):
        """
        Compute the distance in a product of Poincare balls from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : str or numpy.array
            Word or vector from which distances are to be computed.

        other_vectors: numpy.array or None
            For each vector in `other_vectors` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words` from input `word_or_vector`,
            in the same order as `other_words`.

        Notes
        -----
        Raises KeyError if either `word_or_vector` or any word in `other_words` is absent from vocab.

        """
        if self.use_poincare_distance:
            use_norm = False
        else:
            use_norm = True

        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector, use_norm=use_norm)
        else:
            input_vector = word_or_vector
        if other_vectors is None:
            if use_norm:
                other_vectors = self.vectors_norm
            else:
                other_vectors = self.vectors

        return self.mix_distances_mat(np.array([input_vector]), other_vectors)

    def distance(self, word_or_vector1, word_or_vector2):
        """
        Compute distance between two words or vectors represented in a Cartesian product of Poincare balls.

        Example
        --------
        >>> trained_model.distance('woman', 'man')

        """
        if self.use_poincare_distance:
            use_norm = False
        else:
            use_norm = True
        v1 = self.word_vec(word_or_vector1, use_norm=use_norm) if isinstance(word_or_vector1, string_types) else word_or_vector1
        v2 = self.word_vec(word_or_vector2, use_norm=use_norm) if isinstance(word_or_vector2, string_types) else word_or_vector2

        return self.mix_distances_mat(np.array([v1]), np.array([v2]))[0][0]

    def similarity(self, w1, w2):
        """
        Compute similarity between two words based on the Poincare distance between them.

        Example
        --------
        >>> trained_model.similarity('woman', 'man')

        """
        return -self.distance(w1, w2)

    def embedding_norm(self, word_or_vector):
        """
        Compute embedding norm in product of Poincare balls for a given word.

        Parameters
        ----------

        w : string
            word

        """
        v = self[word_or_vector] if isinstance(word_or_vector, string_types) else word_or_vector
        small_emb_size = int(self.vector_size / self.num_embs)
        norms = empty(self.num_embs)
        for i in range(self.num_embs):
            start = small_emb_size * i
            end = small_emb_size * (i+1)
            if self.use_poincare_distance:
                norms[i] = arccosh(1 + 2 * dot(v[start:end], v[start:end]) / (1 - dot(v[start:end], v[start:end])))
            else:
                norms[i] = norm(v[start:end])
        return norm(norms)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'vectors_norm', None) is None or replace:
            print("init_sims from MixPoincareWordEmbeddings")
            logger.info("precomputing L2-norms of word weight vectors; replace={}".format(replace))
            dtype = REAL
            if hasattr(self, 'vector_dtype'):
                dtype = self.vector_dtype
            self.vectors_norm = np.empty_like(self.vectors, dtype=dtype)
            small_emb_size = int(self.vector_size / self.num_embs)
            for i in range(self.num_embs):
                start = small_emb_size * i
                end = small_emb_size * (i+1)
                # norms = PoincareWordEmbeddingsKeyedVectors.embedding_norms_mat(self.vectors[:, start:end]) + 1e-5
                norms = norm(self.vectors[:, start:end], axis=1) + 1e-5
                self.vectors_norm[:, start:end] = (self.vectors[:, start:end] / norms[:, None]).astype(dtype)


class VanillaWordEmbeddingsKeyedVectors(WordEmbeddingsKeyedVectors):
    """
    Class used as base class for vanilla word embeddings that use cosine similarity (e.g. word2vec, fasttext).
    """

    def most_similar(self, positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None, debug=False):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive : :obj: `list` of :obj: `str`
            List of words that contribute positively.
        negative : :obj: `list` of :obj: `str`
            List of words that contribute negatively.
        topn : int
            Number of top-N similar words to return.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        Examples
        --------
        >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
        [('queen', 0.50882536), ...]

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
            (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in negative
        ]

        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            else:
                mean.append(weight * self.word_vec(word, use_norm=True))
                if word in self.vocab:
                    all_words.add(self.vocab[word].index)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        dtype = REAL
        if hasattr(self, "vector_dtype"):
            dtype = self.vector_dtype
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(dtype)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]
        # Compute 3COSADD.
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    def batch_most_similar_analogy(self, positive=None, negative=None, restrict_vocab=None, debug=False):
        """
        Solve an analogy task. The result should be similar to the positive words and unlike the negative word.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words and the vectors for each word in the model.
        The method corresponds to the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive : list of two numpy.array
            List of two 2D numpy arrays. Each of them contains positive instances. The number of rows is equal to the
            number of questions in a batch.
        negative : numpy.array
            2D array that contains on each row the embedding of the negative word from that question. The number of
            rows is equal to the number of questions in a batch.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        self.init_sims()
        batch_size = len(negative)

        # XXX: before calling this method, #accuracy is setting self.vocab to be only the restricted vocab.
        # So here, self.vocab is actually self.vocab[:restricted]

        # Retrieve embeddings.
        pos_emb = [
            self.vectors_norm[positive[0]],
            self.vectors_norm[positive[1]]
        ]
        neg_emb = self.vectors_norm[negative]

        # compute the weighted average of all input words, where positive words have weight 1
        # and negative words have weight -1
        weighted_mean = (pos_emb[0] + pos_emb[1] - neg_emb) / 3  # batch_size * vector_size
        mean_norm = norm(weighted_mean, axis=1)
        weighted_mean = weighted_mean / mean_norm[:, None]

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]  # vocab_size * vector_size
        # Compute 3COSADD.
        sims = dot(weighted_mean, limited.T)  # batch_size * vocab_size

        min_float = np.finfo(sims.dtype).min
        batch_size_range = np.arange(batch_size)
        x = np.concatenate((batch_size_range, batch_size_range, batch_size_range))
        y = np.concatenate((positive[0], positive[1], negative))
        sims[x, y] = min_float  # batch_size * (vocab_size - 3)

        best = []
        if debug:
            for i in batch_size_range:
                top_ids = matutils.argsort(sims[i], topn=10, reverse=True)
                curr_best = [(self.index2word[idx],
                              self.distances(limited[idx], np.array([neg_emb[i], pos_emb[0][i], pos_emb[1][i]])))
                             for idx in top_ids]
                best.append(curr_best)

        best_ids = np.argmax(sims, axis=1)
        result = (
            [self.index2word[i] for i in best_ids],
            sims[batch_size_range, best_ids].astype(np.float32),
            best)
        return [result]

    def batch_most_similar_cosmul_analogy(self, positive=None, negative=None, restrict_vocab=None, debug=False):
        """
        Solve an analogy task. The result should be similar to the positive words and unlike the negative word.

        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        Parameters
        ----------
        positive : list of two numpy.array
            List of two 2D numpy arrays. Each of them contains positive instances. The number of rows is equal to the
            number of questions in a batch.
        negative : numpy.array
            2D array that contains on each row the embedding of the negative word from that question. The number of
            rows is equal to the number of questions in a batch.
        restrict_vocab : int
            Optional integer which limits the range of vectors which
            are searched for most-similar values. For example, restrict_vocab=10000 would
            only check the first 10000 word vectors in the vocabulary order. (This may be
            meaningful if you've sorted the vocabulary by descending frequency.)

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (word, similarity)

        """
        self.init_sims()
        batch_size = len(negative)

        # XXX: before calling this method, #accuracy is setting self.vocab to be only the restricted vocab.
        # So here, self.vocab is actually self.vocab[:restricted]

        # Retrieve embeddings.
        pos_emb = [
            self.vectors_norm[positive[0]],
            self.vectors_norm[positive[1]]
        ]
        neg_emb = self.vectors_norm[negative]

        # # compute the weighted average of all input words, where positive words have weight 1
        # # and negative words have weight -1
        # weighted_mean = (pos_emb[0] + pos_emb[1] - neg_emb) / 3  # batch_size * vector_size
        # mean_norm = norm(weighted_mean, axis=1)
        # weighted_mean = weighted_mean / mean_norm[:, None]

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]  # vocab_size * vector_size
        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [
            (1 + dot(pos_emb[0], limited.T)) / 2,  # batch_size * vocab_size
            (1 + dot(pos_emb[1], limited.T)) / 2   # batch_size * vocab_size
        ]
        neg_dists = (1 + dot(neg_emb, limited.T)) / 2  # batch_size * vocab_size
        sims = pos_dists[0] * pos_dists[1] / (neg_dists + 0.000001)  # batch_size * vocab_size

        min_float = np.finfo(sims.dtype).min
        batch_size_range = np.arange(batch_size)
        x = np.concatenate((batch_size_range, batch_size_range, batch_size_range))
        y = np.concatenate((positive[0], positive[1], negative))
        sims[x, y] = min_float  # batch_size * (vocab_size - 3)

        best_ids = np.argmax(sims, axis=1)
        result = (
            [self.index2word[i] for i in best_ids],
            sims[batch_size_range, best_ids].astype(np.float32))
        return [result]

    def most_similar_cosmul(self, positive=None, negative=None, topn=10, restrict_vocab=None, debug=False):
        """
        Find the top-N most similar words, using the multiplicative combination objective
        proposed by Omer Levy and Yoav Goldberg. Positive words still contribute
        positively towards the similarity, negative words negatively, but with less
        susceptibility to one large distance dominating the calculation.

        In the common analogy-solving case, of two positive and one negative examples,
        this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

        Additional positive or negative examples contribute to the numerator or denominator,
        respectively – a potentially sensible but untested extension of the method. (With
        a single positive example, rankings will be the same as in the default most_similar.)

        Example::

          >>> trained_model.most_similar_cosmul(positive=['baghdad', 'england'], negative=['london'])
          [(u'iraq', 0.8488819003105164), ...]

        .. Omer Levy and Yoav Goldberg. Linguistic Regularities in Sparse and Explicit Word Representations, 2014.

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()

        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
            positive = [positive]

        all_words = {
            self.vocab[word].index for word in positive + negative
            if not isinstance(word, ndarray) and word in self.vocab
        }

        positive = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in positive
        ]
        negative = [
            self.word_vec(word, use_norm=True) if isinstance(word, string_types) else word
            for word in negative
        ]

        if not positive:
            raise ValueError("cannot compute similarity with no input")

        limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[:restrict_vocab]

        # equation (4) of Levy & Goldberg "Linguistic Regularities...",
        # with distances shifted to [0,1] per footnote (7)
        pos_dists = [((1 + dot(limited, term)) / 2) for term in positive]
        neg_dists = [((1 + dot(limited, term)) / 2) for term in negative]
        dists = prod(pos_dists, axis=0) / (prod(neg_dists, axis=0) + 0.000001)

        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
        return result[:topn]

    @staticmethod
    def cosine_similarities(vector_1, vectors_all):
        """
        Return cosine similarities between one vector and a set of other vectors.

        Parameters
        ----------
        vector_1 : numpy.array
            vector from which similarities are to be computed.
            expected shape (dim,)
        vectors_all : numpy.array
            for each row in vectors_all, distance from vector_1 is computed.
            expected shape (num_vectors, dim)

        Returns
        -------
        :obj: `numpy.array`
            Contains cosine distance between vector_1 and each row in vectors_all.
            shape (num_vectors,)

        """
        norm = np.linalg.norm(vector_1)
        all_norms = np.linalg.norm(vectors_all, axis=1)
        dot_products = dot(vectors_all, vector_1)
        similarities = dot_products / (norm * all_norms)
        return similarities

    def distances(self, word_or_vector, other_words_or_vectors=()):
        """
        Compute cosine distances from given word or vector to all words in `other_words`.
        If `other_words` is empty, return distance between `word_or_vectors` and all words in vocab.

        Parameters
        ----------
        word_or_vector : str or numpy.array
            Word or vector from which distances are to be computed.

        other_words_or_vectors : iterable(str) or numpy.array
            For each word in `other_words_or_vectors` distance from `word_or_vector` is computed.
            If None or empty, distance of `word_or_vector` from all words in vocab is computed (including itself).

        Returns
        -------
        numpy.array
            Array containing distances to all words in `other_words_or_vectors` from input `word_or_vector`,
            in the same order as `other_words`.

        Notes
        -----
        Raises KeyError if either `word_or_vector` or any word in `other_words_or_vectors` is absent from vocab.

        """
        if isinstance(word_or_vector, string_types):
            input_vector = self.word_vec(word_or_vector)
        else:
            input_vector = word_or_vector
        if not len(other_words_or_vectors):
            other_vectors = self.vectors
        else:
            if isinstance(other_words_or_vectors[0], string_types):
                other_indices = [self.vocab[word].index for word in other_words_or_vectors]
                other_vectors = self.vectors[other_indices]
            else:
                other_vectors = other_words_or_vectors
        return 1 - self.cosine_similarities(input_vector, other_vectors)

    def distance(self, word_or_vector1, word_or_vector2):
        """
        Compute cosine distance between two words.

        Examples
        --------

        >>> trained_model.distance('woman', 'man')
        0.34

        >>> trained_model.distance('woman', 'woman')
        0.0

        """
        v1 = self.word_vec(word_or_vector1) if isinstance(word_or_vector1, string_types) else word_or_vector1
        v2 = self.word_vec(word_or_vector2) if isinstance(word_or_vector2, string_types) else word_or_vector2

        return 1 - dot(matutils.unitvec(v1), matutils.unitvec(v2))

    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Examples
        --------

        >>> trained_model.similarity('woman', 'man')
        0.73723527

        >>> trained_model.similarity('woman', 'woman')
        1.0

        """
        return dot(matutils.unitvec(self[w1]), matutils.unitvec(self[w2]))

    def embedding_norm(self, word_or_vector):
        """
        Compute embedding norm for a given word.

        Parameters
        ----------

        word_or_vector : string or array
            word or vector

        """
        v = self.word_vec(word_or_vector) if isinstance(word_or_vector, string_types) else word_or_vector
        return norm(v)

    def n_similarity(self, ws1, ws2):
        """
        Compute cosine similarity between two sets of words.

        Examples
        --------

        >>> trained_model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
        0.61540466561049689

        >>> trained_model.n_similarity(['restaurant', 'japanese'], ['japanese', 'restaurant'])
        1.0000000000000004

        >>> trained_model.n_similarity(['sushi'], ['restaurant']) == trained_model.similarity('sushi', 'restaurant')
        True

        """
        if not(len(ws1) and len(ws2)):
            raise ZeroDivisionError('At least one of the passed list is empty.')
        v1 = [self[word] for word in ws1]
        v2 = [self[word] for word in ws2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Parameters
        ----------
        words : :obj: `list` of :obj: `str`
            List of words

        Returns
        -------
        str
            The word further away from the mean of all words.

        Example
        -------
        >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
        'cereal'

        """
        self.init_sims()

        used_words = [word for word in words if word in self]
        if len(used_words) != len(words):
            ignored_words = set(words) - set(used_words)
            logger.warning("vectors for words %s are not present in the model, ignoring these words", ignored_words)
        if not used_words:
            raise ValueError("cannot select a word from an empty list")
        vectors = vstack(self.word_vec(word, use_norm=True) for word in used_words).astype(self.vector_dtype)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(self.vector_dtype)
        dists = dot(vectors, mean)
        return sorted(zip(dists, used_words))[0][1]


class Word2VecKeyedVectors(VanillaWordEmbeddingsKeyedVectors):
    """Class to contain vectors and vocab for word2vec model.
    Used to perform operations on the vectors such as vector lookup, distance, similarity etc.
    """
    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)

        """
        # from gensim.models.word2vec import save_word2vec_format
        _save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, encoding='utf8', unicode_errors='strict',
                             limit=None, datatype=REAL):
        """Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        Parameters
        ----------
        fname : str
            The file path to the saved word2vec-format file.
        fvocab : str
                Optional file path to the vocabulary.Word counts are read from `fvocab` filename,
                if set (this is the file generated by `-save-vocab` flag of the original C tool).
        binary : bool
            If True, indicates whether the data is in binary word2vec format.
        encoding : str
            If you trained the C model using non-utf8 encoding for words, specify that
            encoding in `encoding`.
        unicode_errors : str
            default 'strict', is a string suitable to be passed as the `errors`
            argument to the unicode() (Python 2.x) or str() (Python 3.x) function. If your source
            file may include word tokens truncated in the middle of a multibyte unicode character
            (as is common from the original word2vec.c tool), 'ignore' or 'replace' may help.
        limit : int
            Sets a maximum number of word-vectors to read from the file. The default,
            None, means read all.
        datatype : :class: `numpy.float*`
            (Experimental) Can coerce dimensions to a non-default float type (such
            as np.float16) to save memory. (Such types may result in much slower bulk operations
            or incompatibility with optimized routines.)

        Returns
        -------
        :obj: `~gensim.models.word2vec.Wod2Vec`
            Returns the loaded model as an instance of :class: `~gensim.models.word2vec.Wod2Vec`.

        """
        # from gensim.models.word2vec import load_word2vec_format
        return _load_word2vec_format(
            Word2VecKeyedVectors, fname, fvocab=fvocab, binary=binary, encoding=encoding, unicode_errors=unicode_errors,
            limit=limit, datatype=datatype)

    def get_keras_embedding(self, train_embeddings=False):
        """Return a Keras 'Embedding' layer with weights set as the Word2Vec model's learned word embeddings

        Parameters
        ----------
        train_embeddings : bool
            If False, the weights are frozen and stopped from being updated.
            If True, the weights can/will be further trained/updated.

        Returns
        -------
        :obj: `keras.layers.Embedding`
            Embedding layer

        """
        try:
            from keras.layers import Embedding
        except ImportError:
            raise ImportError("Please install Keras to use this function")
        weights = self.vectors

        # set `trainable` as `False` to use the pretrained word embedding
        # No extra mem usage here as `Embedding` layer doesn't create any new matrix for weights
        layer = Embedding(
            input_dim=weights.shape[0], output_dim=weights.shape[1],
            weights=[weights], trainable=train_embeddings
        )
        return layer


KeyedVectors = Word2VecKeyedVectors  # alias for backward compatibility


class Doc2VecKeyedVectors(BaseKeyedVectors):

    def __init__(self, vector_size, mapfile_path):
        super(Doc2VecKeyedVectors, self).__init__(vector_size=vector_size)
        self.doctags = {}  # string -> Doctag (only filled if necessary)
        self.max_rawint = -1  # highest rawint-indexed doctag
        self.offset2doctag = []  # int offset-past-(max_rawint+1) -> String (only filled if necessary)
        self.count = 0
        self.vectors_docs = []
        self.mapfile_path = mapfile_path
        self.vector_size = vector_size
        self.vectors_docs_norm = None

    @property
    def index2entity(self):
        return self.offset2doctag

    @index2entity.setter
    def index2entity(self, value):
        self.offset2doctag = value

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs instead")
    def doctag_syn0(self):
        return self.vectors_docs

    @property
    @deprecated("Attribute will be removed in 4.0.0, use docvecs.vectors_docs_norm instead")
    def doctag_syn0norm(self):
        return self.vectors_docs_norm

    def __getitem__(self, index):
        """
        Accept a single key (int or string tag) or list of keys as input.

        If a single string or int, return designated tag's vector
        representation, as a 1D numpy array.

        If a list, return designated tags' vector representations as a
        2D numpy array: #tags x #vector_size.
        """
        if index in self:
            if isinstance(index, string_types + integer_types + (integer,)):
                return self.vectors_docs[self._int_index(index, self.doctags, self.max_rawint)]
            return vstack([self[i] for i in index])
        raise KeyError("tag '%s' not seen in training corpus/invalid" % index)

    def __contains__(self, index):
        if isinstance(index, integer_types + (integer,)):
            return index < self.count
        else:
            return index in self.doctags

    def __len__(self):
        return self.count

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.doc2vec.Doc2VecKeyedVectors.load` which supports
        operations on trained document vectors like `most_similar`.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['vectors_docs_norm'])
        super(Doc2VecKeyedVectors, self).save(*args, **kwargs)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training or inference** after doing a replace.
        The model becomes effectively read-only = you can call `most_similar`, `similarity`
        etc., but not `train` or `infer_vector`.

        """
        if getattr(self, 'vectors_docs_norm', None) is None or replace:
            logger.info("precomputing L2-norms of doc weight vectors")
            if replace:
                for i in xrange(self.vectors_docs.shape[0]):
                    self.vectors_docs[i, :] /= sqrt((self.vectors_docs[i, :] ** 2).sum(-1))
                self.vectors_docs_norm = self.vectors_docs
            else:
                if self.mapfile_path:
                    self.vectors_docs_norm = np_memmap(
                        self.mapfile_path + '.vectors_docs_norm', dtype=REAL,
                        mode='w+', shape=self.vectors_docs.shape)
                else:
                    self.vectors_docs_norm = empty(self.vectors_docs.shape, dtype=REAL)
                np_divide(
                    self.vectors_docs, sqrt((self.vectors_docs ** 2).sum(-1))[..., newaxis], self.vectors_docs_norm)

    def most_similar(self, positive=None, negative=None, topn=10, clip_start=0, clip_end=None, indexer=None):
        """
        Find the top-N most similar docvecs known from training. Positive docs contribute
        positively towards the similarity, negative docs negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given docs. Docs may be specified as vectors, integer indexes
        of trained docvecs, or if the documents were originally presented with string tags,
        by the corresponding tags.

        The 'clip_start' and 'clip_end' allow limiting results to a particular contiguous
        range of the underlying `vectors_docs_norm` vectors. (This may be useful if the ordering
        there was chosen to be significant, such as more popular tag IDs in lower indexes.)

        Parameters
        ----------
        positive : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute positively.
        negative : :obj: `list`
            List of Docs specifed as vectors, integer indexes of trained docvecs or string tags
            that contribute negatively.
        topn : int
            Number of top-N similar docvecs to return.
        clip_start : int
            Start clipping index.
        clip_end : int
            End clipping index.

        Returns
        -------
        :obj: `list` of :obj: `tuple`
            Returns a list of tuples (doc, similarity)

        """
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        self.init_sims()
        clip_end = clip_end or len(self.vectors_docs_norm)

        if isinstance(positive, string_types + integer_types + (integer,)) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each doc, if not already present; default to 1.0 for positive and -1.0 for negative docs
        positive = [
            (doc, 1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in positive
        ]
        negative = [
            (doc, -1.0) if isinstance(doc, string_types + integer_types + (ndarray, integer))
            else doc for doc in negative
        ]

        # compute the weighted average of all docs
        all_docs, mean = set(), []
        for doc, weight in positive + negative:
            if isinstance(doc, ndarray):
                mean.append(weight * doc)
            elif doc in self.doctags or doc < self.count:
                mean.append(weight * self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)])
                all_docs.add(self._int_index(doc, self.doctags, self.max_rawint))
            else:
                raise KeyError("doc '%s' not in trained set" % doc)
        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        dists = dot(self.vectors_docs_norm[clip_start:clip_end], mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_docs), reverse=True)
        # ignore (don't return) docs from the input
        result = [
            (self._index_to_doctag(sim + clip_start, self.offset2doctag, self.max_rawint), float(dists[sim]))
            for sim in best
            if (sim + clip_start) not in all_docs
        ]
        return result[:topn]

    def doesnt_match(self, docs):
        """
        Which doc from the given list doesn't go with the others?

        (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        docs : :obj: `list` of (str or int)
            List of seen documents specified by their corresponding string tags or integer indices.

        Returns
        -------
        str or int
            The document further away from the mean of all the documents.

        """
        self.init_sims()

        docs = [doc for doc in docs if doc in self.doctags or 0 <= doc < self.count]  # filter out unknowns
        logger.debug("using docs %s", docs)
        if not docs:
            raise ValueError("cannot select a doc from an empty list")
        vectors = vstack(
            self.vectors_docs_norm[self._int_index(doc, self.doctags, self.max_rawint)] for doc in docs).astype(REAL)
        mean = matutils.unitvec(vectors.mean(axis=0)).astype(REAL)
        dists = dot(vectors, mean)
        return sorted(zip(dists, docs))[0][1]

    def similarity(self, d1, d2):
        """
        Compute cosine similarity between two docvecs in the trained set, specified by int index or
        string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        d1 : int or str
            Indicate the first document by it's string tag or integer index.
        d2 : int or str
            Indicate the second document by it's string tag or integer index.

        Returns
        -------
        float
            The cosine similarity between the vectors of the two documents.

        """
        return dot(matutils.unitvec(self[d1]), matutils.unitvec(self[d2]))

    def n_similarity(self, ds1, ds2):
        """
        Compute cosine similarity between two sets of docvecs from the trained set, specified by int
        index or string tag. (TODO: Accept vectors of out-of-training-set docs, as if from inference.)

        Parameters
        ----------
        ds1 : :obj: `list` of (str or int)
            Specify the first set of documents as a list of their integer indices or string tags.
        ds2 : :obj: `list` of (str or int)
            Specify the second set of documents as a list of their integer indices or string tags.

        Returns
        -------
        float
            The cosine similarity between the means of the documents in each of the two sets.

        """
        v1 = [self[doc] for doc in ds1]
        v2 = [self[doc] for doc in ds2]
        return dot(matutils.unitvec(array(v1).mean(axis=0)), matutils.unitvec(array(v2).mean(axis=0)))

    def distance(self, d1, d2):
        """
        Compute cosine distance between two documents.

        """
        return 1 - self.similarity(d1, d2)

    # required by base keyed vectors class
    def distances(self, d1, other_docs=()):
        """Compute distances from given document (string tag or int index) to all documents in `other_docs`.
        If `other_docs` is empty, return distance between `d1` and all documents seen during training.
        """
        input_vector = self[d1]
        if not other_docs:
            other_vectors = self.vectors_docs
        else:
            other_vectors = self[other_docs]
        return 1 - WordEmbeddingsKeyedVectors.cosine_similarities(input_vector, other_vectors)

    def similarity_unseen_docs(self, model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5):
        """
        Compute cosine similarity between two post-bulk out of training documents.

        Parameters
        ----------
        model : :obj: `~gensim.models.doc2vec.Doc2Vec`
            An instance of a trained `Doc2Vec` model.
        doc_words1 : :obj: `list` of :obj: `str`
            The first document. Document should be a list of (word) tokens.
        doc_words2 : :obj: `list` of :obj: `str`
            The second document. Document should be a list of (word) tokens.
        alpha : float
            The initial learning rate.
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        steps : int
            Number of times to train the new document.

        Returns
        -------
        float
            The cosine similarity between the unseen documents.

        """
        d1 = model.infer_vector(doc_words=doc_words1, alpha=alpha, min_alpha=min_alpha, steps=steps)
        d2 = model.infer_vector(doc_words=doc_words2, alpha=alpha, min_alpha=min_alpha, steps=steps)
        return dot(matutils.unitvec(d1), matutils.unitvec(d2))

    def save_word2vec_format(self, fname, prefix='*dt_', fvocab=None,
                             total_vec=None, binary=False, write_first_line=True):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        prefix : str
            Uniquely identifies doctags from word vocab, and avoids collision
            in case of repeated string in doctag and word vocab.
        fvocab : str
            Optional file path used to save the vocabulary
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards)
        write_first_line : bool
            Whether to print the first line in the file. Useful when saving doc-vectors after word-vectors.

        """
        total_vec = total_vec or len(self)
        with utils.smart_open(fname, 'ab') as fout:
            if write_first_line:
                logger.info("storing %sx%s projection weights into %s", total_vec, self.vectors_docs.shape[1], fname)
                fout.write(utils.to_utf8("%s %s\n" % (total_vec, self.vectors_docs.shape[1])))
            # store as in input order
            for i in range(len(self)):
                doctag = u"%s%s" % (prefix, self._index_to_doctag(i, self.offset2doctag, self.max_rawint))
                row = self.vectors_docs[i]
                if binary:
                    fout.write(utils.to_utf8(doctag) + b" " + row.tostring())
                else:
                    fout.write(utils.to_utf8("%s %s\n" % (doctag, ' '.join("%f" % val for val in row))))

    @staticmethod
    def _int_index(index, doctags, max_rawint):
        """Return int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset

    @staticmethod
    def _index_to_doctag(i_index, offset2doctag, max_rawint):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - max_rawint - 1
        if 0 <= candidate_offset < len(offset2doctag):
            return offset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def index_to_doctag(self, i_index):
        """Return string key for given i_index, if available. Otherwise return raw int doctag (same int)."""
        candidate_offset = i_index - self.max_rawint - 1
        if 0 <= candidate_offset < len(self.offset2doctag):
            return self.ffset2doctag[candidate_offset]
        else:
            return i_index

    # for backward compatibility
    def int_index(self, index, doctags, max_rawint):
        """Return int index for either string or int index"""
        if isinstance(index, integer_types + (integer,)):
            return index
        else:
            return max_rawint + 1 + doctags[index].offset


class FastTextKeyedVectors(VanillaWordEmbeddingsKeyedVectors):
    """
    Class to contain vectors and vocab for the FastText training class and other methods not directly
    involved in training such as most_similar()
    """

    def __init__(self, vector_size, min_n, max_n):
        super(FastTextKeyedVectors, self).__init__(vector_size=vector_size)
        self.vectors_vocab = None
        self.vectors_vocab_norm = None
        self.vectors_ngrams = None
        self.vectors_ngrams_norm = None
        self.buckets_word = None
        self.hash2index = {}
        self.min_n = min_n
        self.max_n = max_n
        self.num_ngram_vectors = 0

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab instead")
    def syn0_vocab(self):
        return self.vectors_vocab

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_vocab_norm instead")
    def syn0_vocab_norm(self):
        return self.vectors_vocab_norm

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams instead")
    def syn0_ngrams(self):
        return self.vectors_ngrams

    @property
    @deprecated("Attribute will be removed in 4.0.0, use self.wv.vectors_ngrams_norm instead")
    def syn0_ngrams_norm(self):
        return self.vectors_ngrams_norm

    def __contains__(self, word):
        """
        Check if `word` or any character ngrams in `word` are present in the vocabulary.
        A vector for the word is guaranteed to exist if `__contains__` returns True.
        """
        if word in self.vocab:
            return True
        else:
            char_ngrams = _compute_ngrams(word, self.min_n, self.max_n)
            return any(_ft_hash(ng) % self.bucket in self.hash2index for ng in char_ngrams)

    def save(self, *args, **kwargs):
        """Saves the keyedvectors. This saved model can be loaded again using
        :func:`~gensim.models.fasttext.FastTextKeyedVectors.load` which supports
        getting vectors for out-of-vocabulary words.

        Parameters
        ----------
        fname : str
            Path to the file.

        """
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get(
            'ignore', ['vectors_norm', 'vectors_vocab_norm', 'vectors_ngrams_norm', 'buckets_word'])
        super(FastTextKeyedVectors, self).save(*args, **kwargs)

    def word_vec(self, word, use_norm=False):
        """
        Accept a single word as input.
        Returns the word's representations in vector space, as a 1D numpy array.

        If `use_norm` is True, returns the normalized word vector.

        """
        if word in self.vocab:
            return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        else:
            # from gensim.models.fasttext import compute_ngrams
            word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
            ngrams = _compute_ngrams(word, self.min_n, self.max_n)
            if use_norm:
                ngram_weights = self.vectors_ngrams_norm
            else:
                ngram_weights = self.vectors_ngrams
            ngrams_found = 0
            for ngram in ngrams:
                ngram_hash = _ft_hash(ngram) % self.bucket
                if ngram_hash in self.hash2index:
                    word_vec += ngram_weights[self.hash2index[ngram_hash]]
                    ngrams_found += 1
            if word_vec.any():
                return word_vec / max(1, ngrams_found)
            else:  # No ngrams of the word are present in self.ngrams
                raise KeyError('all ngrams for word %s absent from model' % word)

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can only call `most_similar`, `similarity` etc.

        """
        super(FastTextKeyedVectors, self).init_sims(replace)
        if getattr(self, 'vectors_ngrams_norm', None) is None or replace:
            logger.info("precomputing L2-norms of ngram weight vectors")
            if replace:
                for i in range(self.vectors_ngrams.shape[0]):
                    self.vectors_ngrams[i, :] /= sqrt((self.vectors_ngrams[i, :] ** 2).sum(-1))
                self.vectors_ngrams_norm = self.vectors_ngrams
            else:
                self.vectors_ngrams_norm = \
                    (self.vectors_ngrams / sqrt((self.vectors_ngrams ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):
        """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.

        Parameters
        ----------
        fname : str
            The file path used to save the vectors in.
        fvocab : str
            Optional file path used to save the vocabulary.
        binary : bool
            If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
        total_vec :  int
            Optional parameter to explicitly specify total no. of vectors
            (in case word vectors are appended with document vectors afterwards).

        """
        # from gensim.models.word2vec import save_word2vec_format
        _save_word2vec_format(
            fname, self.vocab, self.vectors, fvocab=fvocab, binary=binary, total_vec=total_vec)
