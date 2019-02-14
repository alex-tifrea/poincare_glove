Implementation of classes that interface the GloVe model. Also contains low
level cython implementation of the GloVe loss gradients used to perform the
updates.

**Co-occurrence + vocabulary data formats**:
We use two data formats. First of all, the format for the co-occurrence matrix
and the vocabulary as it is output by the public GloVe code (referred to as
'the GloVe format') is like this:
  For the vocabulary:
    - (word, frequency\_in\_corpus)
  For the co-occurrence matrix:
    - (word1\_index: int, word2\_index: int, coocc\_count: double)

Besides this, we use another format (referred to as 'our format'). This is the
as follows:
  For the vocabulary:
    - (index, word, frequency\_in\_corpus)
  For the co-occurrence matrix:
    - (word1\_index: int, word2\_index: int, coocc\_count: int)


vfast\_invsqrt.h:
  - contains fast approximation of 1/sqrt(x) (see
      https://en.wikipedia.org/wiki/Fast_inverse_square_root)

glove\_inner.pyx:
  - cython implementation of GloVe updates and GloVe loss gradients, when the 
  dot product, the Euclidean distance or the Poincare distance have been used in
  the loss formula. The file follows the same pattern as in word2vec\_inner.pyx
  from the gensim repo.

glove.py:
  - contains the class that represents one GloVe model (containing the
  parameters, the vocabulary etc). Calls the fast cython functions to train the
  model.

**Evaluation**:
The file `keyedvectors.py` contains functions that perform the evaluation of the
trained model. Here we added new classes for Poincare Skip-gram and GloVe
embeddings and for the Cartesian product of Poincare balls. Note that evaluation
remains unchanged between a Poincare Skip-gram model and a Poincare GloVe model.
We implemented the methods that compute the distance and the
appropriate parallel transport procedure. The analogy evaluation has been
changed so that it is performed in batches, which speeds up the process
significantly. All the analogy methods prefixed by `batch_` are computed in
batches.

