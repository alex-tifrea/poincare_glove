In this README we will describe the changes that we made to train Riemannian 
skip-gram models.

We followed mostly the code structure of the gensim repository. We
only made adjusetments where necessary in order to allow for the training and
evaluation of Riemannian embeddings.

For the Poincare embeddings we use double floating point precision, to avoid
numerical errors. Because of this, some of the cython functions in
word2vec\_inner.pyx had to be rewritten.

For now, the only fully implemented optimization methods for Poincare embeddings
are RSGD with retraction (referred to as RSGD), RSGD with the exact exponantial
map (i.e. Full RSGD) and RSGD or Full RSGD where the learning rate is weighted
by a function that is inversely proportional to the frequency of the word whose
parameters are being updated (i.e. Weighted RSGD or Weighted Full RSGD). RMSprop
and AdaGrad are not completely implemented and tested yet.

Using an instance of `InitializationConfig` we have the option to initialize
embeddings from a pre-trained model.  Particularly, we use this to initialize
Poincare embeddings from a pre-trained Euclidean model that we then project 
onto the Poincare ball. We use either the identity map or the exponential map 
for the projection. We scale the embeddings such that we control how close to
the origin they are initialized.

The `WordEmbeddingCheckpoints` class is used to save snapshots of the embeddings
of some words during training, in order to be able to create animations with how
they evolve. It is usually used with 2D embeddings. The words for which we save
the snapshots need to be provided.

The file `keyedvectors.py` contains functions that perform the evaluation of the
trained model. Here we added new classes for Poincare Skip-gram and GloVe
embeddings and for the Cartesian product of Poincare balls. Note that evaluation
remains unchanged between a Poincare Skip-gram model and a Poincare GloVe model.
For each of them, we implemented the methods that compute the distance and the
appropriate parallel transport procedure. We changed the analogy evaluation to
be performed in batches, which speeds up the process significantly. All the
analogy methods prefixed by `batch_` are computed in batches.

In eval\_pretrained\_emb.py resides the code that evaluates pretrained word2vec
and GloVe embeddings.

