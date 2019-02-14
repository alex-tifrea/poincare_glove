This is where most of the prototyping and plotting takes place.

target\_context\_word\_embeddings\_2D\_plots{'', \_GLOVE, \_MIX-GLOVE}:
  - contains plots with all the embeddings from a fixed vocabulary and with
  target and context embeddings visualized in the same plot. The vocabulary for
  the plots is all the unique tokens that appear in the Google analogy dataset.

analysis\_of\_init\_from\_pretrained\_word2vec:
  - plots that show the impact of the learning rate and of the scaling factor
  when initializing word2vec Poincare embeddings from pretrained Euclidean
  embeddings that are projected in the Poincare ball. The scaling factor refers
  to how far from the origin these projected Euclidean embeddings can be placed.

baselines\_checks:
  - plots that analyze vanilla word2vec vs Euclidean distance-based word2vec
  trained with biases.

leimeister\_paper\_analysis:
  - analysis of https://arxiv.org/abs/1809.01498. Plot of the embeddings of the
  Google analogy dataset vocabulary.

extract\_google\_analogy\_dataset\_vocab:
  - functions that extract the vocabulary of the Google analogy dataset in a
  file. The vocabulary can b saved either unlabled (just the unique words) or
  labeled with the section in which each word appears (a random section is
  selected if a word appears in more than one section).

training\_2D\_animation\_GLOVE:
  - contains animations of how a number of (target) word embeddings change
  during training. It contains animations for both the vanilla and the Poincare
  embeddings. Each frame is a checkpoint of the values for the embeddings
  of these words, dumped to file during training. For details of how to run the
  training script so that it saves these checkpoints, see the main README.

training\_2D\_animation\_and\_tSNE\_word2vec:
  - contains animations of training word2vec embeddings (see description of the
    above). Also has tSNE plots of 600D embeddings (vanilla and Poincare).

cherry\_picked\_hierarchies\_2D\_plots{'', MIX-GLOVE}:
  - 2D plots of some cherry picked hierarchies, for various models.

wordnet\_levels\_colored\_2D{'', MIX-GLOVE}:
  - plots of WordNet levels colored so as to show how they occupy different
  parts of the embedding space.

poincare\_checks:
  - some sanity checks and numerical checks (using autograd) to make sure that
  the gradients are correct.

supervised\_hyperlex\_eval:
  - some attempts to learn simple regressors with the embedding of a word pair
  as input (e.g. w1+w2, w1-w2 etc) and the entailment score as output (see
  HyperLex paper section 7.3 for more details on this)

hyperlex\_and\_wbless\_eval\_MIX-GLOVE:
  - HyperLex and WBLESS prototyping for evaluation functions (see
  util\_scripts/lexical\_entailment\_eval.py for the cleaned-up version that
  contains all hypernymy score functions). It also contains analysis and plots
  related to hypernymy for the Cartesian product of Poincare embeddings.

hyperlex\_analysis\_word2vec (outdated):
  - prototyping for hypernymy functions for word2vec.

poincare\_embeddings\_analysis\_{'', GLOVE}:
  - analysis of embedding norms, nearest neighbors, avg relative contrast etc.
