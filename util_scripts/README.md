Description of the bash and python scripts in this folder. They are mostly used
for (pre)processing data, evaluation, starting batch jobs etc.

compute_avg_delta_hyperbolicity.py:
  - takes as input a file with a vocabulary, a file with coocc. counts, and a
  file with quadruples that satisfy the condition that any two words among them
  have a non-zero co-occurrence count. It offers the possibility to use
  log-coocc probabilities log(Pij) instead of log-coocc counts log(Xij) for
  computing the avg delta hyperbolicity and the ratio delta_avg / d_avg

extract_quads_for_avg_delta_hyperbolicity.py:
  - extracts the quadruples that will be used to compute the avg delta
  hyperbolicity. Offers the possibility to use a limited vocabulary from a file
  given as parameter. Thus it makes it possible to compute the delta
  hyperbolicity of a certain smaller datase (e.g. one of the similarity
  benchmarks)

run_all_eval_leonhard.sh:
  - takes as argument one path (accepts globbing). It will start jobs to 
  perform full evaluation of similarity and analogy evaluation on Leonhard. It
  evaluates models with w and w+c with Poincare distance and cosine distance.

format_eval_logs.py:
  - formats the evaluation logs so that they can be easily copy-pasted as a row
  in the Google spreadsheet with results.

get_model_eval_and_stats.py (partly outdated):
  - performs some (mostly outdated) hypernymy evaluation and computes various
    metrics for a given model e.g. avg relative contrast, correlation between
    embedding norms and WordNet level etc.

extract_coocc_pairs_restrict_vocab.py:
  - extract in a separate file only the co-occurrence pairs for words that come
  from a restricted vocabulary.

nickel_transitive_closure.py:
  - extract noun transitive closure from WordNet (from Nickel repository).

lexical_entailment_eval.py:
  - contains all the entailment score alternatives that were tried for the
  Cartesian product Poincare GloVe. It evaluates on both HyperLex and WBLESS.

run_hypernymy_eval.sh:
  - run all the hypernymy evaluation settings for Cartesian product Poincare
  Glove.

eval_w2v_initialization_experiments.sh:
  - evaluates a batch of experiments that were performed with different
  initialization schemes. For these experiments we trained Poincare word2vec
  using as initialization pretrained Euclidean embeddings that were projected
  (either using the exponential map or a retraction) and scaled differently, to
  be further or closer to the origin.
