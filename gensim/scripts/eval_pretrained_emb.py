import gensim
import os
import sys

GOOGLE_SIZE = 19544
MSR_SIZE = 8000

filename = sys.argv[1]
restrict_vocab = 400000  # 189533
root = "../../../"


def precision(eval_result):
    return len(eval_result['correct']) / (len(eval_result['correct']) + len(eval_result['incorrect']))


basename = os.path.basename(filename)
binary = False
if ".bin" in basename:
    binary = True
model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=binary, limit=restrict_vocab)

print('(Model, {}), (Restrict vocab, {})'.format(basename, restrict_vocab))

print("============== Similarity evaluation ==============")
pearson, spearman, ratio = model.evaluate_word_pairs(os.path.join(root, 'msc_tifreaa/gensim/test/test_data', 'rare_word.txt'))
print("Stanford Rare World: {} {} {}".format(pearson[0], spearman[0], ratio))
pearson, spearman, ratio = model.evaluate_word_pairs(os.path.join(root, 'msc_tifreaa/gensim/test/test_data', 'wordsim353.tsv'))
print("WordSim353: {} {} {}".format(pearson[0], spearman[0], ratio))
pearson, spearman, ratio = model.evaluate_word_pairs(os.path.join(root, 'msc_tifreaa/gensim/test/test_data', 'simlex999.txt'))
print("SimLex999: {} {} {}".format(pearson[0], spearman[0], ratio))

print("=========== Analogy evaluation (3COSADD) ==========")
most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.batch_most_similar_analogy
analogy_eval = model.accuracy(
    os.path.join(root, 'msc_tifreaa/gensim/test/test_data', 'questions-words.txt'),
    restrict_vocab=restrict_vocab,
    most_similar=most_similar)
print("Google: {} {} {} {}".format(analogy_eval[-1]['correct'][0],
                                   analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                   analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                   analogy_eval[-1]['correct'][0] / GOOGLE_SIZE))
analogy_eval = model.accuracy(
    os.path.join(root, 'data/MSR-analogy-test-set', 'word_relationship.processed'),
    restrict_vocab=restrict_vocab,
    most_similar=most_similar)
print("Microsoft: {} {} {} {}".format(analogy_eval[-1]['correct'][0],
                                      analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0],
                                      analogy_eval[-1]['correct'][0] / (analogy_eval[-1]['correct'][0] + analogy_eval[-1]['incorrect'][0]),
                                      analogy_eval[-1]['correct'][0] / MSR_SIZE))


print("=========== Analogy evaluation (3COSMUL) ==========")
most_similar = gensim.models.keyedvectors.VanillaWordEmbeddingsKeyedVectors.most_similar_cosmul
analogy_eval = model.accuracy(
    os.path.join(root, 'msc_tifreaa/gensim/test/test_data', 'questions-words.txt'),
    restrict_vocab=restrict_vocab,
    most_similar=most_similar)
print("Google: {} {} {} {}".format(len(analogy_eval[-1]['correct']),
                                   len(analogy_eval[-1]['correct']) + len(analogy_eval[-1]['incorrect']),
                                   precision(analogy_eval[-1]),
                                   len(analogy_eval[-1]['correct']) / GOOGLE_SIZE))
analogy_eval = model.accuracy(
    os.path.join(root, 'msc_tifreaa/gensim/test/test_data/', 'msr_word_relationship.processed'),
    restrict_vocab=restrict_vocab,
    most_similar=most_similar)
print("Microsoft: {} {} {} {}".format(len(analogy_eval[-1]['correct']),
                                      len(analogy_eval[-1]['correct']) + len(analogy_eval[-1]['incorrect']),
                                      precision(analogy_eval[-1]),
                                      len(analogy_eval[-1]['correct']) / MSR_SIZE))
