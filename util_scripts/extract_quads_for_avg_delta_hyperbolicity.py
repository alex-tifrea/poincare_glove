import argparse
from gensim.models.keyedvectors import VanillaWordEmbeddingsKeyedVectors, Vocab
from glove_code.src.glove_inner import read_all, read_as_neighbor_lists
from numpy import array, uint32, save, median
from random import sample, randint, choice
from timeit import default_timer


PRINT_EVERY = 1
NUM_QUADS = 10
graph_map = {}


def load_vocab(wv, vocab_file, use_glove_format, restrict_vocab):
    # Read vocab.
    vocab_size = 0
    with open(vocab_file, "r") as f:
        wv.index2freq = []
        all_lines = f.readlines()[:restrict_vocab] if restrict_vocab > 0 else f.readlines()
        for index, line in enumerate(all_lines):
            if use_glove_format:
                word, count = line.strip().split(" ")  # vocab is indexed from 0; for co-occ we use 1-based indexing
                index = index
            else:
                index, word, count = line.strip().split("\t")
                index = int(index) - 1  # indexing starts at 1 in the file; for co-occ we use 0-based indexing
            wv.index2word.append(word)
            wv.vocab[word] = Vocab(index=index, count=int(count))
            wv.index2freq.append(count)
            vocab_size += 1

    wv.index2freq = array(wv.index2freq).astype(uint32)

    # Unused members from VanillaWordEmbeddingsKeyedVectors.
    wv.vectors_norm = None

    print("Loaded vocabulary with {} words".format(vocab_size))
    return vocab_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_our_format', dest='use_glove_format', action='store_false',
                        help='Use our format for reading the vocabulary and the co-occ matrix, instead of the format '
                             'from the original GloVe code.')
    parser.add_argument('--coocc_file', type=str,
                        help='Filename which contains the coocc matrix in text format.')
    parser.add_argument('--vocab_file', type=str,
                        help='Filename which contains the vocabulary.')
    parser.add_argument('--quad_file', type=str,
                        help='Filename in which to save the list of the quads that will be found.')
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--restrict_vocab', type=int, default=400000,
                        help='Only use the `restrict_vocab` most frequent words')
    parser.add_argument('--vocab_from_file', type=str, default='',
                        help='Filename from which to extract a vocabulary. Only words from this vocab will be used to'
                             'get valid quadruples.')
    parser.set_defaults(use_glove_format=True)
    args = parser.parse_args()

    wv = VanillaWordEmbeddingsKeyedVectors(0)
    vocab_size = load_vocab(
        wv,
        vocab_file=args.vocab_file,
        use_glove_format=args.use_glove_format,
        restrict_vocab=args.restrict_vocab)

    num_pairs = read_all(args.use_glove_format, args.coocc_file)
    print("Finished first traversal of corpus. Detected a total of {} pairs".format(num_pairs))

    limited_vocab = {}
    if args.vocab_from_file != '':
        with open(args.vocab_from_file, "r") as f:
            for line in f:
                if "#" in line:
                    continue

                words = line.strip().split('\t')[:2]
                if words[0] in wv.vocab:
                    limited_vocab[wv.vocab[words[0]].index] = words[0]
                if words[1] in wv.vocab:
                    limited_vocab[wv.vocab[words[1]].index] = words[1]
    print(len(limited_vocab.keys()))

    print(min(limited_vocab.keys()), max(limited_vocab.keys()), median(list(limited_vocab.keys())))

    # Load all the co-occ pairs in memory, as a map
    print("Reading the adjacency lists.")
    start = default_timer()
    graph_map = read_as_neighbor_lists(
        use_glove_format=args.use_glove_format, filename=args.coocc_file, num_pairs=num_pairs, vocab_size=vocab_size,
        limited_vocab=limited_vocab)
    print("Finished reading the adjacency lists for {} words in {:.2f}".format(
        vocab_size, default_timer() - start
    ))

    # Filter dictionary to only contain keys that have a non-void adjacency list.
    graph_map = {k: v for k, v in graph_map.items() if len(v) != 0}

    valid_quads_found = []
    overall_start, start = default_timer(), default_timer()
    while len(valid_quads_found) < NUM_QUADS:
        w1, w2, w3, w4 = choice(list(graph_map.keys())), None, None, None
        curr_quad = [w1]
        if len(graph_map[w1]) == 0:
            continue
        while not w2 or w2 in curr_quad:
            w2 = sample(graph_map[w1], 1)[0]
        curr_quad = [w1, w2]
        intersection12 = graph_map[w1].intersection(graph_map[w2])
        if len(intersection12) == 0:
            continue
        while not w3 or w3 in curr_quad:
            w3 = sample(intersection12, 1)[0]
        curr_quad = [w1, w2, w3]
        intersection123 = intersection12.intersection(graph_map[w3])
        if len(intersection123) == 0:
            continue
        while not w4 or w4 in curr_quad:
            w4 = sample(intersection123, 1)[0]
        curr_quad = [w1, w2, w3, w4]
        valid_quads_found.append(curr_quad)

        if len(valid_quads_found) % PRINT_EVERY == 0:
            print("Found {} quads in {:.2f} sec".format(len(valid_quads_found), default_timer()-start))
            start = default_timer()

    valid_quads_found = array(valid_quads_found)
    print("Found {} valid quads in a total of {:.2f} sec".format(
        len(valid_quads_found), default_timer() - overall_start))


    start = default_timer()
    save(args.quad_file, valid_quads_found)
    print("Finished writing quads to file in {:.2f} sec".format(default_timer() - start))
