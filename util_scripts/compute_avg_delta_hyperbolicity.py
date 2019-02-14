import argparse
from gensim.models.keyedvectors import VanillaWordEmbeddingsKeyedVectors, Vocab
from glove_code.src.glove_inner import read_all
from numpy import array, uint32, load, sort, log, sqrt, arccosh, exp
from timeit import default_timer


PRINT_EVERY = 1000000
graph_map = {}
max_coocc_count = 0.0
USE_PROBS = False


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=dist and g(coocc_count)=log
# coocc can be either coocc_count or coocc_probability
def h_id_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return -log(coocc)
    else:
        return log(max_coocc_count / coocc)


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=-dist^2 and g(coocc)=log
# coocc can be either coocc_count or coocc_probability
def h_sq_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return sqrt(-log(coocc))
    else:
        return sqrt(log(max_coocc_count / coocc))


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=-cosh(dist) and g(coocc)=log
# coocc can be either coocc_count or coocc_probability
def h_cosh_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return arccosh(1-log(coocc))
    else:
        return arccosh(1-log(max_coocc_count / coocc))


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=-cosh(dist)^2 and g(coocc)=log
# coocc can be either coocc_count or coocc_probability
def h_cosh_sq_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return arccosh(1+sqrt(-log(coocc)))
    else:
        return arccosh(1+sqrt(log(max_coocc_count / coocc)))


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=-log(dist^2 + 1) and g(coocc)=log
# coocc can be either coocc_count or coocc_probability
def h_log_of_sq_minus_one_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return sqrt(exp(-log(coocc)) - 1)
    else:
        return sqrt(exp(log(max_coocc_count / coocc)) - 1)


# Function h on which we will compute avg delta hyp that corresponds to f(dist)=-log(dist^2) and g(coocc)=log
# coocc can be either coocc_count or coocc_probability
def h_log_of_sq_log(coocc):
    global max_coocc_count, USE_PROBS
    if USE_PROBS:
        return sqrt(exp(-log(coocc)))
    else:
        return sqrt(exp(log(max_coocc_count / coocc)))


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
    parser.add_argument('--vocab_file', type=str,
                        help='Filename which contains the vocabulary.')
    parser.add_argument('--coocc_file', type=str,
                        help='Filename which contains the coocc pairs.')
    parser.add_argument('--quad_file', type=str,
                        help='Filename from which to load the list of quads.')
    parser.add_argument('--root', type=str,
                        default='~/Documents/Master/Thesis',
                        help='Path to the root folder that contains msc_tifreaa, data etc.')
    parser.add_argument('--restrict_vocab', type=int, default=400000,
                        help='Only use the `restrict_vocab` most frequent words')
    parser.add_argument('--use_probs', dest='use_probs', action='store_true',
                        help='Use log-probabilities log(P_ij) as edge weights instead of log-counts log(X_ij)')
    parser.set_defaults(use_glove_format=True, use_probs=False)
    args = parser.parse_args()

    USE_PROBS = args.use_probs

    wv = VanillaWordEmbeddingsKeyedVectors(0)
    vocab_size = load_vocab(
        wv,
        vocab_file=args.vocab_file,
        use_glove_format=args.use_glove_format,
        restrict_vocab=args.restrict_vocab)

    # Load all the co-occ pairs in memory, as a map
    print("Reading the co-occ pairs.")
    start = default_timer()
    num_pairs, graph_map, max_coocc_count = read_all(
        use_glove_format=args.use_glove_format, filename=args.coocc_file, return_pairs=True)
    print("Finished reading {} co-occ pairs in {:.2f}. Max co-occ count = {:.4f}".format(
        num_pairs, default_timer() - start, max_coocc_count
    ))

    quad_list = load(args.quad_file)

    functions = {
        "h_id_log": h_id_log,
        "h_sq_log": h_sq_log,
        "h_cosh_log": h_cosh_log,
        "h_cosh_sq_log": h_cosh_sq_log,
        "h_log_of_sq_minus_one_log": h_log_of_sq_minus_one_log,
        "h_log_of_sq_log": h_log_of_sq_log,
    }

    for func_name, func in functions.items():
        d_avg, count_d_avg = 0.0, 0
        delta_avg, count_delta_avg = 0.0, 0
        overall_start, start = default_timer(), default_timer()
        for quad in quad_list:
            x, y, v, w = sort(quad)
            # Extract pairwise distances.
            sums = []
            if USE_PROBS:
                sums.append(func(graph_map[str(x)+" "+str(y)] / wv.index2freq[x-1] / wv.index2freq[y-1]) + func(graph_map[str(v)+" "+str(w)] / wv.index2freq[v-1] / wv.index2freq[w-1]))
                sums.append(func(graph_map[str(x)+" "+str(v)] / wv.index2freq[x-1] / wv.index2freq[v-1]) + func(graph_map[str(y)+" "+str(w)] / wv.index2freq[y-1] / wv.index2freq[w-1]))
                sums.append(func(graph_map[str(x)+" "+str(w)] / wv.index2freq[x-1] / wv.index2freq[w-1]) + func(graph_map[str(y)+" "+str(v)] / wv.index2freq[y-1] / wv.index2freq[v-1]))
            else:
                sums.append(func(graph_map[str(x)+" "+str(y)]) + func(graph_map[str(v)+" "+str(w)]))
                sums.append(func(graph_map[str(x)+" "+str(v)]) + func(graph_map[str(y)+" "+str(w)]))
                sums.append(func(graph_map[str(x)+" "+str(w)]) + func(graph_map[str(y)+" "+str(v)]))
            sums = sorted(sums)

            d_avg += sums[0] + sums[1] + sums[2]
            delta_avg += float(sums[2] - sums[1]) / 2

            count_d_avg += 6
            count_delta_avg += 1

            if count_delta_avg % PRINT_EVERY == 0:
                print("[func={}] Processed {} quads in {:.2f} sec. avg_d={:.4f}, avg_delta={:.4f}".format(func_name,
                      count_delta_avg, default_timer() - start, d_avg / count_d_avg, delta_avg / count_delta_avg))
                start = default_timer()

        print("[func={}] Finished processing {} quads. Took {:.2f} sec".format(func_name, count_delta_avg,
                                                                               default_timer() - overall_start))

        d_avg = d_avg / count_d_avg
        delta_avg = delta_avg / count_delta_avg
        print("[func={}] d_avg = {:.4f}".format(func_name, d_avg))
        print("[func={}] delta_avg = {:.4f}".format(func_name, delta_avg))
        print("[func={}] 2 * delta_avg / d_avg = {:.4f}".format(func_name, 2 * delta_avg / d_avg))
