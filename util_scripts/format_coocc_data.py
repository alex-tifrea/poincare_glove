# OUTDATED
import sys

filename = sys.argv[1]
basename = filename.rsplit(".", 1)[0]
print(basename)

vocab = {}
with open("/Users/alext/Documents/Master/Thesis/data/wiki_coocc_data/w_freq.csv", "r") as f:
    all_lines = f.readlines()
    for line in all_lines:
        index, word, count = line.strip().split("\t")
        vocab[word] = int(index)-1  # indexing starts at 1 in the file

new_filename = basename+"_formatted.tsv"
print(new_filename)
with open(new_filename, "w") as fout:
    for line in open(filename, "r"):
        w1, w2, count = line.strip().split("\t")
        w1_index = vocab[w1]
        w2_index = vocab[w2]
        fout.write(str(w1_index) + "\t" + str(w2_index) + "\t" + count + "\n")
