from glove_code.src.glove_inner import extract_restrict_vocab_pairs
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

restrict_vocab = 50000
num_pairs = extract_restrict_vocab_pairs(in_file, out_file, restrict_vocab=restrict_vocab)
print(num_pairs, "for a vocab of", restrict_vocab, "words")

# basename = filename.rsplit(".", 1)[0]
# output = basename+"_vocab"+str(restrict_vocab)+".bin"
# print(output)
#
# num_pairs = read_all(use_glove_format=True, filename=output)
# print(num_pairs)