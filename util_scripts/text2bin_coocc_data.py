from glove_code.src.glove_inner import write_all, read_all
import sys

filename = sys.argv[1]

restrict_vocab = 200000
write_all(filename, restrict_vocab=restrict_vocab)

# basename = filename.rsplit(".", 1)[0]
# output = basename+"_vocab"+str(restrict_vocab)+".bin"
# print(output)
#
# num_pairs = read_all(use_glove_format=True, filename=output)
# print(num_pairs)