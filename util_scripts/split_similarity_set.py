import random
import sys

init_file = sys.argv[1]
validation_file = sys.argv[2]
test_file = sys.argv[3]

VALIDATION_RATIO = 0.5

with open(init_file, "r") as fin, open(validation_file, "w") as fv, open(test_file, "w") as ft:
    lines = fin.readlines()
    num_validation = int(VALIDATION_RATIO * len(lines))
    valid_idxs = random.sample(range(len(lines)), num_validation)
    valid_lines = [lines[idx] for idx in valid_idxs]

    for l in valid_lines:
        fv.write(l)

    for i in range(len(lines)):
        if i not in valid_idxs:
            ft.write(lines[i])
