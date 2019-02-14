import glob
import os
import sys

model_path_pattern = sys.argv[1]
root = sys.argv[2]


def extract_scores_from_file(file):
    scores = [-1] * 13
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if ":" not in line:
                continue
            line = line.strip().split(":")
            info = line[1].strip().split(" ")
            if "Rare World" in line[0]:
                scores[0] = info[1]
            elif "WordSim" in line[0]:
                scores[1] = info[1]
            elif "SimLex" in line[0]:
                scores[2] = info[1]
            elif "MTurk" in line[0]:
                scores[3] = info[1]
            elif "SimVerb" in line[0]:
                scores[4] = info[1]
            elif "MEN" in line[0]:
                scores[5] = info[1]
            elif "MC" in line[0]:
                scores[6] = info[1]
            elif "RG" in line[0]:
                scores[7] = info[1]
            elif "YP" in line[0]:
                scores[8] = info[1]

            elif "Semantic Google" in line[0]:
                scores[9] = info[4]
            elif "Syntactic Google" in line[0]:
                scores[10] = info[4]
            elif "Google" in line[0]:
                scores[11] = info[4]
            elif "Microsoft" in line[0]:
                scores[12] = info[4]

    return scores


for model_file in list(glob.glob(model_path_pattern)):
    basename = os.path.basename(model_file)
    print(basename)

    eval_log_file = "eval_" + basename.split("_", 1)[1]
    eval_log_file = os.path.join(root, "eval_logs", eval_log_file)

    if "NUMEMBS" in basename:
        eval_log_file += "_mix-hyp_pt"
    else:
        eval_log_file += "_hyp_pt"
    print(eval_log_file)

    # Hyperbolic distance, W
    hyp_w_scores = extract_scores_from_file(eval_log_file)

    # Hyperbolic distance, W+C
    hyp_wc_scores = extract_scores_from_file(eval_log_file + "_agg")

    # Euclidean cosine distance, W
    cos_w_scores = extract_scores_from_file(eval_log_file + "-eucl-cos-dist")

    # Euclidean cosine distance, W
    cos_wc_scores = extract_scores_from_file(eval_log_file + "-eucl-cos-dist_agg")

    print("\t".join([str(w) + " / " + str(wc) for w, wc in zip(hyp_w_scores, hyp_wc_scores)]))
    print("\t".join([str(w) + " / " + str(wc) for w, wc in zip(cos_w_scores, cos_wc_scores)]))

    print()

