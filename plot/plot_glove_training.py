#!/usr/local/bin/python3

import matplotlib
import numpy as np
import os
import sys

if matplotlib.get_backend() != "MacOSX" and os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
argv[1] = one log file and it will plot the evolution of the word similarity scores
argv[2] = the folder in which the output image will be saved; if argv[2] is missing, then the figure will not be saved
          to file
"""

log_file = sys.argv[1]
fig_dir = sys.argv[2] if len(sys.argv) > 2 else None
fig_name = log_file.split('/')[-1].replace("log_", "fig_") + ".png"

with open(log_file, "r") as f:
    sim_scores = {
        "rareword": [],
        "wordsim": [],
        "simlex": []
    }
    analogy_scores = {
        "epoch_number": [],
        "google": [],
        "msr": []
    }
    epoch_loss = []
    norms = {
        "target_top10": [],
        "target_outside_top1000": [],
        "context_top10": [],
        "context_outside_top1000": []
    }
    lines = f.readlines()
    for line in lines:
        if "Similarity:" in line:
            split_line = [s.strip(" ,") for s in line.strip().split(":", 1)[1].split(" ")]
            sim_scores["rareword"].append(float(split_line[2]))
            sim_scores["wordsim"].append(float(split_line[4]))
            sim_scores["simlex"].append(float(split_line[6]))
            norms["target_top10"].append(float(split_line[8]))
            norms["target_outside_top1000"].append(float(split_line[12]))
            norms["context_top10"].append(float(split_line[10]))
            norms["context_outside_top1000"].append(float(split_line[14]))
        elif "Analogy:" in line:
            split_line = [s.strip(" ,") for s in line.strip().split(" ")]
            analogy_scores["epoch_number"].append(int(split_line[2]))
            analogy_scores["google"].append(float(split_line[5]))
            analogy_scores["msr"].append(float(split_line[7]))
        elif "EPOCH - " in line:
            epoch_loss.append(float(line.strip().rsplit(" ", 1)[1]))

    # Convert to numpy arrays.
    sim_scores["rareword"] = np.array(sim_scores["rareword"])
    sim_scores["wordsim"] = np.array(sim_scores["wordsim"])
    sim_scores["simlex"] = np.array(sim_scores["simlex"])
    norms["target_top10"] = np.array(norms["target_top10"])
    norms["target_outside_top1000"] = np.array(norms["target_outside_top1000"])
    norms["context_top10"] = np.array(norms["context_top10"])
    norms["context_outside_top1000"] = np.array(norms["context_outside_top1000"])
    analogy_scores["epoch_number"] = np.array(analogy_scores["epoch_number"])
    analogy_scores["google"] = np.array(analogy_scores["google"])
    analogy_scores["msr"] = np.array(analogy_scores["msr"])
    epoch_loss = np.array(epoch_loss)

    # Plotting similarity scores and norms.
    fig = plt.figure("Glove training", figsize=(15, 8))
    x = range(sim_scores["rareword"].shape[0])

    # Plot evolution of similarity scores.
    plt.subplot(221)
    l1, = plt.plot(x, sim_scores["rareword"], color="orange")
    l2, = plt.plot(x, sim_scores["wordsim"], color="green")
    l3, = plt.plot(x, sim_scores["simlex"], color="red")
    plt.ylabel("Spearman correlation")
    plt.title("Evolution of similarity scores during training")
    plt.legend((l1, l2, l3), ("RareWord", "WordSim", "SimLex"))

    # Plot evolution of analogy scores.
    plt.subplot(222)
    analogy_x = analogy_scores["epoch_number"]
    l1, = plt.plot(analogy_x, analogy_scores["google"], color="orange")
    l2, = plt.plot(analogy_x, analogy_scores["msr"], color="green")
    plt.ylabel("Analogy accuracy")
    plt.title("Evolution of analogy scores during training")
    plt.legend((l1, l2), ("Google", "MSR"))

    plt.subplot(223)
    l1, = plt.plot(x, norms["target_top10"], color="orange")
    l2, = plt.plot(x, norms["target_outside_top1000"], color="red")
    l3, = plt.plot(x, norms["context_top10"], color="green")
    l4, = plt.plot(x, norms["context_outside_top1000"], color="blue")
    plt.ylabel("Vector norms")
    plt.title("Evolution of vector norms during training")
    plt.legend((l1, l2, l3, l4), ("Target-Top10", "Target-Outside Top1k", "Context-Top10", "Context-Outside Top1k"))

    plt.subplot(224)
    l1, = plt.plot(x, epoch_loss, color="red")
    plt.ylabel("Epoch loss")
    plt.title("Evolution of the epoch loss during training")

    plt.tight_layout()

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, fig_name))

    plt.show()

