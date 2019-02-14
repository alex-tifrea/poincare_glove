#!/usr/local/bin/python3

import matplotlib
import numpy as np
import os
import sys

if matplotlib.get_backend() != "MacOSX" and os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
argv[1] = one train log file and it will plot the info about the loss and the word similarity
          scores after each epoch
argv[2] = the folder in which the output image will be saved; if argv[2] is missing, then the figure will not be saved
          to file
"""

train_log_file = sys.argv[1]
fig_dir = sys.argv[2] if len(sys.argv) > 2 else None
fig_name = train_log_file.split('/')[-1].replace("train_", "fig_train_loss_")

with open(train_log_file, "r") as f:
    epoch_end_scores = {
        "loss": [],
        "rareword": [],
        "wordsim": [],
        "simlex": []
    }
    lines = f.readlines()
    for line in lines:
        if "EPOCH END" in line:
            split_line = line.split(",")
            epoch_end_scores["loss"].append(float(split_line[0].split(" ")[7]))
            epoch_end_scores["rareword"].append(float(split_line[1].split(" ")[2]))
            epoch_end_scores["wordsim"].append(float(split_line[2].split(" ")[2]))
            epoch_end_scores["simlex"].append(float(split_line[3].split(" ")[2]))

    # Convert to numpy arrays.
    epoch_end_scores["loss"] = np.array(epoch_end_scores["loss"])
    epoch_end_scores["rareword"] = np.array(epoch_end_scores["rareword"])
    epoch_end_scores["wordsim"] = np.array(epoch_end_scores["wordsim"])
    epoch_end_scores["simlex"] = np.array(epoch_end_scores["simlex"])

    # Plotting.
    fig = plt.figure(1)

    x = range(1, len(epoch_end_scores["loss"])+1)
    # Plot end-of-epoch losses.
    plt.subplot(211)
    plt.plot(x, epoch_end_scores["loss"], color="blue", label="Loss")
    plt.xticks(x)
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title("End of epoch loss")

    # Plot end-of-epoch similarity scores.
    plt.subplot(212)
    l1, = plt.plot(x, epoch_end_scores["rareword"], color="orange")
    l2, = plt.plot(x, epoch_end_scores["wordsim"], color="green")
    l3, = plt.plot(x, epoch_end_scores["simlex"], color="red")
    plt.xticks(x)
    plt.xlabel("Epochs")
    plt.ylabel("Spearman correlation")
    plt.title("End of epoch similarity scores")
    plt.legend((l1, l2, l3), ("RareWord", "WordSim", "SimLex"), loc="upper right")

    plt.tight_layout()

    if fig_dir:
        fig.savefig(os.path.join(fig_dir, fig_name))

    plt.show()

