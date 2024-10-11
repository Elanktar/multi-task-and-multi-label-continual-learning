import argparse
import os.path
import parameters
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="1 if task_free scenario, default to 0",
)
parser.add_argument(
    "--nb_clusters",
    type=int,
    default=4,
    help="Number of clusters, default to 4",
)
parser.add_argument(
    "--ordering",
    type=str,
    default="grad",
    help="grad for gradual drift, sudden for sudden drift, random for random order",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="20NG",
    help="20NG, Bibtex, Birds, Bookmarks, Chess, Cooking, Corel16k001, Enron, Eukaryote, Human, Imdb, Mediamill, Ohsumed, Reuters-K500, Scene, Slashdot, tmc2007-500, Water-quality, Yeast, Yelp",
)
args = parser.parse_args()

scenario = args.scenario
ordering = args.ordering
data = args.dataset
nb_clusters = args.nb_clusters

bench = parameters.benchmark()
datasets = bench.datasets
models = bench.models
workdir = bench.workdir

color_list = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Teal
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
    "#c49c94",  # Light Brown
    "#f7b6d2",  # Light Pink
    "#c7c7c7",  # Light Gray
    "#dbdb8d",  # Light Yellow-Green
    "#9edae5",  # Light Teal
]

style_list = [
    "-",
    "--",
    "-.",
    ":",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
    "-",
    "--",
    "-.",
    ":",
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]


### Functions ###################################################################


def cluster_count(nb_clusters: int, workdir: str, data: str):
    """Count the real number of clusters that were created for this dataset.

    Args:
        nb_clusters (int): number of clusters that was used to generate tasks clusters

    Returns:
        real_nb_cluster (int): real number of clusters
    """
    real_nb_cluster = 0
    for i in range(nb_clusters):
        if os.path.isfile(
            workdir + "Eval_set/{}".format(data) + "_{}".format(i) + "_eval_set.csv"
        ):
            real_nb_cluster += 1
    return real_nb_cluster


def get_results(
    scenario: str, workdir: str, data: str, m: str, ordering: str, results: dict
):
    """Get the online metrics results.

    Args:
        scenario (str): scenario used
        workdir (str): working directory
        data (str): name of the dataset
        m (str): name of the model
        ordering (str): name of the stream ordering
        results (dict): dict that will contain the results

    Returns:
        step (list): list containing the results for the given parameters
    """
    if os.path.isfile(
        workdir
        + "results/{}_".format(data)
        + "{}_".format(m)
        + "{}_continual_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    ):
        with open(
            workdir
            + "results/{}_".format(data)
            + "{}_".format(m)
            + "{}_continual_".format(scenario)
            + "{}_results".format(ordering)
            + ".json",
            "r",
            encoding="utf-8",
        ) as json_file:
            continual_results = json.load(json_file)
        return continual_results


def get_ordering(ordering: str, workdir: str, data: str):
    """Get the ordering of the clusters in the stream according to the used ordering.

    Args:
        ordering (str): ordering used

    Returns:
        cluster_order (dict) : a dict containing the order of the clusters
    """
    if ordering == "grad":
        assert os.path.isfile(
            workdir + "Orders/{}_grad.json".format(data)
        ), "ordering not found in datasets directory"
        with open(
            workdir + "Orders/{}_grad.json".format(data), "r", encoding="utf-8"
        ) as json_file:
            cluster_order = json.load(json_file)
    elif ordering == "sudden":
        assert os.path.isfile(
            workdir + "Orders/{}_sudden.json".format(data)
        ), "ordering not found in datasets directory"
        with open(
            workdir + "Orders/{}_sudden.json".format(data), "r", encoding="utf-8"
        ) as json_file:
            cluster_order = json.load(json_file)
    elif ordering == "random":
        assert os.path.isfile(
            workdir + "Orders/{}_random_task_based.json".format(data)
        ), "ordering not found in datasets directory"
        with open(
            workdir + "Orders/{}_random_task_based.json".format(data),
            "r",
            encoding="utf-8",
        ) as json_file:
            cluster_order = json.load(json_file)
    return cluster_order


def get_average_accuracy(final_results: dict, continual_matrix: np.array, order):
    seen_task = []
    instant_average_accuracy = []

    for i in range(continual_matrix.shape[0]):
        if order["{}".format(i)] not in seen_task:
            seen_task.append(order["{}".format(i)])
        sum = 0
        acc_count = 0
        for j in range(continual_matrix.shape[1]):
            if j in seen_task:
                sum += continual_matrix[i, j]
                acc_count += 1
        if acc_count != 0:
            instant_average_accuracy.append(sum / acc_count)
        else:
            instant_average_accuracy.append("NaN")

    final_results["instant_average_accuracy"] = instant_average_accuracy

    sum = 0
    acc_count = 0
    for value in instant_average_accuracy:
        if isinstance(value, float) or isinstance(value, int):
            sum += value
            acc_count += 1

    if acc_count != 0:
        final_results["average_accuracy"] = sum / acc_count
    else:
        final_results["average_accuracy"] = "NaN"

    return instant_average_accuracy


def get_backward_transfer(final_results: dict, continual_matrix: np.array, order):
    seen_task = []
    instant_backward_transfer = []

    for i in range(continual_matrix.shape[0]):
        sum = 0
        acc_count = 0
        for j in range(continual_matrix.shape[1]):
            if j in seen_task:
                sum += continual_matrix[i, j] - continual_matrix[i - 1, j]
                acc_count += 1
        if order["{}".format(i)] not in seen_task:
            seen_task.append(order["{}".format(i)])
        if acc_count != 0:
            instant_backward_transfer.append(sum / acc_count)
        else:
            instant_backward_transfer.append("NaN")

    final_results["instant_backward_transfer"] = instant_backward_transfer

    sum = 0
    acc_count = 0
    for value in instant_backward_transfer:
        if isinstance(value, float) or isinstance(value, int):
            sum += value
            acc_count += 1

    if acc_count != 0:
        final_results["backward_transfer"] = sum / acc_count
    else:
        final_results["backward_transfer"] = "NaN"


def get_forward_transfer(final_results: dict, continual_matrix: np.array, order):
    seen_task = []
    instant_forward_transfer = []

    for i in range(continual_matrix.shape[0]):
        if order["{}".format(i)] not in seen_task:
            seen_task.append(order["{}".format(i)])
        sum = 0
        acc_count = 0
        for j in range(continual_matrix.shape[1]):
            if j not in seen_task:
                sum += continual_matrix[i, j] - continual_matrix[i - 1, j]
                acc_count += 1
        if acc_count != 0:
            instant_forward_transfer.append(sum / acc_count)
        else:
            instant_forward_transfer.append("NaN")

    final_results["instant_forward_transfer"] = instant_forward_transfer

    sum = 0
    acc_count = 0
    for value in instant_forward_transfer:
        if isinstance(value, float) or isinstance(value, int):
            sum += value
            acc_count += 1

    if acc_count != 0:
        final_results["forward_transfer"] = sum / acc_count
    else:
        final_results["forward_transfer"] = "NaN"


def get_frugality_score(final_results: dict, consumption):
    final_results["frugality_score"] = final_results["average_accuracy"] - (
        1 / (1 + (1 / consumption["energy_consumed"][0]))
    )


###########################################################################################
############### Main ######################################################################
###########################################################################################

nb_cluster = cluster_count(nb_clusters, workdir, data)
for m in models:
    results = dict()
    continual_results = get_results(scenario, workdir, data, m, ordering, results)

    ### Preparing the numpy array matrix :
    continual_metrics_dict = dict()
    for i in range(nb_cluster):
        continual_metrics_dict[i] = []
    for k_cl, v_cl in continual_results.items():
        for i in range(nb_cluster):
            if int(re.search(r"\d+", k_cl).group()) == i:
                if v_cl != []:
                    continual_metrics_dict[i].append(v_cl[0])
    for k, v in continual_metrics_dict.items():
        if v == []:
            continual_metrics_dict[k] = [0]

    continual_matrix = pd.DataFrame.from_dict(continual_metrics_dict)
    cluster_order = get_ordering(ordering, workdir, data)
    continual_matrix = continual_matrix.to_numpy()

    consumption = pd.read_csv(
        "consumption/{}".format(data)
        + "_{}_".format(m)
        + "{}_".format(ordering)
        + "{}".format(scenario)
        + "_consumption.csv"
    )

    ### Computing the continual metrics :
    final_results = dict()
    # final_results["accuracy_matrix"] = continual_matrix
    get_average_accuracy(final_results, continual_matrix, cluster_order)
    get_backward_transfer(final_results, continual_matrix, cluster_order)
    get_forward_transfer(final_results, continual_matrix, cluster_order)
    get_frugality_score(final_results, consumption)

    output_file = (
        workdir
        + "results/{}_".format(data)
        + "final_continual_{}_".format(m)
        + "{}_".format(scenario)
        + "{}_results".format(ordering)
        + ".json"
    )
    with open(output_file, "w") as outfile:
        json.dump(final_results, outfile)

    ### Plotting the evaluation set accuracy over time :
    plt.figure(figsize=(10, 5), dpi=600)
    for k, v in cluster_order.items():
        plt.axvline(x=int(k), color="black", linestyle="-", linewidth=2)
        trans = plt.gca().get_xaxis_transform()
        plt.text(
            int(k) + 0.05,
            0.005,
            "Exp tâche {}".format(v),
            rotation=0,
            transform=trans,
        )
    plt.axvline(x=len(cluster_order), color="black", linestyle="-", linewidth=2)

    for i in range(nb_cluster):
        values = continual_matrix[:, i]
        values = np.append(0, values)
        plt.plot(
            values,
            color=color_list[i],
            linestyle=style_list[i],
            label="Tâche {}".format(i),
            alpha=0.7,
        )
    for i in range(nb_cluster):
        for j in range(continual_matrix.shape[0]):
            plt.scatter(
                j + 1,
                continual_matrix[j, i],
                color=color_list[i],
                alpha=0.7,
            )
    plt.ylabel("Macro-averaged Balanced Accuracy")
    plt.xlabel("Evaluation continue n°")
    plt.xlim(0, (nb_cluster * 2) + 0.05)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        ncol=1,
        fancybox=True,
        shadow=True,
    )
    plt.title(
        "macro_BA de {}".format(m)
        + " sur l'ensemble d'évaluation de {} au cours du temps".format(data)
    )
    plt.tight_layout()
    plt.savefig(
        workdir
        + "graphs/{}_".format(data)
        + "{}_continual_".format(m)
        + "{}_".format(ordering)
        + "{}_".format(scenario),
        bbox_inches="tight",
    )
    plt.close()
