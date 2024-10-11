import pandas as pd
import argparse
import json
import parameters
import os.path
import matplotlib.pyplot as plt

### Setting variables ########################################################

parser = argparse.ArgumentParser(description="Main")
parser.add_argument(
    "--scenario",
    type=str,
    default="task_based",
    help="Other choice : task_free. Default to task_based.",
)
parser.add_argument(
    "--time_stamped",
    type=bool,
    default=False,
    help="Wether the dataset is initially ordered by a timestamp or not. Default to False.",
)
parser.add_argument(
    "--ordering",
    type=str,
    default="grad",
    help="grad for gradual drift, sudden for sudden drift, random for random ordering",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed, default to 1",
)
args = parser.parse_args()

scenario = args.scenario
ordering = args.ordering
time_stamped = args.time_stamped
seed = args.seed

bench = parameters.benchmark()
datasets = bench.datasets
models = bench.models
workdir = bench.workdir

metrics_continual = [
    "average_accuracy",
    "backward_transfer",
    "forward_transfer",
    "frugality_score",
]

metrics_consumption = [
    "duration",
    "cpu_energy",
    "ram_energy",
    "energy_consumed",
]

coord_consumption = {}
coord_time = {}


# Pour les métriques continual :
for metric in metrics_continual:
    metric_dict = {}
    for tested_model in models:
        model_perf = []
        sum = 0
        metric_count = 0
        for set_data in datasets:
            with open(
                workdir
                + "results/{}_final_continual_".format(set_data)
                + "{}_".format(tested_model)
                + "{}_".format(scenario)
                + "{}_results.json".format(ordering),
                "r",
                encoding="utf-8",
            ) as json_file:
                continual_file = json.load(json_file)
                model_perf.append(continual_file[metric])
                sum += continual_file[metric]
                metric_count += 1
        mean = sum / metric_count
        if metric == "average_accuracy":
            coord_consumption[tested_model] = [mean]
            coord_time[tested_model] = [mean]
        model_perf.append(mean)
        metric_dict[tested_model] = model_perf
    metric_df = pd.DataFrame(metric_dict)
    metric_df.insert(
        0,
        "Datasets",
        [
            "20NG",
            "Mediamill",
            "Scene",
            "Yeast",
            "Synthetic_monolab",
            "Synthetic_bilab",
            "Synthetic_rand",
            "Avg. value",
        ],
    )
    metric_df.set_index("Datasets", inplace=True)
    metric_df = metric_df.T
    with open("results/{}_table.tex".format(metric), "w", encoding="utf-8") as f:
        f.write(metric_df.to_latex(index=True))

# Pour les métriques de conso :
for metric in metrics_consumption:
    metric_dict = {}
    for tested_model in models:
        model_perf = []
        sum = 0
        metric_count = 0
        for set_data in datasets:
            consumption_df = pd.read_csv(
                workdir
                + "consumption/{}_".format(set_data)
                + "{}_".format(tested_model)
                + "{}_".format(ordering)
                + "{}_consumption.csv".format(scenario)
            )
            model_perf.append(consumption_df.loc[0][metric])
            sum += consumption_df.loc[0][metric]
            metric_count += 1
        mean = sum / metric_count
        if metric == "energy_consumed":
            coord_consumption[tested_model].append(mean)
        if metric == "duration":
            coord_time[tested_model].append(mean)
        model_perf.append(mean)
        metric_dict[tested_model] = model_perf
    metric_df = pd.DataFrame(metric_dict)
    metric_df.insert(
        0,
        "Datasets",
        [
            "20NG",
            "Mediamill",
            "Scene",
            "Yeast",
            "Synthetic_monolab",
            "Synthetic_bilab",
            "Synthetic_rand",
            "Avg. value",
        ],
    )
    metric_df.set_index("Datasets", inplace=True)
    metric_df = metric_df.T
    with open("consumption/{}_table.tex".format(metric), "w", encoding="utf-8") as f:
        f.write(metric_df.to_latex(index=True))

### Plotting the consumption against accuracy :
plt.figure(figsize=(10, 5), dpi=600)
for k, v in coord_consumption.items():
    plt.scatter(coord_consumption[k][0], coord_consumption[k][1])
    plt.annotate(k, (coord_consumption[k][0], coord_consumption[k][1]))
plt.ylabel("Energy consumption (kWh)")
plt.xlabel("Average accuracy")
plt.xlim(0.5, 1)
plt.ylim(bottom=0)
plt.title("Consommation et average accuracy moyennes des approches comparées")
plt.tight_layout()
plt.savefig(
    workdir
    + "graphs/0_Final_graph_consumption_{}_".format(ordering)
    + "{}".format(scenario),
    bbox_inches="tight",
)
plt.close()

### Plotting the duration against accuracy :
plt.figure(figsize=(10, 5), dpi=600)
for k, v in coord_time.items():
    plt.scatter(coord_time[k][0], coord_time[k][1])
    plt.annotate(k, (coord_time[k][0], coord_time[k][1]))
plt.ylabel("Experimentation duration (s)")
plt.xlabel("Average accuracy")
plt.xlim(0.5, 1)
plt.ylim(bottom=0)
plt.title("Durées et average accuracy moyennes des approches comparées")
plt.tight_layout()
plt.savefig(
    workdir
    + "graphs/0_Final_graph_duration_{}_".format(ordering)
    + "{}".format(scenario),
    bbox_inches="tight",
)
plt.close()
