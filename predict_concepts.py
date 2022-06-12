import argparse
import math
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, roc_auc_score
from pylab import cm

plt.rcParams.update({"font.size": 25})


def load_dataset(dir_training, dir_validation):
    """Loads the dataset of RNN and concepts

    Args:
        dir_training (str): directory containing training data
        dir_validation (str): directory containing training data

    Returns:
        dict , dict: data for training and validation respectively 
    """
    data_training = {}
    data_validation = {}

    data_training["rnn"] = pd.read_pickle(os.path.join(dir_training, "rnn.pkl"))
    data_training["concepts"] = pd.read_pickle(
        os.path.join(dir_training, "metadata.pkl")
    )
    
    data_validation["rnn"] = pd.read_pickle(os.path.join(dir_validation, "rnn.pkl"))
    data_validation["concepts"] = pd.read_pickle(
        os.path.join(dir_validation, "metadata.pkl")
    )

    return data_training, data_validation


def get_concept_groups(config_name):
    """Returns concept groups investigated in this work

    Args:
        config_name (str): config to check if task is pointnav or objectnav

    Returns:
        dict: dictionary containing list of concept for each group
    """
    concept_groups = {}

    concept_groups["reachability"] = []
    rotate_step_degrees = 30
    num_rotation_angles = math.ceil(360.0 / rotate_step_degrees)
    reachability_radii = [
        2
    ]  # we also tried 4,6 times the gridSize. It leads to similar results

    for radius in reachability_radii:
        for rotation_angle in range(num_rotation_angles):
            dict_key = (
                "reachable_R="
                + str(radius)
                + "_theta="
                + str(int(rotation_angle * rotate_step_degrees)).zfill(3)
            )
            concept_groups["reachability"].append(dict_key)

    if "pointnav" in config_name:
        target_info = ["target_r", "target_theta"]
    else:
        target_info = [
            "target_r",
            "target_theta",
            "target_visibility",
        ]  # additional target concept (visibility) for objectnav
    concept_groups["target_info"] = target_info

    agents_info = ["r", "theta"]
    concept_groups["agents_info"] = agents_info

    collision_info = ["collision"]
    concept_groups["collision_info"] = collision_info

    concept_groups["areas"] = ["Wall_area", "Floor_area"]
    concept_groups["visited"] = ["visited_l", "visited_lr", "visited_lrh"]
    concept_groups["ac_outputs"] = [
        "AC_MOVE_AHEAD",
        "AC_ROTATE_LEFT",
        "AC_ROTATE_RIGHT",
        "AC_LOOK_DOWN",
        "AC_LOOK_UP",
        "AC_END",
        "AC_POLICY",
    ]
    return concept_groups


def evaluate_concept_predictability(data_training, data_validation, concept):
    """Gradient boosted tree training to evaluate how well we can predict a 
    concept from RNN.

    Args:
        data_training (dict): dataset of RNN and concepts (Training)
        data_validation (dict): dataset of RNN and concepts (Validation)
        concept (str): which concept to predict?

    Returns:
        dict: results of evaluation e.g. correlation, F1 score, ROC_AUC
    """
    y_train = data_training["concepts"][concept]
    y_test = data_validation["concepts"][concept]
    X_train = data_training["rnn"]
    X_test = data_validation["rnn"]

    # changing dataset format suitable for xgboost
    dtrain = xgb.DMatrix(X_train.to_numpy(), label=y_train.to_numpy())
    dval = xgb.DMatrix(X_test.to_numpy(), label=y_test.to_numpy())
    results = {}

    # check if concept is binary or not
    binary_concepts = ["visited", "reachable", "visible", "visibility", "collision"]
    is_binary = False
    for binary_concept in binary_concepts:
        if binary_concept in concept:
            is_binary = True

    if is_binary:  
        # Logistic loss for binary concepts
        param = {
            "max_depth": 10,
            "objective": "binary:logistic",
            "nthread": 4,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }
    else:
        # MSE loss for non-binary concepts
        param = {
            "max_depth": 10,
            "objective": "reg:squarederror",
            "nthread": 4,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        }

    # Train GBT
    evallist = [(dtrain, "train"), (dval, "val")]
    num_round = 50
    model = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    # Evaluate GBT using base mse
    mean = np.mean(y_train.to_numpy())
    base_mse = np.mean((mean - y_test.to_numpy()) ** 2)
    mse = np.mean(
        (model.predict(xgb.DMatrix(X_test.to_numpy())) - y_test.to_numpy()) ** 2
    )

    # Also calculate ROC_AUC and fscore if the concept is binary
    if is_binary:
        predicted = np.zeros_like(y_test.to_numpy())
        predicted[model.predict(xgb.DMatrix(X_test.to_numpy())) > 0.5] = 1
        fscore = f1_score(predicted, y_test.to_numpy(), average="macro")
        roc_auc = roc_auc_score(
            y_test.to_numpy(), model.predict(xgb.DMatrix(X_test.to_numpy()))
        )
        results["roc_auc"] = roc_auc
        results["fscore"] = fscore

    # Correlation between predicted and groundtruth concept values
    correlation, _ = pearsonr(
        model.predict(xgb.DMatrix(X_test.to_numpy())), y_test.to_numpy()
    )
    results["correlation"] = correlation
    return results


def evaluate_all_concepts(config, eval_results, path):
    """runs a for loop to evaluate predictability of all concepts investigated

    Args:
        config (str): config to check if task is pointnav or objectnav
        eval_results (dict): to store evaluation results
        path (str): root directory containing training and validation dataset
    """
    dir_training = os.path.join(path, "train", config)
    dir_validation = os.path.join(path, "val", config)
    data_training, data_validation = load_dataset(dir_training, dir_validation)

    concept_groups = get_concept_groups(config)

    for concept_group, concepts in concept_groups.items():
        for concept in concepts:
            eval_results[config][concept] = evaluate_concept_predictability(
                data_training, data_validation, concept
            )


def plot_baseline_comparison(args, eval_results, concept_groups, save_dir_plots):
    """plots how well trained agent's RNN predict a concept vs. an untrained baseline

    Args:
        args (_type_): command arguments
        eval_results (dict): evaluation results of all concepts
        concept_groups (dict): dictionary containing list of concept for each group
        save_dir_plots (str): path to save plots
    """
    task = args.task
    model = args.model
    weights = ["pretrained", "random"]

    figure_labels = {}
    figure_labels["pretrained"] = task
    figure_labels["random"] = "baseline"

    groups = [
        "visited",
        "reachability",
        "agents_info",
        "target_info",
    ]  # ,'areas','collision_info','ac_outputs']

    cmap = cm.get_cmap("tab20")
    colors = []
    for i in range(4):
        colors.append(cmap(i))

    for group in groups:
        group_dict = {}

        legend = []
        correlation_data = []

        for weight in weights:
            model_id = task + "_ithor_default_" + model + "_" + weight
            group_dict[model_id] = {}
            for concept in concept_groups[group]:
                group_dict[model_id][concept] = eval_results[model_id][concept][
                    "correlation"
                ]

            legend.append(figure_labels[weight])
            correlation_data.append(group_dict[model_id])

        df = pd.DataFrame(correlation_data, index=legend).transpose()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        df.plot.bar(ax=ax, color=colors)
        ax.set_ylabel("correlation")
        ax.set_title(group)
        ax.axhline(y=0, color="k")
        ax.set_ylim([-0.1, 1.0])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        if "reachability" in group:
            thetas = [str(theta).zfill(3) for theta in range(0, 360, 30)]
            labels = [r"$\theta_{" + theta + "}$" for theta in thetas]
        elif "agents_info" in group:
            labels = [r"$R_{a}$", r"$\theta_{a}$"]
        elif "target_info" in group:
            if task=="objectnav":
                labels = [r"$R_{t}$", r"$\theta_{t}$", r"$Visible_{t}$"]
            else:
                labels = [r"$R_{t}$", r"$\theta_{t}$"]
        elif "visited" in group:
            labels = [r"${l}$", r"${lr}$", r"${lrh}$"]

        ax.set_xticklabels(labels)
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir_plots, group + ".svg"))

        fig.show()



def parse_args():
    parser = argparse.ArgumentParser(
        description="predict concepts from agent's dynamic representation (RNN)"
    )
    parser.add_argument(
        "-t", "--task", help="which task", default="objectnav", type=str
    )
    parser.add_argument("-m", "--model", help="which model", default="resnet", type=str)
    parser.add_argument(
        "-p",
        "--path",
        help="path to trajectory features",
        default="./data/trajectory_dataset",
        type=str,
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="save evaluation results and plots",
        default="./results",
        type=str,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    eval_results = {}

    # evaluate how well a trained agent's RNN can predict concepts
    config_pretrained = args.task + "_ithor_default_" + args.model + "_pretrained"
    concept_groups = get_concept_groups(config_pretrained)
    eval_results[config_pretrained] = {}
    evaluate_all_concepts(config_pretrained, eval_results, args.path)

    # evaluate how well a baseline untrained agent's RNN can predict concepts
    config_baseline = args.task + "_ithor_default_" + args.model + "_random"
    eval_results[config_baseline] = {}
    evaluate_all_concepts(config_baseline, eval_results, args.path)

    # directory to save evaluation results data
    save_dir_data = os.path.join(args.save_dir, args.task, args.model, "data")
    if not os.path.exists(save_dir_data):
        os.makedirs(save_dir_data)

    # directory to save evaluation results plots
    save_dir_plots = os.path.join(args.save_dir, args.task, args.model, "plots")
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    # save evaluation results
    with open(os.path.join(save_dir_data, "feat_importance.pickle"), "wb") as handle:
        pickle.dump(eval_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # generate and plot evaluation results of trained agent vs. a baseline
    plot_baseline_comparison(args, eval_results, concept_groups, save_dir_plots)


if __name__ == "__main__":
    main()
