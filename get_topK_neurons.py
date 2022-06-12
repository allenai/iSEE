import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


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


def show_topK_neurons(data_training, data_validation, concept, save_dir_plots, k=5):
    """_summary_

    Args:
        data_training (dict): dataset of RNN and concepts (Training)
        data_validation (dict): dataset of RNN and concepts (Validation)
        concept (str): concept for which we want to find top-K neurons
        save_dir_plots (str): path to save plots
        k (int, optional): k in top-K. Defaults to 5.
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

    # Apply SHAP to find top-k relevant neurons in prediction
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, max_display=k, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_plots, concept + "_beeswarm.svg"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="predict concepts from agent's dynamic representation (RNN)"
    )
    parser.add_argument(
        "-t", "--task", help="which task", default="objectnav", type=str
    )
    parser.add_argument("-m", "--model", help="which model", default="resnet", type=str)
    parser.add_argument("-c", "--concept", help="which concept", default="reachable_R=2_theta=000", type=str)
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
        default="./results/",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Load dataset
    config_pretrained = args.task + "_ithor_default_" + args.model + "_pretrained"
    dir_training = os.path.join(args.path, "train", config_pretrained)
    dir_validation = os.path.join(args.path, "val", config_pretrained)
    data_training, data_validation = load_dataset(dir_training, dir_validation)

    # Create directory to save plots
    save_dir_plots = os.path.join(args.save_dir, args.task, args.model, "shap_plots")
    if not os.path.exists(save_dir_plots):
        os.makedirs(save_dir_plots)

    # get top-K relevant neurons for predicting a concept
    concept = args.concept 
    show_topK_neurons(data_training, data_validation, concept, save_dir_plots, k=20)


if __name__ == "__main__":
    main()
