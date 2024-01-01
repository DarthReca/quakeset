import logging
from argparse import ArgumentParser
from typing import Literal

import joblib
import numpy as np
from dataset import EarthQuakeDataModule
from sklearn import decomposition, ensemble, pipeline, svm
from sklearn.metrics import accuracy_score, mean_absolute_error


def main(
    model_name: Literal["svm-poly", "rf", "svm-rbf"],
    task: Literal["classification", "regression", "multiregression"],
):
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] [%(asctime)s] %(message)s"
    )
    post_only = True
    dm = EarthQuakeDataModule(
        file="data/earthquakes.h5",
        magnitudes_file="data/earthquakes.parquet",
        split_file="data/earthquakes_split.parquet",
        num_workers=4,
        batch_size=8,
        fill_nan=0,
        post_only=post_only,
    )
    dm.setup()
    label_key = "magnitude" if task == "regression" else "label"

    train_dataset = [(s["sample"].flatten(), s[label_key]) for s in dm.train_dataset]
    val_dataset = [(s["sample"].flatten(), s[label_key]) for s in dm.val_dataset]

    train_x, train_y = zip(*train_dataset)
    val_x, val_y = zip(*val_dataset)
    logging.info(f"Train size: {len(train_x)}")
    if "svm" in model_name:
        kernel = "poly" if "poly" in model_name else "rbf"
        model_class = svm.SVR if task == "regression" else svm.SVC
        model = model_class(kernel=kernel, verbose=True)
    else:
        model_class = (
            ensemble.RandomForestRegressor
            if task == "regression"
            else ensemble.RandomForestClassifier
        )
        model = model_class(verbose=True)
    # Fit
    pca = decomposition.PCA()
    pip = pipeline.make_pipeline(pca, model)
    pip.fit(train_x, train_y)
    # Save
    addition = "_post" if post_only else ""
    logging.info(f"Saving model to checkpoints/{model_name}_{task}{addition}.joblib")
    joblib.dump(pip, f"checkpoints/{model_name}_{task}{addition}.joblib")
    # Validate
    val_y_pred = pip.predict(val_x)
    if task == "classification":
        val_acc = accuracy_score(val_y, val_y_pred)
        logging.info(f"Val accuracy: {val_acc}")
    else:
        val_mae = mean_absolute_error(val_y, val_y_pred)
        logging.info(f"Val MAE: {val_mae}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, choices=["svm-poly", "rf", "svm-rbf"])
    parser.add_argument(
        "--task", type=str, choices=["classification", "regression", "multiregression"]
    )
    args = parser.parse_args()
    main(args.model_name, args.task)
