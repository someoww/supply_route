import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    KFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline


NON_FEATURE_COLUMNS = {
    "timestamp",
    "route_id",
    "delay_hours",
    "supply_route_condition_score",
    "condition_class",
}


COMPLEXITY_PRESETS = {
    "lite": {
        "max_iter": 150,
        "learning_rate": 0.08,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "min_samples_leaf": 40,
    },
    "balanced": {
        "max_iter": 350,
        "learning_rate": 0.06,
        "max_leaf_nodes": 63,
        "l2_regularization": 1.5,
        "min_samples_leaf": 30,
    },
    "deep": {
        "max_iter": 700,
        "learning_rate": 0.045,
        "max_leaf_nodes": 127,
        "l2_regularization": 2.5,
        "min_samples_leaf": 20,
    },
    "ultra": {
        "max_iter": 1200,
        "learning_rate": 0.03,
        "max_leaf_nodes": 255,
        "l2_regularization": 4.0,
        "min_samples_leaf": 15,
    },
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def setup_logging(log_file: str, verbosity: int) -> logging.Logger:
    logger = logging.getLogger("supply_route_trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_level = logging.WARNING
    if verbosity == 1:
        console_level = logging.INFO
    elif verbosity >= 2:
        console_level = logging.DEBUG

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def prepare_features(
    df: pd.DataFrame,
    target: str,
    task: str,
    logger: logging.Logger,
):
    df = df.copy()

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    feature_columns = [
        c for c in df.columns
        if c not in NON_FEATURE_COLUMNS and c != target
    ]

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_fraction = X.isna().mean()
    keep_missing = missing_fraction < 0.95
    dropped_missing = missing_fraction[~keep_missing].index.tolist()

    X = X.loc[:, keep_missing]

    nunique = X.nunique()
    keep_nonconstant = nunique > 1
    dropped_constant = nunique[~keep_nonconstant].index.tolist()

    X = X.loc[:, keep_nonconstant]
    feature_columns = list(X.columns)

    if len(feature_columns) == 0:
        raise ValueError("No usable features left after cleaning.")

    y = df[target]

    if task == "classification":
        y = y.astype(int)
    else:
        y = y.astype(float)

    logger.info(f"Prepared feature count: {len(feature_columns)}")

    if dropped_missing:
        logger.debug(
            f"Dropped features with >=95% missing: {dropped_missing}"
        )

    if dropped_constant:
        logger.debug(f"Dropped constant features: {dropped_constant}")

    return X, y, feature_columns


def split_dataset(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    seed: int,
    logger: logging.Logger,
):
    n = len(X)

    if "timestamp" in df.columns:
        split = int(n * (1.0 - test_size))
        train_idx = X.index[:split]
        test_idx = X.index[split:]
        logger.info("Using time-ordered train/test split.")
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split = int(n * (1.0 - test_size))
        train_idx = idx[:split]
        test_idx = idx[split:]
        logger.info("Using random train/test split.")

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    logger.info(
        f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}"
    )

    return X_train, X_test, y_train, y_test


def get_cv_splitter(
    cv_mode: str,
    has_timestamp: bool,
    task: str,
    folds: int,
    seed: int,
    y_train: pd.Series,
):
    if cv_mode == "none" or folds < 2:
        return None

    if cv_mode == "time" or (cv_mode == "auto" and has_timestamp):
        return TimeSeriesSplit(n_splits=folds)

    if task == "classification":
        min_class_count = int(pd.Series(y_train).value_counts().min())

        if min_class_count >= folds:
            return StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )

        return KFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        )

    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )


def build_pipeline(
    task: str,
    complexity: str,
    seed: int,
    verbose: int,
    early_stopping: bool,
) -> Pipeline:
    if complexity not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Invalid complexity '{complexity}'. "
            f"Valid values: {list(COMPLEXITY_PRESETS.keys())}"
        )

    params = COMPLEXITY_PRESETS[complexity].copy()
    model_verbose = 1 if verbose >= 2 else 0

    common_params = {
        "random_state": seed,
        "verbose": model_verbose,
    }

    if early_stopping:
        common_params.update(
            {
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 25,
                "tol": 1e-4,
            }
        )
    else:
        common_params.update(
            {
                "early_stopping": False,
            }
        )

    if task == "classification":
        model = HistGradientBoostingClassifier(
            **params,
            **common_params,
        )
    else:
        model = HistGradientBoostingRegressor(
            **params,
            **common_params,
        )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    classes=None,
) -> dict:
    y_pred = pipeline.predict(X)

    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y, y_pred)
            ),
            "f1_macro": float(
                f1_score(
                    y,
                    y_pred,
                    average="macro",
                    zero_division=0,
                    labels=classes,
                )
            ),
        }

    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred)),
    }


def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    has_timestamp = "timestamp" in X_train.columns or args.cv_mode == "time"
    folds = max(2, int(args.folds))

    splitter = get_cv_splitter(
        cv_mode=args.cv_mode,
        has_timestamp=has_timestamp,
        task=task,
        folds=folds,
        seed=args.seed,
        y_train=y_train,
    )

    if splitter is None:
        logger.info("Cross-validation disabled.")
        return None, None

    logger.info(
        f"Starting {folds}-fold cross-validation using "
        f"{splitter.__class__.__name__}."
    )

    classes = None
    if task == "classification":
        classes = np.unique(y_train)

    fold_metrics = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(X_train, y_train),
        start=1,
    ):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        logger.info(
            f"Fold {fold_number}: "
            f"train_rows={len(X_fold_train):,}, "
            f"valid_rows={len(X_fold_valid):,}"
        )

        if task == "classification":
            class_distribution = (
                pd.Series(y_fold_train)
                .value_counts()
                .sort_index()
                .to_dict()
            )
            logger.debug(
                f"Fold {fold_number} train class distribution: "
                f"{class_distribution}"
            )

        try:
            pipeline = build_pipeline(
                task=task,
                complexity=args.complexity,
                seed=args.seed,
                verbose=args.verbose,
                early_stopping=True,
            )

            start_time = time.time()
            pipeline.fit(X_fold_train, y_fold_train)
            fit_time = time.time() - start_time

            metrics = evaluate_model(
                pipeline=pipeline,
                X=X_fold_valid,
                y=y_fold_valid,
                task=task,
                classes=classes,
            )

            metrics["fit_time_sec"] = float(fit_time)
            fold_metrics.append(metrics)

            logger.info(
                f"Fold {fold_number} metrics: "
                f"{json.dumps(_jsonable(metrics), indent=2)}"
            )

        except Exception as exc:
            logger.warning(f"Fold {fold_number} failed: {exc}")

    if not fold_metrics:
        logger.warning("No cross-validation folds completed successfully.")
        return None, None

    metric_keys = [
        k for k in fold_metrics[0].keys()
        if k != "fit_time_sec"
    ]

    summary = {}

    for metric_name in metric_keys:
        values = np.array([m[metric_name] for m in fold_metrics], dtype=float)
        summary[f"{metric_name}_mean"] = float(np.mean(values))
        summary[f"{metric_name}_std"] = float(np.std(values))

    summary["fit_time_sec_mean"] = float(
        np.mean([m["fit_time_sec"] for m in fold_metrics])
    )

    logger.info(
        f"Cross-validation summary: "
        f"{json.dumps(_jsonable(summary), indent=2)}"
    )

    return fold_metrics, summary


def train(args: argparse.Namespace) -> None:
    logger = setup_logging(args.log_file, args.verbose)

    logger.info("Starting training run.")
    logger.debug(f"Arguments: {json.dumps(_jsonable(vars(args)), indent=2)}")

    df = pd.read_csv(args.data)
    logger.info(f"Loaded dataset shape: {df.shape}")

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found in dataset.")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Dataset sorted by timestamp.")

    X, y, feature_columns = prepare_features(
        df=df,
        target=args.target,
        task=args.task,
        logger=logger,
    )

    if args.task == "classification":
        class_distribution = y.value_counts().sort_index().to_dict()
        logger.info(
            f"Target class distribution: "
            f"{json.dumps(_jsonable(class_distribution), indent=2)}"
        )
    else:
        logger.info(
            f"Target statistics: "
            f"{json.dumps(_jsonable(y.describe().to_dict()), indent=2)}"
        )

    X_train, X_test, y_train, y_test = split_dataset(
        df=df,
        X=X,
        y=y,
        test_size=args.test_size,
        seed=args.seed,
        logger=logger,
    )

    fold_metrics, cv_summary = run_cross_validation(
        X_train=X_train,
        y_train=y_train,
        task=args.task,
        args=args,
        logger=logger,
    )

    logger.info("Training final model on train split.")

    final_pipeline = build_pipeline(
        task=args.task,
        complexity=args.complexity,
        seed=args.seed,
        verbose=args.verbose,
        early_stopping=True,
    )

    final_start = time.time()
    final_pipeline.fit(X_train, y_train)
    final_fit_time = time.time() - final_start

    logger.info(f"Final model fit time: {final_fit_time:.2f} seconds")

    classes = None
    if args.task == "classification":
        classes = np.unique(y_train)

    train_metrics = evaluate_model(
        pipeline=final_pipeline,
        X=X_train,
        y=y_train,
        task=args.task,
        classes=classes,
    )

    test_metrics = evaluate_model(
        pipeline=final_pipeline,
        X=X_test,
        y=y_test,
        task=args.task,
        classes=classes,
    )

    logger.info(
        f"Train metrics: {json.dumps(_jsonable(train_metrics), indent=2)}"
    )

    logger.info(
        f"Holdout test metrics: {json.dumps(_jsonable(test_metrics), indent=2)}"
    )

    medians = X_train.median().fillna(0.0)

    bundle = {
        "model": final_pipeline,
        "feature_columns": feature_columns,
        "medians": medians.to_dict(),
        "target": args.target,
        "task": args.task,
        "complexity": args.complexity,
        "folds": max(2, int(args.folds)) if args.cv_mode != "none" else 0,
        "cv_mode": args.cv_mode,
        "created_at": datetime.utcnow().isoformat(),
        "train_metrics": _jsonable(train_metrics),
        "holdout_metrics": _jsonable(test_metrics),
        "cv_summary": _jsonable(cv_summary) if cv_summary else None,
        "cv_fold_metrics": _jsonable(fold_metrics) if fold_metrics else None,
        "classes": classes.tolist() if args.task == "classification" else None,
        "model_type": final_pipeline.named_steps["model"].__class__.__name__,
    }

    out_dir = os.path.dirname(args.model_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.model_out, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"Saved reusable model bundle: {args.model_out}")

    print(
        json.dumps(
            _jsonable(
                {
                    "model_out": args.model_out,
                    "cv_summary": cv_summary,
                    "holdout_metrics": test_metrics,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced trainer for Bangladesh supply-route condition model."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/synthetic_supply_routes_bd.csv",
        help="Input CSV dataset path.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="condition_class",
        help="Target column. Example: condition_class, delay_hours, supply_route_condition_score.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Model task type.",
    )

    parser.add_argument(
    "--model-out",
    type=str,
    default=os.path.join("models", "model.bin"),
    help="Output model bundle path.",
    )

    parser.add_argument(
        "--complexity",
        type=str,
        default="balanced",
        choices=list(COMPLEXITY_PRESETS.keys()),
        help="Gradient boosting complexity preset.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    parser.add_argument(
        "--cv-mode",
        type=str,
        default="auto",
        choices=["auto", "time", "random", "none"],
        help=(
            "Cross-validation mode. "
            "auto uses time split if timestamp exists, otherwise random/stratified."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Holdout test size fraction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/train.log",
        help="Log file path.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity. 0=minimal, 1=normal, 2=debug/model verbose.",
    )

    args = parser.parse_args()

    train(args)