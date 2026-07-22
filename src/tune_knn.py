from __future__ import annotations

"""Tune a KNN regression head on features from a frozen Transformer."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.tune_svr import _extract_or_load_cache, _group_labels
from src.utils import distance_metrics, save_json, select_device, set_seed


def _prepare_features(
    fit_features: np.ndarray,
    predict_features: np.ndarray,
    scale_features: bool,
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    if not scale_features:
        return fit_features, predict_features, None
    scaler = StandardScaler()
    return (
        scaler.fit_transform(fit_features),
        scaler.transform(predict_features),
        scaler,
    )


def _build_regressor(
    neighbors: int,
    metric: str,
) -> KNeighborsRegressor:
    return KNeighborsRegressor(
        n_neighbors=neighbors,
        weights="distance",
        metric=metric,
        algorithm="brute",
    )


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(args.model_dir / "pipeline.joblib")
    train_frame = pd.read_csv(args.train_csv)
    evaluation_frame = pd.read_csv(args.eval_csv)
    train_frame = train_frame.loc[
        train_frame["BUILDINGID"] == args.building_id
    ].reset_index(drop=True)
    evaluation_frame = evaluation_frame.loc[
        evaluation_frame["BUILDINGID"] == args.building_id
    ].reset_index(drop=True)
    if train_frame.empty or evaluation_frame.empty:
        raise ValueError("No rows remain after applying the building filter.")

    train_latent, evaluation_latent = _extract_or_load_cache(
        args,
        pipeline,
        train_frame,
        evaluation_frame,
        device,
    )
    train_targets = train_frame[["LONGITUDE", "LATITUDE"]].to_numpy()
    evaluation_targets = evaluation_frame[
        ["LONGITUDE", "LATITUDE"]
    ].to_numpy()

    groups = _group_labels(train_frame, args.cv_group)
    folds = min(args.folds, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("Grouped cross-validation needs at least 2 groups.")
    splits = list(GroupKFold(n_splits=folds).split(
        train_latent,
        train_targets,
        groups,
    ))
    largest_neighbors = min(len(fit_idx) for fit_idx, _ in splits)

    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    trial = 0
    for scale_features in (False, True):
        for metric in args.metrics:
            for neighbors in args.neighbors:
                if neighbors > largest_neighbors:
                    continue
                trial += 1
                fold_truth: list[np.ndarray] = []
                fold_prediction: list[np.ndarray] = []
                for fit_idx, validation_idx in splits:
                    fit_features, validation_features, _ = _prepare_features(
                        train_latent[fit_idx],
                        train_latent[validation_idx],
                        scale_features,
                    )
                    regressor = _build_regressor(neighbors, metric)
                    regressor.fit(fit_features, train_targets[fit_idx])
                    fold_truth.append(train_targets[validation_idx])
                    fold_prediction.append(
                        regressor.predict(validation_features)
                    )

                metrics = distance_metrics(
                    np.concatenate(fold_truth),
                    np.concatenate(fold_prediction),
                )
                score = (
                    metrics["distance_mean"]
                    + 0.5 * metrics["distance_median"]
                )
                params = {
                    "neighbors": neighbors,
                    "metric": metric,
                    "weights": "distance",
                    "scale_features": scale_features,
                }
                rows.append({
                    "trial": trial,
                    **params,
                    **metrics,
                    "objective": score,
                })
                print(
                    f"trial={trial:03d} mean={metrics['distance_mean']:.4f} "
                    f"median={metrics['distance_median']:.4f} "
                    f"p90={metrics['distance_p90']:.4f} "
                    f"k={neighbors} metric={metric} "
                    f"scaled={scale_features}"
                )
                if score < best_score:
                    best_score = score
                    best_params = params

    if best_params is None:
        raise RuntimeError("No KNN candidate was evaluated.")
    trials_frame = pd.DataFrame(rows).sort_values("objective")
    trials_frame.to_csv(
        args.output_dir / "knn_trials.csv",
        index=False,
        encoding="utf-8",
    )

    fit_features, evaluation_features, scaler = _prepare_features(
        train_latent,
        evaluation_latent,
        bool(best_params["scale_features"]),
    )
    final_regressor = _build_regressor(
        int(best_params["neighbors"]),
        str(best_params["metric"]),
    )
    final_regressor.fit(fit_features, train_targets)
    evaluation_prediction = final_regressor.predict(evaluation_features)
    evaluation_metrics = distance_metrics(
        evaluation_targets,
        evaluation_prediction,
    )

    joblib.dump(
        {
            "regressor": final_regressor,
            "latent_scaler": scaler,
            "params": best_params,
            "transformer_model_dir": str(args.model_dir),
        },
        args.output_dir / "tuned_knn.joblib",
    )
    result = {
        "transformer_frozen": True,
        "regression_head": "distance_weighted_knn",
        "model_dir": str(args.model_dir),
        "building_id": args.building_id,
        "cv_group": args.cv_group,
        "cv_folds": folds,
        "training_rows": len(train_frame),
        "evaluation_rows": len(evaluation_frame),
        "best_params": best_params,
        "best_cross_validation": trials_frame.iloc[0].to_dict(),
        "evaluation": evaluation_metrics,
    }
    save_json(result, args.output_dir / "best_knn_metrics.json")

    prediction_frame = evaluation_frame.copy()
    prediction_frame["PRED_LONGITUDE"] = evaluation_prediction[:, 0]
    prediction_frame["PRED_LATITUDE"] = evaluation_prediction[:, 1]
    prediction_frame["ERROR_DISTANCE"] = np.linalg.norm(
        evaluation_targets - evaluation_prediction,
        axis=1,
    )
    prediction_frame.to_csv(
        args.output_dir / "predictions_evaluation.csv",
        index=False,
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a KNN regression head while keeping an existing "
            "Transformer frozen."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_CONFIG.data.train_csv,
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_CONFIG.data.eval_csv,
    )
    parser.add_argument("--building-id", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument(
        "--cv-group",
        choices=("phone", "position"),
        default="phone",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 30],
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("euclidean", "cosine"),
        default=["euclidean", "cosine"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model" / "frozen-transformer-knn",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
