from __future__ import annotations

"""Tune an SVR on frozen Transformer latent plus filtered RSSI features."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.tune_svr import (
    _extract_or_load_cache,
    _group_labels,
    _scaled_features,
)
from src.utils import distance_metrics, save_json, select_device, set_seed


def _build_regressor(c_value: float, epsilon: float) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        SVR(
            kernel="rbf",
            C=c_value,
            epsilon=epsilon,
            gamma="scale",
        )
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
    train_features = np.hstack(
        [train_latent, _scaled_features(train_frame, pipeline)]
    )
    evaluation_features = np.hstack(
        [evaluation_latent, _scaled_features(evaluation_frame, pipeline)]
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
        train_features,
        train_targets,
        groups,
    ))

    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, float | str] | None = None
    trial = 0
    for c_value in args.c_values:
        for epsilon in args.epsilon_values:
            trial += 1
            fold_truth: list[np.ndarray] = []
            fold_prediction: list[np.ndarray] = []
            for fit_idx, validation_idx in splits:
                regressor = _build_regressor(c_value, epsilon)
                regressor.fit(
                    train_features[fit_idx],
                    train_targets[fit_idx],
                )
                fold_truth.append(train_targets[validation_idx])
                fold_prediction.append(
                    regressor.predict(train_features[validation_idx])
                )

            metrics = distance_metrics(
                np.concatenate(fold_truth),
                np.concatenate(fold_prediction),
            )
            score = (
                metrics["distance_mean"]
                + 0.5 * metrics["distance_median"]
            )
            params: dict[str, float | str] = {
                "C": c_value,
                "epsilon": epsilon,
                "gamma": "scale",
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
                f"C={c_value:.5g} epsilon={epsilon:.5g}"
            )
            if score < best_score:
                best_score = score
                best_params = params

    if best_params is None:
        raise RuntimeError("No fusion SVR candidate was evaluated.")
    trials_frame = pd.DataFrame(rows).sort_values("objective")
    trials_frame.to_csv(
        args.output_dir / "fusion_svr_trials.csv",
        index=False,
        encoding="utf-8",
    )

    final_regressor = _build_regressor(
        float(best_params["C"]),
        float(best_params["epsilon"]),
    )
    final_regressor.fit(train_features, train_targets)
    evaluation_prediction = final_regressor.predict(evaluation_features)
    evaluation_metrics = distance_metrics(
        evaluation_targets,
        evaluation_prediction,
    )
    joblib.dump(
        {
            "regressor": final_regressor,
            "params": best_params,
            "feature_fusion": [
                "frozen_transformer_latent",
                "filtered_scaled_rssi",
            ],
            "transformer_model_dir": str(args.model_dir),
        },
        args.output_dir / "tuned_fusion_svr.joblib",
    )

    result = {
        "transformer_frozen": True,
        "regression_head": "multioutput_rbf_svr",
        "target_scaling": False,
        "feature_fusion": [
            "frozen_transformer_latent",
            "filtered_scaled_rssi",
        ],
        "model_dir": str(args.model_dir),
        "building_id": args.building_id,
        "cv_group": args.cv_group,
        "cv_folds": folds,
        "training_rows": len(train_frame),
        "evaluation_rows": len(evaluation_frame),
        "feature_dimensions": {
            "latent": int(train_latent.shape[1]),
            "filtered_rssi": int(train_features.shape[1] - train_latent.shape[1]),
            "fused": int(train_features.shape[1]),
        },
        "best_params": best_params,
        "best_cross_validation": trials_frame.iloc[0].to_dict(),
        "evaluation": evaluation_metrics,
    }
    save_json(
        result,
        args.output_dir / "best_fusion_svr_metrics.json",
    )

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
            "Tune SVR on frozen Transformer latent plus filtered RSSI."
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
        default="position",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 200.0],
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model" / "frozen-transformer-fusion-svr",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
