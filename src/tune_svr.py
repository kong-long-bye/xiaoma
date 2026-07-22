from __future__ import annotations

"""Freeze a trained Transformer and tune only the downstream SVR."""

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import DEFAULT_CONFIG, ModelConfig, PROJECT_ROOT
from src.model import WiFiTransformerAutoencoder
from src.train import extract_latent
from src.utils import distance_metrics, save_json, select_device, set_seed


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def _load_frozen_model(
    model_dir: Path,
    pipeline: dict[str, Any],
    device: torch.device,
) -> WiFiTransformerAutoencoder:
    checkpoint = _load_checkpoint(
        model_dir / str(pipeline["transformer_file"]),
        device,
    )
    valid_names = {item.name for item in fields(ModelConfig)}
    model_config = ModelConfig(
        **{
            key: value
            for key, value in checkpoint["model_config"].items()
            if key in valid_names
        }
    )
    model = WiFiTransformerAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        config=model_config,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _scaled_features(
    frame: pd.DataFrame,
    pipeline: dict[str, Any],
) -> np.ndarray:
    feature_names = list(pipeline["feature_names"])
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing {len(missing)} required WAP columns.")
    features = (
        frame[feature_names]
        .replace(
            pipeline["raw_missing_value"],
            pipeline["filled_missing_value"],
        )
        .fillna(pipeline["filled_missing_value"])
    )
    return pipeline["feature_scaler"].transform(
        features.to_numpy()
    ).astype(np.float32)


def _position_groups(frame: pd.DataFrame) -> np.ndarray:
    columns = ["FLOOR", "SPACEID", "RELATIVEPOSITION"]
    if not set(columns).issubset(frame.columns):
        columns = ["LONGITUDE", "LATITUDE"]
    return (
        frame[columns]
        .round(4)
        .astype(str)
        .agg("|".join, axis=1)
        .to_numpy()
    )


def _group_labels(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "phone":
        if "PHONEID" not in frame.columns:
            raise KeyError("PHONEID is required for phone-grouped CV.")
        return frame["PHONEID"].astype(str).to_numpy()
    return _position_groups(frame)


def _build_regressor(
    c_value: float,
    epsilon: float,
    gamma: str | float,
    scale_targets: bool = True,
) -> Any:
    feature_regressor = Pipeline(
        [
            (
                "svr",
                MultiOutputRegressor(
                    SVR(
                        kernel="rbf",
                        C=c_value,
                        epsilon=epsilon,
                        gamma=gamma,
                    )
                ),
            ),
        ]
    )
    if not scale_targets:
        return feature_regressor
    return TransformedTargetRegressor(
        regressor=feature_regressor,
        transformer=StandardScaler(),
    )


def _candidates(count: int, seed: int) -> list[dict[str, str | float]]:
    fixed: list[dict[str, str | float]] = [
        {"C": 100.0, "epsilon": 0.1, "gamma": "scale"},
        {
            "C": 223.1289322251285,
            "epsilon": 0.5082090355009669,
            "gamma": "scale",
        },
    ]
    rng = np.random.default_rng(seed)
    while len(fixed) < count:
        fixed.append(
            {
                "C": float(10 ** rng.uniform(np.log10(10), np.log10(500))),
                "epsilon": float(
                    10 ** rng.uniform(np.log10(0.01), np.log10(0.8))
                ),
                "gamma": float(
                    10 ** rng.uniform(np.log10(0.001), np.log10(0.2))
                ),
            }
        )
    return fixed[:count]


def _extract_or_load_cache(
    args: argparse.Namespace,
    pipeline: dict[str, Any],
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = args.output_dir / "latent_cache.npz"
    if cache_path.exists() and not args.refresh_cache:
        cached = np.load(cache_path)
        if (
            len(cached["train_latent"]) != len(train_frame)
            or len(cached["evaluation_latent"]) != len(evaluation_frame)
        ):
            raise ValueError(
                "Latent cache row count does not match the input data. "
                "Use --refresh-cache."
            )
        print(f"Loaded latent cache: {cache_path}")
        return cached["train_latent"], cached["evaluation_latent"]

    model = _load_frozen_model(args.model_dir, pipeline, device)
    print("Extracting frozen Transformer features; no training is performed.")
    train_latent = extract_latent(
        model,
        _scaled_features(train_frame, pipeline),
        device,
        args.feature_batch_size,
    )
    evaluation_latent = extract_latent(
        model,
        _scaled_features(evaluation_frame, pipeline),
        device,
        args.feature_batch_size,
    )
    np.savez_compressed(
        cache_path,
        train_latent=train_latent,
        evaluation_latent=evaluation_latent,
    )
    print(f"Saved latent cache: {cache_path}")
    return train_latent, evaluation_latent


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
    unique_groups = np.unique(groups)
    folds = min(args.folds, len(unique_groups))
    if folds < 2:
        raise ValueError("Grouped cross-validation needs at least 2 groups.")

    splitter = GroupKFold(n_splits=folds)
    rows: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, str | float] | None = None
    for trial_number, params in enumerate(
        _candidates(args.trials, args.seed),
        start=1,
    ):
        fold_truth: list[np.ndarray] = []
        fold_prediction: list[np.ndarray] = []
        for fit_idx, validation_idx in splitter.split(
            train_latent,
            train_targets,
            groups,
        ):
            regressor = _build_regressor(
                float(params["C"]),
                float(params["epsilon"]),
                params["gamma"],
                scale_targets=not args.no_target_scaling,
            )
            regressor.fit(train_latent[fit_idx], train_targets[fit_idx])
            fold_truth.append(train_targets[validation_idx])
            fold_prediction.append(
                regressor.predict(train_latent[validation_idx])
            )
        truth = np.concatenate(fold_truth)
        prediction = np.concatenate(fold_prediction)
        metrics = distance_metrics(truth, prediction)
        score = metrics["distance_mean"] + 0.5 * metrics["distance_median"]
        row = {
            "trial": trial_number,
            **params,
            **metrics,
            "objective": score,
        }
        rows.append(row)
        print(
            f"trial={trial_number:03d}/{args.trials:03d} "
            f"mean={metrics['distance_mean']:.4f} "
            f"median={metrics['distance_median']:.4f} "
            f"p90={metrics['distance_p90']:.4f} "
            f"C={float(params['C']):.5g} "
            f"epsilon={float(params['epsilon']):.5g} "
            f"gamma={params['gamma']}"
        )
        if score < best_score:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("No SVR candidate was evaluated.")
    trials_frame = pd.DataFrame(rows).sort_values("objective")
    trials_frame.to_csv(
        args.output_dir / "svr_trials.csv",
        index=False,
        encoding="utf-8",
    )

    final_regressor = _build_regressor(
        float(best_params["C"]),
        float(best_params["epsilon"]),
        best_params["gamma"],
        scale_targets=not args.no_target_scaling,
    )
    final_regressor.fit(train_latent, train_targets)
    evaluation_prediction = final_regressor.predict(evaluation_latent)
    evaluation_metrics = distance_metrics(
        evaluation_targets,
        evaluation_prediction,
    )
    joblib.dump(
        final_regressor,
        args.output_dir / "tuned_regressor.joblib",
    )
    result = {
        "transformer_frozen": True,
        "model_dir": str(args.model_dir),
        "building_id": args.building_id,
        "cv_group": args.cv_group,
        "target_scaling": not args.no_target_scaling,
        "cv_folds": folds,
        "training_rows": len(train_frame),
        "evaluation_rows": len(evaluation_frame),
        "best_params": best_params,
        "best_cross_validation": trials_frame.iloc[0].to_dict(),
        "evaluation": evaluation_metrics,
    }
    save_json(result, args.output_dir / "best_svr_metrics.json")
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
        description="Tune SVR while keeping an existing Transformer frozen."
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
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument(
        "--cv-group",
        choices=("phone", "position"),
        default="phone",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "model" / "frozen-svr-tuning",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--no-target-scaling",
        action="store_true",
        help="Fit SVR on raw coordinates, matching the original paper pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
