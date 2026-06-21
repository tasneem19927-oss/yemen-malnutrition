"""
XGBoost Model Training for Malnutrition Prediction.
Trains 3 independent models: Stunting, Wasting, Underweight.
"""

import os
import sys
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Train and evaluate XGBoost malnutrition prediction models."""

    FEATURE_COLUMNS = [
        "age_months", "sex", "weight", "height", "oedema",
        "breastfeeding", "vitamin_a", "diarrhea_recent", "fever_recent",
        "cough_recent", "maternal_education", "wealth_index", "residence_type",
        "haz", "whz", "waz", "bmi", "weight_height_ratio",
        "age_weight_interaction", "age_height_interaction",
        "health_risk_score", "nutrition_risk_score", "muac_zscore",
        "head_circumference_zscore", "growth_velocity",
        "wasting_stunting_interaction", "age_group", "season",
        "food_security_index",
    ]

    TARGET_COLUMNS = {
        "stunting": "stunting",
        "wasting": "wasting",
        "underweight": "underweight",
    }

    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.models = {}
        self.metrics = {}

    def load_data(self):
        """Load and preprocess Yemen MICS6 dataset."""
        logger.info(f"Loading data from {self.data_path}")

        # Load CSV or parquet
        if self.data_path.endswith(".csv"):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith(".parquet"):
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Parquet.")

        logger.info(f"Loaded {len(self.data)} records")

        # Feature engineering
        self.data = self._engineer_features(self.data)

        # Handle missing values
        self.data = self.data.fillna(self.data.median(numeric_only=True))

        logger.info("Data preprocessing complete")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        # BMI
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

        # Weight-height ratio
        df["weight_height_ratio"] = df["weight"] / df["height"]

        # Age interactions
        df["age_weight_interaction"] = df["age_months"] * df["weight"]
        df["age_height_interaction"] = df["age_months"] * df["height"]

        # Risk scores
        df["health_risk_score"] = (
            df["diarrhea_recent"].astype(int) +
            df["fever_recent"].astype(int) +
            df["cough_recent"].astype(int) +
            df["oedema"].astype(int)
        )

        # Nutrition risk
        df["nutrition_risk_score"] = (
            (~df["breastfeeding"]).astype(int) * 0.3 +
            (~df["vitamin_a"]).astype(int) * 0.2 +
            df["oedema"].astype(int) * 0.4
        )

        # Age group
        df["age_group"] = (df["age_months"] // 12).clip(0, 4)

        # Season (simplified)
        df["season"] = pd.to_datetime(df.get("measurement_date", pd.Timestamp.now())).dt.month % 4

        # Food security
        df["food_security_index"] = (
            df["wealth_index"] / 5.0 - df["health_risk_score"] / 10.0
        )

        return df

    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50) -> dict:
        """Use Optuna for hyperparameter optimization."""

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "booster": "gbtree",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
            }

            model = xgb.XGBClassifier(**params)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def train_model(self, target_name: str, optimize: bool = True) -> dict:
        """Train a single malnutrition prediction model."""
        logger.info(f"Training {target_name} model...")

        X = self.data[self.FEATURE_COLUMNS]
        y = self.data[self.TARGET_COLUMNS[target_name]]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Hyperparameter optimization
        if optimize:
            best_params = self.optimize_hyperparameters(X_train, y_train, n_trials=30)
        else:
            best_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

        # Train final model
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "booster": "gbtree",
            **best_params,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False,
        )

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_prob),
        }

        logger.info(f"{target_name} metrics: {metrics}")

        # Feature importance
        importance = pd.DataFrame({
            "feature": self.FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        logger.info(f"Top features for {target_name}:")
        logger.info(importance.head(10).to_string())

        # Save model
        model_path = self.output_dir / f"xgboost_{target_name}_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics
        metrics_path = self.output_dir / f"metrics_{target_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save feature importance
        importance_path = self.output_dir / f"importance_{target_name}.csv"
        importance.to_csv(importance_path, index=False)

        self.models[target_name] = model
        self.metrics[target_name] = metrics

        return metrics

    def train_all(self, optimize: bool = True):
        """Train all three models."""
        for target in self.TARGET_COLUMNS.keys():
            self.train_model(target, optimize=optimize)

        # Save combined metrics
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "models": self.metrics,
        }
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("All models trained successfully")

    def cross_validate(self, target_name: str, n_splits: int = 5) -> dict:
        """Perform cross-validation."""
        X = self.data[self.FEATURE_COLUMNS]
        y = self.data[self.TARGET_COLUMNS[target_name]]

        model = self.models.get(target_name)
        if not model:
            raise ValueError(f"Model {target_name} not trained yet")

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        return {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "fold_scores": scores.tolist(),
        }


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost malnutrition models")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--output", default="./data/models", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--target", choices=["stunting", "wasting", "underweight", "all"], default="all")

    args = parser.parse_args()

    trainer = XGBoostTrainer(args.data, args.output)
    trainer.load_data()

    if args.target == "all":
        trainer.train_all(optimize=args.optimize)
    else:
        trainer.train_model(args.target, optimize=args.optimize)


if __name__ == "__main__":
    main()
