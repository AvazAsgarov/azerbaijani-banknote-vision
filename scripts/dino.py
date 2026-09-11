"""DINOv2 Foundation Model Feature Extraction and Linear Probe Evaluation.

Self-supervised Vision Transformer (DINOv2 ViT-L/14) embeddings are extracted
from curated banknote scenes, persisted to HDF5/NumPy/Parquet containers, and
evaluated via linear classification probing across zero-leakage splits.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths
from src.data.embeddings import ExtractionConfig, FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DINOv2Engine")


class DINOv2LinearProbe:
    """Conducts linear probing evaluation on extracted DINOv2 representations.

    Attributes:
        paths: Configuration container resolving artifact and dataset locations.
        embeddings_path: Path pointing to the serialized dense NumPy feature matrix.
        metadata_path: Path pointing to the Parquet table containing sample splits.
    """

    def __init__(
        self,
        paths: Optional[ProjectPaths] = None,
        embeddings_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        """Initializes the evaluation probe with dataset artifact paths.

        Args:
            paths: ProjectPaths container instance.
            embeddings_path: Path pointing to numpy feature matrix.
            metadata_path: Path pointing to parquet sample manifest.
        """
        self.paths = paths or ProjectPaths()
        self.embeddings_path = embeddings_path or self.paths.embeddings_npy_path
        self.metadata_path = metadata_path or self.paths.embeddings_parquet_path

    def load_representations(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Loads dense feature vectors and associated split metadata.

        Returns:
            Tuple containing feature array and sample dataframe.
        """
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not located at {self.embeddings_path}. Run extraction first."
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not located at {self.metadata_path}. Run extraction first."
            )

        features = np.load(self.embeddings_path)
        metadata = pd.read_parquet(self.metadata_path)
        logger.info(
            "Loaded %d sample vectors with dimension %d from %s",
            features.shape[0],
            features.shape[1],
            self.embeddings_path.name,
        )
        return features, metadata

    def train_and_evaluate(
        self,
        c_regularization: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Trains a multiclass logistic regression probe on the train split.

        Args:
            c_regularization: Inverse regularization strength parameter.
            max_iter: Maximum iterations allowed for convergence.
            random_state: Deterministic random generator state.

        Returns:
            Structured dictionary housing evaluation metrics across validation and test.
        """
        features, metadata = self.load_representations()

        train_mask = (metadata["split"] == "train").to_numpy()
        val_mask = (metadata["split"] == "val").to_numpy()
        test_mask = (metadata["split"] == "test").to_numpy()

        x_train, y_train = features[train_mask], metadata.loc[train_mask, "class_name"].to_numpy()
        x_val, y_val = features[val_mask], metadata.loc[val_mask, "class_name"].to_numpy()
        x_test, y_test = features[test_mask], metadata.loc[test_mask, "class_name"].to_numpy()

        classes = sorted(np.unique(y_train))
        logger.info(
            "Partition summary: Train=%d, Val=%d, Test=%d across %d classes",
            len(y_train),
            len(y_val),
            len(y_test),
            len(classes),
        )

        logger.info(
            "Fitting Logistic Regression probe (C=%.2f, max_iter=%d)...",
            c_regularization,
            max_iter,
        )
        start_time = time.perf_counter()
        classifier = LogisticRegression(
            C=c_regularization,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        classifier.fit(x_train, y_train)
        fit_duration = time.perf_counter() - start_time
        logger.info("Classifier optimization finished in %.2f seconds", fit_duration)


        val_preds = classifier.predict(x_val)
        test_preds = classifier.predict(x_test)

        val_acc = float(accuracy_score(y_val, val_preds))
        test_acc = float(accuracy_score(y_test, test_preds))

        val_report = classification_report(y_val, val_preds, output_dict=True, zero_division=0)
        test_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
        test_cm = confusion_matrix(y_test, test_preds, labels=classes).tolist()


        results: Dict[str, Any] = {
            "model": "DINOv2-ViT-L/14 (Linear Probe Baseline)",
            "regularization_c": c_regularization,
            "fit_time_seconds": fit_duration,
            "sample_counts": {
                "train": int(len(y_train)),
                "val": int(len(y_val)),
                "test": int(len(y_test)),
            },
            "validation_metrics": {
                "accuracy": val_acc,
                "macro_f1": float(val_report["macro avg"]["f1-score"]),
                "weighted_f1": float(val_report["weighted avg"]["f1-score"]),
                "detailed_report": val_report,
            },
            "test_metrics": {
                "accuracy": test_acc,
                "macro_f1": float(test_report["macro avg"]["f1-score"]),
                "weighted_f1": float(test_report["weighted avg"]["f1-score"]),
                "detailed_report": test_report,
                "confusion_matrix": {
                    "labels": classes,
                    "matrix": test_cm,
                },
            },
        }


        output_dir = self.paths.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dino_linear_probe_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Evaluation report serialized to %s", report_path)
        logger.info("Validation Accuracy: %.4f | Macro F1: %.4f", val_acc, val_report["macro avg"]["f1-score"])
        logger.info("Test Accuracy:       %.4f | Macro F1: %.4f", test_acc, test_report["macro avg"]["f1-score"])

        return results



def run_extraction(config: Optional[ExtractionConfig] = None) -> None:
    """Extracts dense feature representations from curated image files.

    Args:
        config: ExtractionConfig settings instance.
    """
    cfg = config or ExtractionConfig()
    logger.info("Starting DINOv2 feature extraction using model %s...", cfg.model_name)
    extractor = FeatureExtractor(config=cfg)
    extractor.run()
    logger.info("Feature extraction completed successfully.")



def main() -> None:
    """CLI interface handling DINOv2 extraction and linear probing evaluation."""
    parser = argparse.ArgumentParser(
        description="DINOv2 Feature Extraction and Linear Probe Evaluation Engine"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Triggers feature extraction pass over master image pool.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Evaluates a linear classification probe on extracted representations.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executes extraction followed immediately by linear probing.",
    )
    parser.add_argument(
        "--c-reg",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the logistic regression probe.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for linear probe convergence.",
    )


    args = parser.parse_args()
    paths = ProjectPaths()


    if not (args.extract or args.probe or args.all):
        if paths.embeddings_npy_path.exists() and paths.embeddings_parquet_path.exists():
            logger.info("Existing embeddings located. Proceeding with linear probe evaluation.")
            args.probe = True
        else:
            logger.info("Embeddings not detected on disk. Executing extraction and probing.")
            args.all = True


    if args.extract or args.all:
        run_extraction()


    if args.probe or args.all:
        probe = DINOv2LinearProbe(paths=paths)
        probe.train_and_evaluate(c_regularization=args.c_reg, max_iter=args.max_iter)



if __name__ == "__main__":
    main()
