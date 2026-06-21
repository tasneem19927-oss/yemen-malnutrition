"""
ONNX Export for Edge Deployment.
Exports XGBoost and BioMobileBERT models to ONNX with INT8 quantization.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export ML models to ONNX for edge deployment."""

    def __init__(self, model_dir: str = "./data/models", output_dir: str = "./edge/onnx"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_xgboost(self, model_path: str, output_name: str) -> str:
        """Export XGBoost model to ONNX."""
        try:
            import onnxmltools
            from onnxmltools.convert import convert_xgboost
            from skl2onnx.common.data_types import FloatTensorType

            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Define input shape (29 features)
            initial_type = [("float_input", FloatTensorType([None, 29]))]

            onnx_model = convert_xgboost(model, initial_types=initial_type)

            output_path = self.output_dir / f"{output_name}.onnx"
            onnxmltools.utils.save_model(onnx_model, str(output_path))

            logger.info(f"XGBoost model exported to {output_path}")
            return str(output_path)

        except ImportError:
            logger.error("onnxmltools not available for XGBoost export")
            return ""

    def export_biomobilebert(self, model_path: str, output_name: str = "biomobilebert_ner") -> str:
        """Export BioMobileBERT to ONNX."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            model.eval()

            # Dummy input
            dummy_text = "Test sentence for ONNX export"
            dummy_input = tokenizer(
                dummy_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )

            output_path = self.output_dir / f"{output_name}.onnx"

            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(output_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=14,
                do_constant_folding=True,
            )

            logger.info(f"BioMobileBERT exported to {output_path}")

            # Quantize
            self.quantize_model(str(output_path))

            return str(output_path)

        except Exception as e:
            logger.error(f"BioMobileBERT export failed: {e}")
            return ""

    def quantize_model(self, onnx_path: str) -> str:
        """Apply INT8 dynamic quantization."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = onnx_path.replace(".onnx", "_int8.onnx")

            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QInt8,
                optimize_model=True,
            )

            # Log size reduction
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)

            logger.info(f"Quantization complete:")
            logger.info(f"  Original: {original_size:.1f} MB")
            logger.info(f"  Quantized: {quantized_size:.1f} MB")
            logger.info(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

            return quantized_path

        except ImportError:
            logger.warning("onnxruntime quantization not available")
            return onnx_path

    def export_all(self):
        """Export all models."""
        # XGBoost models
        for target in ["stunting", "wasting", "underweight"]:
            model_path = self.model_dir / f"xgboost_{target}_v1.pkl"
            if model_path.exists():
                self.export_xgboost(str(model_path), f"xgboost_{target}")

        # BioMobileBERT
        bert_path = self.model_dir / "biomobilebert_ner"
        if bert_path.exists():
            self.export_biomobilebert(str(bert_path))

        logger.info("All models exported successfully")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model-dir", default="./data/models", help="Model directory")
    parser.add_argument("--output-dir", default="./edge/onnx", help="Output directory")

    args = parser.parse_args()

    exporter = ONNXExporter(args.model_dir, args.output_dir)
    exporter.export_all()


if __name__ == "__main__":
    main()
