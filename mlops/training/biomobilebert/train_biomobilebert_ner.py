"""
BioMobileBERT NER Fine-tuning Script
Implements Chapter 3 methodology for malnutrition-specific NER.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset as HFDataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Entity label mapping (IOB2 format)
LABEL_LIST = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-TREATMENT", "I-TREATMENT",
    "B-MEASUREMENT", "I-MEASUREMENT",
    "B-NUTRIENT", "I-NUTRIENT",
    "B-DEMOGRAPHIC", "I-DEMOGRAPHIC",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

NUM_LABELS = len(LABEL_LIST)


class MalnutritionNERDataset(Dataset):
    """Custom dataset for malnutrition NER."""

    def __init__(self, texts: List[List[str]], labels: List[List[int]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize with word alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # For subword tokens, use same label as first subword
                label_ids.append(labels[word_idx] if labels[word_idx] % 2 == 1 else -100)
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(label_ids)

        return {k: v.squeeze(0) for k, v in encoding.items()}


class BioMobileBERTTrainer:
    """BioMobileBERT NER Trainer."""

    def __init__(
        self,
        model_name: str = "nlpie/bio-mobilebert",
        output_dir: str = "./data/models/biomobilebert_ner",
        data_dir: str = "./data/annotated",
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir)

        self.tokenizer = None
        self.model = None
        self.metric = evaluate.load("seqeval")

    def load_data(self) -> Tuple[HFDataset, HFDataset, HFDataset]:
        """Load annotated NER dataset."""
        logger.info(f"Loading data from {self.data_dir}")

        # Load from JSON files or CoNLL format
        train_data = self._load_conll_file(self.data_dir / "train.conll")
        val_data = self._load_conll_file(self.data_dir / "val.conll")
        test_data = self._load_conll_file(self.data_dir / "test.conll")

        logger.info(f"Loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        return train_data, val_data, test_data

    def _load_conll_file(self, filepath: Path) -> HFDataset:
        """Load CoNLL format file."""
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}. Creating dummy dataset.")
            return self._create_dummy_dataset()

        texts = []
        labels = []

        with open(filepath, "r", encoding="utf-8") as f:
            current_tokens = []
            current_labels = []

            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        texts.append(current_tokens)
                        labels.append([LABEL2ID.get(l, 0) for l in current_labels])
                        current_tokens = []
                        current_labels = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_labels.append(parts[-1])

        return HFDataset.from_dict({
            "tokens": texts,
            "ner_tags": labels,
        })

    def _create_dummy_dataset(self) -> HFDataset:
        """Create dummy dataset for demonstration."""
        dummy_data = {
            "tokens": [
                ["Child", "has", "severe", "acute", "malnutrition", ",", "weight", "5", "kg"],
                ["الطفل", "يعاني", "من", "سوء", "تغذية", "حاد", "،", "وزنه", "5", "كجم"],
            ],
            "ner_tags": [
                [0, 0, 1, 2, 2, 0, 7, 8, 8],  # O, O, B-DISEASE, I-DISEASE, I-DISEASE, O, B-MEASUREMENT, I-MEASUREMENT, I-MEASUREMENT
                [0, 0, 0, 1, 2, 2, 0, 7, 8, 8],
            ],
        }
        return HFDataset.from_dict(dummy_data)

    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with subword tokens."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=512,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # For subword tokens
                    if label[word_idx] % 2 == 1:  # B- tag
                        label_ids.append(label[word_idx] + 1)  # Convert to I-
                    else:
                        label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        """Compute NER metrics using seqeval."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def train(self, train_dataset, val_dataset, test_dataset=None, epochs: int = 5):
        """Fine-tune BioMobileBERT for NER."""
        logger.info("Initializing BioMobileBERT for NER fine-tuning...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # Tokenize datasets
        train_tokenized = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        val_tokenized = val_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=val_dataset.column_names,
        )

        # Training arguments optimized for low-resource environments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Evaluate on test set
        if test_dataset:
            test_tokenized = test_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=test_dataset.column_names,
            )
            test_results = trainer.evaluate(test_tokenized)
            logger.info(f"Test results: {test_results}")

        # Save final model
        trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))

        # Save label mapping
        with open(self.output_dir / "label_map.json", "w") as f:
            json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

        logger.info(f"Model saved to {self.output_dir}")

        return trainer

    def export_to_onnx(self, onnx_path: Optional[str] = None):
        """Export model to ONNX for edge deployment."""
        if onnx_path is None:
            onnx_path = str(self.output_dir / "biomobilebert_ner.onnx")

        logger.info(f"Exporting to ONNX: {onnx_path}")

        dummy_input = self.tokenizer(
            "Test sentence for ONNX export",
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )

        torch.onnx.export(
            self.model,
            (dummy_input["input_ids"], dummy_input["attention_mask"], dummy_input.get("token_type_ids", dummy_input["attention_mask"])),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

        logger.info(f"ONNX model exported to {onnx_path}")

        # Apply dynamic quantization
        self._quantize_onnx(onnx_path)

    def _quantize_onnx(self, onnx_path: str):
        """Apply INT8 dynamic quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )
            logger.info(f"Quantized model saved to {quantized_path}")
        except ImportError:
            logger.warning("onnxruntime quantization not available")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train BioMobileBERT NER")
    parser.add_argument("--data-dir", default="./data/annotated", help="Annotated data directory")
    parser.add_argument("--output-dir", default="./data/models/biomobilebert_ner", help="Output directory")
    parser.add_argument("--model", default="nlpie/bio-mobilebert", help="Base model")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX")

    args = parser.parse_args()

    trainer = BioMobileBERTTrainer(
        model_name=args.model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )

    train_data, val_data, test_data = trainer.load_data()
    trainer.train(train_data, val_data, test_data, epochs=args.epochs)

    if args.export_onnx:
        trainer.export_to_onnx()


if __name__ == "__main__":
    main()
