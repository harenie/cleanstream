"""Fine-tune a transformer classifier for concept coverage labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1.concept_coverage.llm_concept_coverage import LABEL_TO_ID, save_label_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train missing/partial/covered concept coverage classifier."
    )
    parser.add_argument(
        "--training-data",
        type=Path,
        default=Path("data/training/concept_coverage_training.csv"),
        help="CSV with model_input and label columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("module1/models/concept_coverage_model"),
        help="Directory to save the trained model.",
    )
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Use CUDA when available with auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Training requires torch and transformers. Install with: "
            "pip install -r requirements-llm.txt"
        ) from exc

    data = pd.read_csv(args.training_data)
    missing = {"model_input", "label"}.difference(data.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {sorted(missing)}")
    data = data[data["label"].isin(LABEL_TO_ID)].copy()
    if data.empty:
        raise ValueError("No training rows with labels missing, partial, or covered.")

    train_df, val_df = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=data["label"] if data["label"].value_counts().min() >= 2 else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_TO_ID),
        id2label={value: key for key, value in LABEL_TO_ID.items()},
        label2id=LABEL_TO_ID,
    )
    selected_device = choose_device(torch, args.device)
    model.to(selected_device)

    train_dataset = ConceptCoverageDataset(train_df, tokenizer, args.max_length)
    val_dataset = ConceptCoverageDataset(val_df, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(selected_device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.item())
        print(f"epoch={epoch + 1} train_loss={total_loss / max(1, len(train_loader)):.4f}")

    metrics = evaluate_model(model, val_loader, selected_device, torch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_label_config(args.output_dir)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                **metrics,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "base_model": args.base_model,
                "device": selected_device,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True))
    print(f"\nSaved concept coverage model to: {args.output_dir}")


class ConceptCoverageDataset:
    def __init__(self, dataframe: pd.DataFrame, tokenizer: object, max_length: int) -> None:
        self.labels = [LABEL_TO_ID[label] for label in dataframe["label"].tolist()]
        inputs = dataframe["model_input"].astype(str).tolist()
        self.encodings = tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {key: value[index] for key, value in self.encodings.items()}
        import torch

        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def choose_device(torch: object, requested_device: str) -> str:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return "cuda"
    if requested_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model(model: object, data_loader: object, device: str, torch: object) -> dict[str, object]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            expected = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predicted = torch.argmax(logits, dim=-1)
            predictions.extend(predicted.cpu().tolist())
            labels.extend(expected.cpu().tolist())

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(LABEL_TO_ID.values()),
            target_names=list(LABEL_TO_ID.keys()),
            zero_division=0,
            output_dict=True,
        ),
    }


if __name__ == "__main__":
    main()
