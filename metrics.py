import pandas as pd
import ast
from typing import List, Dict
from fuzzywuzzy import fuzz
from data_config import IO_PATH
from data_handler import load_data, save_data
import re


def normalize_text(text: str) -> str:
    """Normalize entity strings for fair comparison."""
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower().strip())


def character_level_similarity(str1: str, str2: str) -> float:
    """Returns a similarity score between 0 and 1."""
    return fuzz.ratio(normalize_text(str1), normalize_text(str2)) / 100.0


def calculate_soft_row_metrics(
    true_entities: List[str],
    pred_entities: List[str],
    threshold: float
) -> Dict[str, float]:
    """Calculate soft precision, recall, F1 using a similarity threshold."""
    if not true_entities and not pred_entities:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "accuracy": 1.0}
    if not pred_entities:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}
    if not true_entities:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0}

    matched_true = set()
    matched_pred = set()

    for i, true in enumerate(true_entities):
        for j, pred in enumerate(pred_entities):
            if character_level_similarity(true, pred) >= threshold:
                matched_true.add(i)
                matched_pred.add(j)

    tp = len(matched_true)
    fp = len(pred_entities) - len(matched_pred)
    fn = len(true_entities) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = len(matched_pred) / len(pred_entities) if pred_entities else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4)
    }


def process_row_soft(row: pd.Series, threshold: float) -> Dict[str, float]:
    true_entities = ast.literal_eval(row["ground_truth"]) if isinstance(row["ground_truth"], str) else row["ground_truth"]
    pred_entities = ast.literal_eval(row["Org_name"]) if isinstance(row["Org_name"], str) else row["Org_name"]
    return calculate_soft_row_metrics(true_entities, pred_entities, threshold)


def find_best_threshold(df: pd.DataFrame, thresholds: List[float]) -> tuple[float, Dict[str, float]]:
    """Evaluate thresholds and return the best one based on F1 score."""
    best_threshold = None
    best_metrics = None
    best_f1 = -1
    results = []

    for threshold in thresholds:
        df["metrics"] = df.apply(lambda row: process_row_soft(row, threshold), axis=1)

        avg_precision = df["metrics"].apply(lambda x: x["precision"]).mean()
        avg_recall = df["metrics"].apply(lambda x: x["recall"]).mean()
        avg_f1 = df["metrics"].apply(lambda x: x["f1_score"]).mean()
        avg_accuracy = df["metrics"].apply(lambda x: x["accuracy"]).mean()

        current_metrics = {
            "threshold": threshold,
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_f1": round(avg_f1, 4),
            "average_accuracy": round(avg_accuracy, 4)
        }
        results.append(current_metrics)

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_threshold = threshold
            best_metrics = current_metrics

    print("Threshold Evaluation Results:")
    for res in results:
        print(res)

    return best_threshold, best_metrics


if __name__ == "__main__":
    df = load_data(IO_PATH)
    thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
    best_threshold, best_metrics = find_best_threshold(df.copy(), thresholds)

    print(f"\nBest threshold: {best_threshold}")
    print(f"Best metrics: {best_metrics}")

    df["metrics"] = df.apply(lambda row: process_row_soft(row, best_threshold), axis=1)
    save_data(df, IO_PATH)
