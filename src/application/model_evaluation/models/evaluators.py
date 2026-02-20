# evaluators.py
from typing import Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EvaluationMetrics:
    """
    Best Practice: Combine multiple metrics for comprehensive evaluation
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def evaluate(self, predictions: list, expected: list) -> Dict[str, float]:
        """
        Calculate all metrics

        Returns metrics dict with:
        - exact_match: Exact string match (0-1)
        - semantic_similarity: Embedding similarity (0-1)
        - accuracy: Classification accuracy (if applicable)
        - avg_confidence: Model confidence (if available)
        """

        metrics = {}

        # 1. Exact Match
        metrics['exact_match'] = self._exact_match(predictions, expected)

        # 2. Semantic Similarity
        metrics['semantic_similarity'] = self._semantic_similarity(
            predictions, expected)

        # 3. Contains Expected (for longer outputs)
        metrics['contains_match'] = self._contains_match(predictions, expected)

        # 4. Classification metrics (if applicable)
        if self._is_classification(expected):
            acc, prec, rec, f1 = self._classification_metrics(
                predictions, expected)
            metrics['accuracy'] = acc
            metrics['precision'] = prec
            metrics['recall'] = rec
            metrics['f1_score'] = f1

        print(f"all metrics")
        print(metrics)
        print(f"="*60)

        return metrics

    def _exact_match(self, predictions: list, expected: list) -> float:
        """Percentage of exact matches"""
        matches = sum(1 for p, e in zip(predictions, expected)
                      if p.strip().lower() == e.strip().lower())
        return matches / len(predictions)

    def _semantic_similarity(self, predictions: list, expected: list) -> float:
        """Average cosine similarity of embeddings"""
        pred_embeddings = self.embedding_model.encode(predictions)
        exp_embeddings = self.embedding_model.encode(expected)

        similarities = []
        for p_emb, e_emb in zip(pred_embeddings, exp_embeddings):
            sim = cosine_similarity([p_emb], [e_emb])[0][0]
            similarities.append(sim)

        return float(np.mean(similarities))

    def _contains_match(self, predictions: list, expected: list) -> float:
        """Percentage where prediction contains expected"""
        matches = sum(1 for p, e in zip(predictions, expected)
                      if e.strip().lower() in p.strip().lower())
        return matches / len(predictions)

    def _is_classification(self, expected: list) -> bool:
        """Check if this is a classification task"""
        unique_values = set(e.strip().lower() for e in expected)
        return len(unique_values) <= 20  # Arbitrary threshold

    def _classification_metrics(self, predictions: list, expected: list):
        """Calculate classification metrics"""
        # Normalize
        pred_norm = [p.strip().lower() for p in predictions]
        exp_norm = [e.strip().lower() for e in expected]

        # Get unique labels
        labels = sorted(set(exp_norm))

        # Calculate metrics
        accuracy = accuracy_score(exp_norm, pred_norm)
        prec, rec, f1, _ = precision_recall_fscore_support(
            exp_norm, pred_norm,
            labels=labels,
            average='weighted',
            zero_division=0
        )

        return accuracy, prec, rec, f1
