import numpy as np
import torch


def flatten_and_assert(array_truth, array_prediction):
    """
    Helper function to flatten the inputs and ensure they have the same length.
    """
    if isinstance(array_truth, torch.Tensor):
        array_truth = array_truth.numpy()
    if isinstance(array_prediction, torch.Tensor):
        array_prediction = array_prediction.numpy()

    array_truth = np.array(array_truth).flatten()
    array_prediction = np.array(array_prediction).flatten()

    assert len(array_truth) == len(array_prediction), "Arrays must have the same length."
    return array_truth, array_prediction


def metric_accuracy(array_truth, array_prediction) -> tuple:
    """
    Calculate accuracy for binary classification.
    """
    array_truth, array_prediction = flatten_and_assert(array_truth, array_prediction)
    correct_predictions = (array_truth == array_prediction).sum()
    return correct_predictions / len(array_truth), correct_predictions, len(array_truth)


def metric_precision(array_truth, array_prediction) -> tuple:
    """
    Calculate precision for binary classification.
    """
    array_truth, array_prediction = flatten_and_assert(array_truth, array_prediction)
    true_positive = ((array_truth == 1) & (array_prediction == 1)).sum()
    predicted_positive = (array_prediction == 1).sum()
    return true_positive / predicted_positive if predicted_positive > 0 else 0.0, true_positive, predicted_positive


def metric_recall(array_truth, array_prediction) -> tuple:
    """
    Calculate recall for binary classification.
    """
    array_truth, array_prediction = flatten_and_assert(array_truth, array_prediction)
    true_positive = ((array_truth == 1) & (array_prediction == 1)).sum()
    actual_positive = (array_truth == 1).sum()
    return true_positive / actual_positive if actual_positive > 0 else 0.0, true_positive, actual_positive


def metric_f1_score(array_truth, array_prediction) -> float:
    """
    Calculate F1 score for binary classification.
    """
    precision, _, _ = metric_precision(array_truth, array_prediction)
    recall, _, _ = metric_recall(array_truth, array_prediction)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


if __name__ == "__main__":
    array_truth = [1, 0, 1, 1, 0, 0]
    array_prediction = [1, 0, 0, 1, 1, 1]

    print("Accuracy:", metric_accuracy(array_truth, array_prediction))
    print("Precision:", metric_precision(array_truth, array_prediction))
    print("Recall:", metric_recall(array_truth, array_prediction))
    print("F1 Score:", metric_f1_score(array_truth, array_prediction))
