def mean_average_precision(predictions, ground_truths):
    """
    Computes the Mean Average Precision (MAP).
    
    Args:
        predictions: List of lists, where each sublist is a ranked list of predicted items for a query.
        ground_truths: List of lists, where each sublist contains the relevant items for a query.

    Returns:
        MAP score as a float.
    """
    def average_precision(predicted, relevant):
        """
        Computes Average Precision (AP) for a single query.
        
        Args:
            predicted: List of predicted items.
            relevant: Set of relevant items.

        Returns:
            AP score as a float.
        """
        if not relevant:
            return 0.0  # No relevant items, AP is zero.

        ap_sum = 0.0
        num_relevant = 0
        
        for i, item in enumerate(predicted):
            if item in relevant:  # Check if the item is relevant
                num_relevant += 1
                precision_at_k = num_relevant / (i + 1)
                ap_sum += precision_at_k
        
        return ap_sum / len(relevant)  # Normalize by the number of relevant items

    # Compute AP for each query and average them
    average_precisions = [
        average_precision(predictions[i], set(ground_truths[i]))
        for i in range(len(predictions))
    ]
    
    return sum(average_precisions) / len(average_precisions)  # Mean of APs


if __name__ == "__main__":
    # Ground truth relevant items
    ground_truths = [
        ["A", "B", "D"],  # Relevant items for query 1
        ["B", "C"],       # Relevant items for query 2
    ]

    # Predicted ranked lists
    predictions = [
        ["A", "D", "C", "B"],  # Predicted list for query 1
        ["C", "B", "A", "D"],  # Predicted list for query 2
    ]

    # Calculate MAP
    map_score = mean_average_precision(predictions, ground_truths)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")