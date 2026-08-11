def ensemble_prediction(ml_pred, dl_pred):
    """
    Stacking / Voting Classifier logic:
    If both ML and DL models agree, return that result.
    If they disagree, give higher weight to the Deep Learning model (e.g., BERT/LSTM).
    """
    if ml_pred == dl_pred:
        return ml_pred, "High Confidence (Models Agreed)"
    else:
        # If ML and DL disagree, we can choose to trust the DL model more due to its advanced architecture and ability to capture complex patterns.
        return dl_pred, "Moderate Confidence (Ensemble Weighted to DL)"

# Example usage of the ensemble_prediction function:
print("Ensemble Test:", ensemble_prediction(1, 1))