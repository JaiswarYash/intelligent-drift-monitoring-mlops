def should_alert(drift_result, threshold=3):

    if drift_result["num_drifted_features"] >= threshold:
        return True, "Dataset-level drift detected"
    
    if drift_result["num_drifted_features"] > threshold:
        return True, f"{drift_result['num_drifted_features']} features have drifted"
    
    return False, "No significant drift detected"