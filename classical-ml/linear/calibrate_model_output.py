from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import numpy as np

# 1. Split dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train base XGBoost model (uncalibrated)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)

# 3. Calibrate using validation set
calibrated_model = CalibratedClassifierCV(
    base_estimator=xgb, method="isotonic", cv="prefit"
)
calibrated_model.fit(X_val, y_val)

# 4. Predict uncalibrated and calibrated probabilities
probs_uncal = xgb.predict_proba(X_val)[:, 1]
probs_calib = calibrated_model.predict_proba(X_val)[:, 1]

# 5. Evaluate calibration using Brier Score (lower is better)
print("Uncalibrated Brier score:", brier_score_loss(y_val, probs_uncal))
print("Calibrated Brier score:  ", brier_score_loss(y_val, probs_calib))


# 6. Optional: Rescale calibrated probabilities to [1, 900]
def rescale_probability(prob, min_val=1, max_val=900):
    return min_val + prob * (max_val - min_val)


rescaled_scores = rescale_probability(probs_calib)
