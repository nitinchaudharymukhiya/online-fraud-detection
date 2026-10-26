
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

# Synthetic demo data — real banking data nahi hai.
np.random.seed(42)
n = 10000

amount = np.random.uniform(100, 100000, n)
transaction_location = np.random.randint(0, 2, n)
usual_location = np.random.randint(0, 2, n)
transactions_last_hour = np.random.randint(0, 12, n)
new_device = np.random.randint(0, 2, n)
otp_verified = np.random.randint(0, 2, n)

# Illustrative rule to generate demo labels.
risk_score = (
    (amount > 50000).astype(int)
    + (transaction_location != usual_location).astype(int) * 2
    + (transactions_last_hour >= 6).astype(int) * 2
    + new_device * 2
    + (1 - otp_verified) * 3
)

fraud_probability = np.clip(
    0.05 + risk_score * 0.12, 0.02, 0.95
)
fraud = np.random.binomial(1, fraud_probability)

X = pd.DataFrame({
    "amount": amount,
    "transaction_location": transaction_location,
    "usual_location": usual_location,
    "transactions_last_hour": transactions_last_hour,
    "new_device": new_device,
    "otp_verified": otp_verified
})

y = fraud

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

joblib.dump(model, "fraud_model.joblib")
print("New demo model saved successfully!")

