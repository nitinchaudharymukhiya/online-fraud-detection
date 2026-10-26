
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    form_data = {
        "amount": "1500",
        "transaction_location": "1",
        "usual_location": "1",
        "transactions_last_hour": "2",
        "new_device": "0",
        "otp_verified": "1"
    }

    if request.method == "POST":
        try:
            # Read submitted form values
            for key in form_data:
                form_data[key] = request.form.get(key, "")

            amount = float(form_data["amount"])
            transaction_location = int(
                form_data["transaction_location"]
            )
            usual_location = int(form_data["usual_location"])
            transactions_last_hour = int(
                form_data["transactions_last_hour"]
            )
            new_device = int(form_data["new_device"])
            otp_verified = int(form_data["otp_verified"])

            # Validate inputs
            if not 0 < amount <= 10_000_000:
                raise ValueError(
                    "Amount must be between ₹1 and ₹1,00,00,000."
                )

            if transaction_location not in (0, 1):
                raise ValueError("Invalid transaction location.")

            if usual_location not in (0, 1):
                raise ValueError("Invalid usual location.")

            if not 0 <= transactions_last_hour <= 100:
                raise ValueError(
                    "Transactions must be between 0 and 100."
                )

            if new_device not in (0, 1):
                raise ValueError("Invalid new-device selection.")

            if otp_verified not in (0, 1):
                raise ValueError("Invalid OTP selection.")

            # Prepare input in the same order/names used for training
            transaction = pd.DataFrame([{
                "amount": amount,
                "transaction_location": transaction_location,
                "usual_location": usual_location,
                "transactions_last_hour": transactions_last_hour,
                "new_device": new_device,
                "otp_verified": otp_verified
            }])

            # Model prediction and estimated probability
            probability = float(
                model.predict_proba(transaction)[0][1]
            )
            prediction = int(model.predict(transaction)[0])

            # Educational demo rule:
            # unverified OTP always triggers a suspicious flag
            if otp_verified == 0:
                prediction = 1

            result = {
                "is_fraud": prediction == 1,
                "risk": round(probability * 100, 1)
            }

        except (ValueError, TypeError) as exc:
            error = str(exc) or "Please enter valid transaction details."

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
