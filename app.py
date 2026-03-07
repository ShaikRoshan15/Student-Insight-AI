from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib
import pandas as pd
import os
import json

# Prevent Tkinter Thread Error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ================= LOAD PREPROCESSORS ================= #
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
num_classes = len(label_encoder.classes_)

# ================= MODEL ================= #
class AdvancedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = AdvancedMLP()
model.load_state_dict(torch.load("model/mlp_model.pt", map_location=torch.device("cpu")))
model.eval()

# ================= HOME ================= #
@app.route('/')
def home():
    return render_template("index.html")

# ================= MODEL DASHBOARD ================= #
@app.route('/dashboard')
def dashboard():

    df = pd.read_csv("student_data.csv")

    # ----------- NEW ANALYTICS -----------
    total_students = len(df)
    high_risk = len(df[df["RiskLabel"] == "High Risk"])
    suspended = len(df[df["RiskLabel"] == "Suspended"])
    dropout = len(df[df["RiskLabel"] == "Dropout"])

    risk_percent = round((high_risk + suspended + dropout) / total_students * 100, 2)

    stats = {
        "total_students": total_students,
        "avg_attendance": round(df["Attendance"].mean(), 2),
        "avg_cgpa": round(df["CGPA"].mean(), 2),
        "high_risk": high_risk,
        "suspended": suspended,
        "dropout": dropout,
        "risk_percent": risk_percent
    }

    # ----------- LOAD METRICS -----------
    try:
        with open("model/metrics.json", "r") as f:
            metrics = json.load(f)
    except:
        metrics = {"accuracy": 0, "f1_score": 0, "precision": 0, "recall": 0}

    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1_score"]

    os.makedirs("static/images", exist_ok=True)

    # ----------- RISK DISTRIBUTION CHART -----------
    plt.figure(figsize=(8, 5))
    df["RiskLabel"].value_counts().plot(kind="barh", color="#2e77a3")
    plt.title("Distribution of Risk Labels")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("static/images/risk_chart.png")
    plt.close()

    # ----------- PERFORMANCE CHART -----------
    metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
    metric_values = [accuracy, f1, precision, recall]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metric_names, metric_values, color="#2e77a3")
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics")

    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value}%",
            ha='center'
        )

    plt.tight_layout()
    plt.savefig("static/images/performance_chart.png")
    plt.close()

    return render_template(
        "dashboard.html",
        stats=stats,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        risk_chart="images/risk_chart.png",
        performance_chart="images/performance_chart.png"
    )

# ================= PREDICTION ================= #
# ================= PREDICTION ================= #
# ================= PREDICTION ================= #
@app.route('/predict/<prediction_type>', methods=['GET', 'POST'])
def predict(prediction_type):

    if request.method == "POST":
        try:

            # ---------------------------
            # 1️⃣ Get Form Data
            # ---------------------------
            cgpa = float(request.form['cgpa'])
            attendance = float(request.form['attendance'])
            backlogs = float(request.form['backlogs'])
            income = float(request.form['income'])
            behavior = float(request.form['behavior'])
            trend = float(request.form['trend'])

            # ---------------------------
            # 2️⃣ Prepare DataFrame
            # ---------------------------
            features = pd.DataFrame([{
                "CGPA": cgpa,
                "Attendance": attendance,
                "Backlogs": backlogs,
                "ParentalIncome": income,
                "BehaviorRating": behavior,
                "PerformanceTrend": trend
            }])

            # ---------------------------
            # 3️⃣ Scale Input
            # ---------------------------
            scaled = scaler.transform(features)
            tensor_input = torch.tensor(scaled, dtype=torch.float32)

            # ---------------------------
            # 4️⃣ Model Prediction
            # ---------------------------
            with torch.no_grad():
                outputs = model(tensor_input)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            model_label = label_encoder.inverse_transform(
                [predicted.item()]
            )[0]

            confidence_percent = round(confidence.item() * 100, 2)

            # ---------------------------
            # 5️⃣ Business Logic Override
            # ---------------------------
            suspension_score = 0
            dropout_score = 0

            if behavior < 4:
                suspension_score += 3

            if attendance < 60:
                suspension_score += 2

            if trend == -1:
                suspension_score += 1


            if income < 20000:
                dropout_score += 3

            if cgpa < 5:
                dropout_score += 2

            if backlogs >= 1 and backlogs <= 3:
                dropout_score += 1

            elif backlogs > 3:
                dropout_score += 2


            if suspension_score >= 3 and suspension_score > dropout_score:
                final_label = "Suspended"

            elif dropout_score >= 3 and dropout_score > suspension_score:
                final_label = "Dropout"

            else:
                final_label = model_label


            # ---------------------------
            # 6️⃣ Generate Explanation
            # ---------------------------
            reasons = []
            suggestions = []

            if final_label == "Suspended":

                if behavior < 4:
                    reasons.append("Low behavior rating.")

                if attendance < 60:
                    reasons.append("Very low attendance.")

                if trend == -1:
                    reasons.append("Declining academic trend.")

                suggestions.append(
                    "Immediate counseling and disciplinary monitoring recommended."
                )


            elif final_label == "Dropout":

                if income < 20000:
                    reasons.append("Financial instability affecting education.")

                if cgpa < 5:
                    reasons.append("Very low CGPA.")

                if backlogs > 3:
                    reasons.append("Multiple academic backlogs.")

                suggestions.append(
                    "Provide financial aid and academic mentoring."
                )


            elif final_label == "High Risk":

                if attendance < 70:
                    reasons.append("Low attendance level.")

                if cgpa < 6:
                    reasons.append("Below average CGPA.")

                if backlogs >= 2:
                    reasons.append("Multiple academic backlogs.")

                suggestions.append(
                    "Close academic monitoring and mentoring required."
                )


            elif final_label == "Medium Risk":

                if attendance < 75:
                    reasons.append("Moderate attendance level.")

                if cgpa < 7:
                    reasons.append("Average academic performance.")

                if backlogs >= 1:
                    reasons.append("Presence of academic backlogs.")

                suggestions.append(
                    "Encourage consistent study habits and monitor progress."
                )


            elif final_label == "Low Risk":

                reasons.append(
                    "Student shows stable academic performance and good behavior."
                )

                suggestions.append(
                    "Maintain current academic performance and engagement."
                )


            explanation = " ".join(reasons)
            suggestion = " ".join(suggestions)


            # ---------------------------
            # 7️⃣ Render Result
            # ---------------------------
            return render_template(
                "prediction.html",
                prediction_type=prediction_type,
                predicted_label=final_label,
                confidence=confidence_percent,
                explanation=explanation,
                suggestion=suggestion,
                cgpa=cgpa,
                attendance=attendance,
                backlogs=backlogs,
                income=income,
                behavior=behavior,
                trend=trend
            )

        except Exception as e:

            return render_template(
                "prediction.html",
                prediction_type=prediction_type,
                error=str(e)
            )


    return render_template(
        "prediction.html",
        prediction_type=prediction_type,
        error=None,
        cgpa=None,
        attendance=None,
        backlogs=None,
        income=None,
        behavior=None,
        trend=None
    )

# ================= RUN ================= #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
