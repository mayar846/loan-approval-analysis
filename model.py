"""
Loan Approval Prediction - Baseline Machine Learning Models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)

IMG = "images/"
sns.set_theme(style="whitegrid", font_scale=1.05)

df = pd.read_csv("data/loan_approval_cleaned.csv")

df_model = df.copy()
df_model["education"] = (df_model["education"] == "Graduate").astype(int)
df_model["self_employed"] = (df_model["self_employed"] == "Yes").astype(int)
df_model["target"] = (df_model["loan_status"] == "Approved").astype(int)

features = ["no_of_dependents", "education", "self_employed", "income_annum",
            "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
            "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]

X = df_model[features]
y = df_model["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "pred": pred,
        "proba": proba
    }
    print(f"\n{name}")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {k}: {results[name][k]:.4f}")

# ---------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------
metrics_df = pd.DataFrame({name: {k: v[k] for k in ["accuracy", "precision", "recall", "f1"]}
                            for name, v in results.items()}).T
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_df.plot(kind="bar", ax=ax, color=["#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A"])
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(IMG + "11_model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Confusion matrix - Random Forest (best model typically)
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
cm = confusion_matrix(y_test, results[best_name]["pred"])
fig, ax = plt.subplots(figsize=(6, 5.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(IMG + "12_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.tight_layout()
plt.savefig(IMG + "13_roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Feature importance - Random Forest
# ---------------------------------------------------------------
rf = results["Random Forest"]["model"]
importance = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
importance.plot(kind="barh", ax=ax, color="#00695C")
ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(IMG + "14_feature_importance.png", dpi=150)
plt.close()

print("\nBest model:", best_name)
print("\nAll charts and metrics generated.")
