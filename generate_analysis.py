"""
Loan Approval Prediction - Exploratory Data Analysis
Generates a cleaned dataset and all chart images used in the notebook/report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {"Approved": "#2E7D32", "Rejected": "#C62828"}
IMG = "images/"

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
df = pd.read_csv("data/loan_approval_dataset_raw.csv")
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()

df.drop(columns=["loan_id"], inplace=True)

# Feature engineering
df["total_assets_value"] = (
    df["residential_assets_value"] + df["commercial_assets_value"] +
    df["luxury_assets_value"] + df["bank_asset_value"]
)
df["loan_to_income_ratio"] = df["loan_amount"] / df["income_annum"]
df["asset_to_loan_ratio"] = df["total_assets_value"] / df["loan_amount"]

cibil_bins = [0, 550, 650, 750, 900]
cibil_labels = ["Poor (<550)", "Fair (550-650)", "Good (650-750)", "Excellent (750+)"]
df["cibil_band"] = pd.cut(df["cibil_score"], bins=cibil_bins, labels=cibil_labels)

df.to_csv("data/loan_approval_cleaned.csv", index=False)
print("Cleaned dataset saved:", df.shape)

# ---------------------------------------------------------------
# 2. Approval rate overview
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["loan_status"].value_counts()
ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
       colors=[PALETTE[k] for k in counts.index], wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Overall Loan Status Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "01_overall_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. CIBIL score - the strongest predictor
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for status, color in PALETTE.items():
    sns.kdeplot(df.loc[df.loan_status == status, "cibil_score"], fill=True,
                label=status, color=color, alpha=0.4, ax=ax)
ax.set_title("CIBIL Score Distribution by Loan Outcome", fontsize=14, fontweight="bold")
ax.set_xlabel("CIBIL Score")
ax.legend(title="Loan Status")
plt.tight_layout()
plt.savefig(IMG + "02_cibil_distribution.png", dpi=150)
plt.close()

# CIBIL band approval rate
band_rate = (df.groupby("cibil_band", observed=True)["loan_status"]
             .apply(lambda s: (s == "Approved").mean() * 100))
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(band_rate.index.astype(str), band_rate.values, color="#1565C0")
ax.set_title("Approval Rate (%) by CIBIL Score Band", fontsize=14, fontweight="bold")
ax.set_ylabel("Approval Rate (%)")
ax.set_ylim(0, 100)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "03_approval_by_cibil_band.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Education & self-employment
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ["education", "self_employed"],
                           ["Approval Rate by Education", "Approval Rate by Self-Employment"]):
    rate = df.groupby(col)["loan_status"].apply(lambda s: (s == "Approved").mean() * 100)
    bars = ax.bar(rate.index, rate.values, color=["#00695C", "#EF6C00"])
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Approval Rate (%)")
    ax.set_ylim(0, 100)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
                ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "04_education_employment.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Income vs Loan Amount scatter
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
for status, color in PALETTE.items():
    sub = df[df.loan_status == status]
    ax.scatter(sub["income_annum"]/1e6, sub["loan_amount"]/1e6, s=14, alpha=0.4,
               color=color, label=status)
ax.set_xlabel("Annual Income (Million)")
ax.set_ylabel("Loan Amount (Million)")
ax.set_title("Income vs Loan Amount by Approval Status", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(IMG + "05_income_vs_loanamount.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Loan-to-income ratio
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for status, color in PALETTE.items():
    sns.kdeplot(df.loc[df.loan_status == status, "loan_to_income_ratio"], fill=True,
                label=status, color=color, alpha=0.4, ax=ax)
ax.set_title("Loan-to-Income Ratio by Loan Outcome", fontsize=14, fontweight="bold")
ax.set_xlabel("Loan Amount / Annual Income")
ax.legend(title="Loan Status")
plt.tight_layout()
plt.savefig(IMG + "06_loan_to_income_ratio.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 7. Assets vs approval
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="loan_status", y="total_assets_value", hue="loan_status",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Total Assets Value by Loan Status", fontsize=14, fontweight="bold")
ax.set_ylabel("Total Assets Value")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(IMG + "07_assets_by_status.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 8. Loan term
# ---------------------------------------------------------------
term_rate = df.groupby("loan_term")["loan_status"].apply(lambda s: (s == "Approved").mean() * 100)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(term_rate.index, term_rate.values, marker="o", color="#4527A0", linewidth=2)
ax.set_title("Approval Rate by Loan Term (Years)", fontsize=14, fontweight="bold")
ax.set_xlabel("Loan Term (Years)")
ax.set_ylabel("Approval Rate (%)")
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(IMG + "08_approval_by_term.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 9. Dependents
# ---------------------------------------------------------------
dep_rate = df.groupby("no_of_dependents")["loan_status"].apply(lambda s: (s == "Approved").mean() * 100)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(dep_rate.index.astype(str), dep_rate.values, color="#AD1457")
ax.set_title("Approval Rate by Number of Dependents", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Dependents")
ax.set_ylabel("Approval Rate (%)")
ax.set_ylim(0, 100)
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5, f"{b.get_height():.1f}%",
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "09_approval_by_dependents.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 10. Correlation heatmap
# ---------------------------------------------------------------
numeric_cols = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
                "cibil_score", "residential_assets_value", "commercial_assets_value",
                "luxury_assets_value", "bank_asset_value", "total_assets_value",
                "loan_to_income_ratio"]
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "10_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# Summary stats printed for reporting
# ---------------------------------------------------------------
print("\n--- KEY STATS ---")
print("Approval rate overall: {:.1f}%".format((df.loan_status == "Approved").mean() * 100))
print("Median CIBIL - approved:", df.loc[df.loan_status == "Approved", "cibil_score"].median())
print("Median CIBIL - rejected:", df.loc[df.loan_status == "Rejected", "cibil_score"].median())
print("Correlation CIBIL vs approval (numeric encode):",
      np.corrcoef(df["cibil_score"], (df["loan_status"] == "Approved").astype(int))[0, 1])
print("Approval rate CIBIL>=650:", (df.loc[df.cibil_score >= 650, "loan_status"] == "Approved").mean() * 100)
print("Approval rate CIBIL<650:", (df.loc[df.cibil_score < 650, "loan_status"] == "Approved").mean() * 100)
