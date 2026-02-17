# 📱 Mobile Health App Review Analysis

---

## 🏆 Results at a Glance

| Metric | Value |
|--------|-------|
| 🎯 ML Accuracy (CV) | **95.61%** |
| 📊 Test Set Accuracy | **92.86%** |
| 📱 Apps Analyzed | **137 unique apps** |
| 🔬 Mean SUS Score | **85.4 / 100** |
| 🩺 Clinician Promoters | **43.8%** |
| 🤖 Best Model | **Soft-Voting Ensemble** |

---

## 📁 Repository Structure

```
├── MobileHealthApp_Hackathon.ipynb   # Main analysis notebook
├── mobile_health_app_review_data.csv # Dataset
├── README.md                         # You are here
└── outputs/
    ├── shap_importance_by_class.png
    ├── shap_beeswarm_excellent.png
    ├── shap_bar_all_classes.png
    ├── shap_waterfall.png
    └── shap_dependence.png
```

---

## 📖 Project Overview

This project analyzes **274 reviews** of mobile health apps collected from both **clinician** and **non-clinician** reviewers. The dataset captures a dual-reviewer structure where:

- `clinician_reviewer` — fills the **NPS-style Recommendation Score** (0–10)
- `nonclinician_reviewer` — fills all **10 SUS (System Usability Scale)** items (1–5 Likert)

The goal is to uncover usability trends, compliance gaps, and engagement patterns — and build a machine learning model that predicts usability grades with **≥95% accuracy**.

---

## 🗂️ Dataset Description

| Column Group | Description |
|---|---|
| `AppID`, `AppName` | App identifier and name |
| `Reviewer`, `Platform`, `OS` | Review metadata |
| `SUS Items (Q1–Q10)` | 10 Likert-scale usability questions |
| `Recommendation Score` | 0–10 NPS-style clinician score |
| `Binary Feature Flags` | Privacy policy, login, ads, device sync, etc. |
| `Developer Info` | Developer type (for-profit, gov, academic, etc.) |
| `App Store Ratings` | iOS/Android avg ratings and download counts |

---

## 🧹 Data Preprocessing

- **Dual-reviewer structure** identified and handled separately
- **Likert parser** converts mixed strings like `"5 (Strongly agree)"` → `5.0`
- **SUS Score** computed using the standard formula:
  - Odd items `(Q1,3,5,7,9)`: contribution = `score − 1`
  - Even items `(Q2,4,6,8,10)`: contribution = `5 − score`
  - Final score = `sum × 2.5` → range **[0, 100]**
- **Binary encoding** for all Yes/No compliance features
- **Median imputation** for any remaining missing SUS values

---

## 📊 Exploratory Data Analysis

The notebook contains **9 visualizations** covering:

1. 📈 SUS Score distribution with grade thresholds
2. 🔢 Item-level mean scores + correlation heatmap
3. 🗳️ Clinician recommendation score distribution
4. 📋 Feature adoption/compliance rates across all apps
5. 🥧 Developer type breakdown
6. 📦 SUS boxplots by key features (privacy policy, device sync, etc.)
7. 🖥️ Platform & OS impact on usability
8. 👥 Patient engagement modalities
9. 🏢 SUS vs clinical expert involvement

### Key EDA Findings

- 📌 Mean SUS of **85.4** — well above the industry benchmark of **68**
- 📌 Only **54%** of apps have a privacy policy
- 📌 Only **36%** involve clinical experts in development
- 📌 Only **7%** claim HIPAA compliance
- 📌 Only **13%** sync with wearable/peripheral devices
- 📌 Only **20%** reward users for engagement

---

## 🤖 Machine Learning

### Target Variable
**SUS Usability Grade** (3-class classification):

| Class | Grade | SUS Range |
|-------|-------|-----------|
| 0 | Poor / Marginal | < 75 |
| 1 | Good | 75 – 89 |
| 2 | Excellent | ≥ 90 |

### Feature Engineering

On top of the raw SUS items and binary flags, these engineered features were created:

| Feature | Formula |
|---|---|
| `pos_subscale` | Sum of Q1+Q3+Q5+Q7+Q9 |
| `neg_subscale` | Sum of Q2+Q4+Q6+Q8+Q10 |
| `pos_neg_ratio` | `pos / (neg + 0.001)` |
| `ease_cluster` | Q3 + Q7 + Q9 |
| `complexity_cluster` | Q2 + Q4 + Q8 |
| `consistency_score` | Q5 − Q6 |
| `compliance_score` | Sum of all binary feature flags |

### Models Evaluated

| Model | CV Accuracy |
|-------|------------|
| Logistic Regression | ~72% |
| Gradient Boosting | ~91% |
| Random Forest | ~95% |
| Extra Trees | ~94% |
| XGBoost *(if installed)* | ~95% |
| LightGBM *(if installed)* | ~95% |
| **Ensemble (Soft Vote)** | **95.61% ✅** |

> 5-Fold Stratified Cross-Validation used throughout.

---

## 🔍 Explainable AI — SHAP Analysis

The notebook includes a full **SHAP (SHapley Additive exPlanations)** analysis with 5 plots:

| Plot | Description |
|------|-------------|
| **Mean \|SHAP\| by class** | Top features driving each usability grade |
| **Beeswarm plot** | How feature values push predictions for "Excellent" class |
| **Stacked bar** | Combined importance across all 3 classes |
| **Waterfall plot** | Step-by-step explanation for one individual app |
| **Dependence plots** | How the top 2 features interact with SHAP values |

### Top SHAP Drivers
- `⊕ Positive Subscale` — single strongest predictor
- `Ease Cluster (Q3+Q7+Q9)` — confidence + learnability
- `⊕/⊖ Ratio` — balance of positive vs negative responses
- `Complexity Cluster` — negatively impacts Excellent class
- `Q9: Felt Confident` — high individual importance

---

## 🚀 How to Run

### Option 1 — Google Colab *(recommended)*

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `MobileHealthApp_Hackathon.ipynb`
3. Upload `mobile_health_app_review_data.csv` to the Colab session
4. Run all cells top to bottom

### Option 2 — Local

```bash
git clone https://github.com/Sundaram04/Innofusion_hackathon.git
cd Innofusion
pip install pandas numpy matplotlib seaborn scikit-learn shap xgboost lightgbm
jupyter notebook MobileHealthApp_Hackathon.ipynb
```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
xgboost        # optional but recommended
lightgbm       # optional but recommended
jupyter
```

---

## 💡 Key Recommendations

Based on EDA and SHAP findings, the top actions for app developers are:

- 🔐 **Add a privacy policy** — missing in 46% of apps, directly impacts user trust
- 👨‍⚕️ **Involve clinical experts** in development and QA (only 36% do)
- 🎮 **Add reward/gamification systems** — only 20% of apps currently do this
- 🔔 **Build safety alerts** for dangerous health data inputs (only 17% do)
- 📲 **Integrate with wearables** — a major gap, only 13% support device sync
- 📋 **Reference clinical guidelines** — boosts clinician recommendation scores
- 🏥 **Pursue HIPAA compliance** — currently claimed by only 7% of apps

---

## 📌 Notebook Structure

| Section | Content |
|---------|---------|
| 1 | Setup & Imports |
| 2 | Data Loading & Inspection |
| 3 | Cleaning & Preprocessing |
| 4 | Exploratory Data Analysis (9 plots) |
| 5 | ML — Feature Engineering |
| 6 | ML — Cross-Validation & Model Comparison |
| 7 | ML — Ensemble & Final Results |
| 8 | ML — Visualizations & Confusion Matrix |
| 9 | Feature Importance (Random Forest) |
| 10 | Explainable AI — SHAP Analysis (5 plots) |
| 11 | Bonus: Recommendation Score Prediction |
| 12 | Key Insights & Recommendations |

---

## 👤 Team BLOCKSMITHS

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
