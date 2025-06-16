
# 🚗 ACIS Car Insurance Claim Analysis & Premium Optimization

This project delivers a robust, data-driven framework for risk assessment and dynamic pricing of car insurance policies for **AlphaCare Insurance Solutions (ACIS)**. It combines exploratory analysis, statistical testing, and machine learning to optimize claim predictions and premium pricing while maintaining fairness and transparency.

---

## 📁 Project Structure

ACIS-Car-Insurance-Claim-Analysis/
│
├── data/ # Raw and processed datasets (tracked via DVC)
├── notebooks/ # Jupyter notebooks for each task
├── src/ # Python modules (data loading, preprocessing, modeling)
├── reports/ # Final visualizations, insights, and exports
├── dvc.yaml # DVC pipeline definitions
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage

yaml
Copy
Edit

---

## 🎯 Project Objectives

- Understand key drivers of insurance claims and pricing.
- Evaluate statistical differences across customer segments.
- Build predictive models for claim severity and occurrence.
- Propose a risk-based pricing formula using interpretable ML.
- Ensure fairness, auditability, and reproducibility in modeling.

---

## 📊 Tasks Overview

### ✅ Task 1: Exploratory Data Analysis (EDA)

- Analyzed features affecting claim behavior (e.g., vehicle age, power).
- Identified regional disparities in claims and premiums.
- Flagged outliers and inconsistencies in demographic data.

### ✅ Task 2: DVC-Based Reproducibility

- Integrated **DVC** for dataset versioning and pipeline tracking.
- Enabled transparent collaboration, rollback, and audit readiness.

### ✅ Task 3: Hypothesis Testing

| Hypothesis | Result | Business Insight |
|------------|--------|------------------|
| Risk difference across provinces? | ❌ Rejected | Gauteng has higher risk. |
| Risk between zip codes? | ✅ Retained | No evidence of zip-specific risk. |
| Margin difference by zip? | ❌ Rejected | Margin varies by region. |
| Gender-based risk difference? | ✅ Retained | Gender does not impact risk. |

### ✅ Task 4: Predictive Modeling & Risk-Based Pricing

- Built two models:
  - **Claim Severity Regression (XGBoost)**: RMSE ≈ 9,765
  - **Claim Occurrence Classifier**: AUC ≈ 0.80
- Combined outputs into a pricing formula:
  
  \[
  \text{Premium} = \mathbb{P}(\text{claim}) \times \mathbb{E}[\text{TotalClaims}] + \text{Expense} + \text{Margin}
  \]

### ✅ Model Interpretability with SHAP

- High-impact features:
  - `RegistrationYear`, `Kilowatts`, `MainCrestaZone`
- Low-impact features:
  - `Gender`, `NumberOfDoors`, `NewVehicle`

> 📌 _Conclusion_: Dynamic pricing should prioritize **vehicle age**, **power**, and **geography** over personal demographics.

---

## 🔍 Key Business Insights

- **Regional Adjustment Needed**: Provinces like Gauteng require higher premiums.
- **Vehicle Features Matter Most**: Prioritize `RegistrationYear` and `Kilowatts`.
- **Ethical Pricing**: Avoid using gender or marital status.
- **Premium Optimization**: Combine risk probability and claim severity for accurate premium setting.

---

## 🛠️ Technologies Used

- Python 3.10+
- Pandas, NumPy, Scikit-learn, XGBoost
- Matplotlib, Seaborn
- SHAP for model explainability
- MLflow for experiment tracking
- DVC for data versioning

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ACIS-Car-Insurance-Claim-Analysis.git
cd ACIS-Car-Insurance-Claim-Analysis
2. Create a virtual environment and install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Pull datasets via DVC (if using remote storage)
bash
Copy
Edit
dvc pull
4. Run analysis notebooks
bash
Copy
Edit
cd notebooks
jupyter notebook
📈 Future Work
Add fraud detection layer for claims.

Use telematics data (driving behavior) for enhanced pricing.

Deploy models in production with REST API and dashboards.

👤 Author
Filimon Hailemariam
Data Scientist | ACIS Risk Analytics Consultant





