# Insurance-Risk-Analytics-Predictive-Modelling
real insurance data to uncover low-risk segments and build smart models that optimize premiums.
# Insurance Risk Analytics – Predictive Modelling

This project is part of a data engineering challenge focused on exploring and modeling insurance risk using historical car insurance data. The key objectives include exploring customer behavior, detecting anomalies, and ensuring that the data pipeline is auditable and reproducible using Data Version Control (DVC).

---

## 🧪 Task 1: Exploratory Data Analysis (EDA)

### Objectives:
- Understand the structure and distribution of the dataset.
- Detect trends across customer demographics and geography.
- Identify anomalies or potential risk segments.

### Key Visual Insights:
1. **Premium Distribution by Province and Cover Type**
   - Shows regional differences in insurance coverage.
   - Helps identify areas with high risk or high-value customers.

2. **Box Plots for Outlier Detection**
   - Highlights outliers in premium and claim-related metrics.
   - Supports risk segmentation and fraud detection.

3. **Categorical Variable Distributions**
   - Distribution of attributes like vehicle make, customer segment, and fuel type.
   - Detects imbalance or common risk features.

> _Screenshots of the visualizations are located in the `images/` folder._

---

## 📦 Task 2: Reproducible Data Pipeline with DVC

### Why DVC?
In regulated industries like insurance, **traceability and reproducibility** are essential. DVC enables versioning for both data and models, just like Git does for code.

### ✅ Steps Completed:

1. **Initialize DVC in the project:**
   ```bash
   dvc init
Create a local storage path:

bash
Copy
Edit
mkdir C:/Users/User/Desktop/Week3/dvc-storage
Configure remote storage:

bash
Copy
Edit
dvc remote add -d localstorage C:/Users/User/Desktop/Week3/dvc-storage
Track dataset using DVC:

bash
Copy
Edit
dvc add data/MachineLearningRating_v3.txt
git add data/MachineLearningRating_v3.txt.dvc .gitignore
git commit -m "Add insurance dataset with DVC tracking"
Push data to local remote:

bash
Copy
Edit
dvc push
Enable auto-staging (optional):

bash
Copy
Edit
dvc config core.autostage true
🔄 Git Workflow
Created and pushed a new feature branch:

bash
Copy
Edit
git checkout -b task-2
git push origin task-2
Merge into main via pull request after review.

📁 Directory Structure
bash
Copy
Edit
Insurance-Risk-Analytics-Predictive-Modelling/
│
├── data/                          # Raw datasets
├── dvc-storage/                   # DVC remote (local)
├── .dvc/                          # DVC configuration files
├── .gitignore
├── images/                        # Screenshots of EDA visuals
├── src/                           # Scripts for preprocessing, modeling, etc.
├── README.md
└── ...
📝 Next Steps
Build modeling pipeline (logistic regression, decision tree, etc.).

Integrate model tracking with DVC or MLFlow.

Deploy insights into a dashboard for stakeholders.

📌 Requirements
Python 3.8+

DVC

Git

Pandas, Seaborn, Matplotlib, Plotly (for EDA)