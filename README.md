
# Loan Approval Prediction

A supervised machine learning project that predicts whether a home loan application will be **approved** or **rejected** based on applicant details — built using three classifiers and evaluated with a comprehensive set of metrics.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Workflow](#-workflow)
  - [Step 1 — Install Dependencies & Import Libraries](#step-1--install-dependencies--import-libraries)
  - [Step 2 — Load Dataset](#step-2--load-dataset)
  - [Step 3 — Exploratory Data Analysis (EDA)](#step-3--exploratory-data-analysis-eda)
  - [Step 4 — Data Preprocessing](#step-4--data-preprocessing)
  - [Step 5 — Model Building & Training](#step-5--model-building--training)
  - [Step 6 — Model Evaluation & Comparison](#step-6--model-evaluation--comparison)
  - [Step 7 — Predict on New Data](#step-7--predict-on-new-data)
  - [Step 8 — Conclusion](#step-8--conclusion)
- [Models Used](#-models-used)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)
- [Feature Importance](#-feature-importance)
- [Future Improvements](#-future-improvements)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)

---

## 🔍 Overview

Financial institutions receive thousands of loan applications daily. Manually reviewing each one is time-consuming and prone to human bias. This project uses **Machine Learning** to automate the loan approval decision by learning patterns from historical applicant data.

Three classification models are trained, cross-validated, and compared:

| Model | Type |
|---|---|
| Logistic Regression | Linear, probabilistic |
| Decision Tree | Non-linear, rule-based |
| Random Forest | Ensemble, tree-based |

The best-performing model is then used to predict the outcome for new applicants.

---

## 📦 Dataset

- **Source:** [Home Loan Approval Dataset — Kaggle](https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval)
- **Downloaded via:** `kagglehub`
- **Format:** CSV
- **Target Column:** `Loan_Status` (binary — Approved / Rejected)

**Key Features typically present in this dataset:**

| Feature | Description |
|---|---|
| `Gender` | Applicant gender |
| `Married` | Marital status |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employed or not |
| `ApplicantIncome` | Applicant's monthly income |
| `CoapplicantIncome` | Co-applicant's monthly income |
| `LoanAmount` | Loan amount requested (in thousands) |
| `Loan_Amount_Term` | Repayment term (in months) |
| `Credit_History` | Credit history meets guidelines (1/0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target** — Y (Approved) / N (Rejected) |

---

## 📁 Project Structure

```
Loan_Approval_Prediction/
│
├── Loan_Approval_Prediction.ipynb   # Main Jupyter Notebook
├── README.md                        # Project documentation
└── data/                            # Dataset (downloaded via kagglehub)
    └── *.csv
```

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| **Language** | Python 3.x |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Preprocessing** | `scikit-learn` (LabelEncoder, StandardScaler, SimpleImputer) |
| **Modeling** | `scikit-learn` (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier) |
| **Evaluation** | `scikit-learn` (accuracy, F1, AUC-ROC, confusion matrix, cross-validation) |
| **Dataset** | `kagglehub` |

---

## 🔄 Workflow

### Step 1 — Install Dependencies & Import Libraries

All required libraries are imported at the start:

- **Core:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
- **Preprocessing:** `train_test_split`, `LabelEncoder`, `StandardScaler`, `SimpleImputer`, `StratifiedKFold`
- **Models:** `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- **Evaluation:** `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `roc_curve`, `ConfusionMatrixDisplay`

```bash
pip install kagglehub scikit-learn pandas numpy matplotlib seaborn
```

---

### Step 2 — Load Dataset

The dataset is downloaded directly from Kaggle using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download('rishikeshkonapure/home-loan-approval')
df = pd.read_csv(os.path.join(path, csv_files[0]))
```

The notebook auto-detects the target column by searching for a column named `Loan_Status` (or similar), falling back to the last column if not found.

---

### Step 3 — Exploratory Data Analysis (EDA)

A thorough EDA is performed across four dimensions:

#### 3.1 Dataset Overview
- Shape, data types, and memory usage via `df.info()`
- Statistical summary via `df.describe(include='all')`

#### 3.2 Missing Values
- Missing count and percentage computed for all columns
- Visualized using a horizontal bar chart (highlighted in orange)

#### 3.3 Target Variable Distribution
- Class counts and percentages shown via bar chart and pie chart
- Helps identify any class imbalance between Approved and Rejected loans

#### 3.4 Categorical Features vs Loan Status
- Stacked/grouped bar charts for each categorical feature
- Shows approval rates across different groups (e.g., married vs. unmarried)

#### 3.5 Numerical Features Distribution
- Overlapping histograms by class (Approved / Rejected) for all numerical columns
- Helps spot distribution shifts between classes

#### 3.6 Correlation Heatmap
- Lower-triangle heatmap showing pairwise correlations between numerical features
- Helps identify multicollinearity and key predictors

---

### Step 4 — Data Preprocessing

Raw data is cleaned and transformed in four sub-steps:

#### 4.1 Drop ID Columns
Any column containing `id` in its name (e.g., `Loan_ID`) is dropped — it carries no predictive value.

#### 4.2 Handle Missing Values
| Column Type | Strategy |
|---|---|
| Categorical | Filled with **mode** (most frequent value) |
| Numerical | Filled with **median** (robust to outliers) |

#### 4.3 Encode Categorical Features
`LabelEncoder` is applied to all remaining object-type columns to convert them to integer labels.

#### 4.4 Feature / Target Split & Train-Test Split
- Features (X) and target (y) are separated
- An **80/20 stratified split** is used to preserve class ratios:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y
  )
  ```

#### 4.5 Feature Scaling
`StandardScaler` is applied to the training and test sets. Scaling is only required for Logistic Regression; tree-based models use unscaled data.

---

### Step 5 — Model Building & Training

Three classifiers are trained independently:

#### 5.1 Logistic Regression
- A **linear probabilistic model** using the sigmoid function
- Trained on **scaled** features (`X_train_sc`)
- Hyperparameters: `max_iter=1000`, `random_state=42`

#### 5.2 Decision Tree Classifier
- A **rule-based, non-linear model** that recursively splits data on feature thresholds
- Trained on **unscaled** features
- Hyperparameters: `max_depth=5`, `random_state=42`
- Visualized as a tree diagram (depth ≤ 3 displayed)

#### 5.3 Random Forest Classifier
- An **ensemble of Decision Trees** trained on bootstrapped subsets (bagging)
- Reduces variance and overfitting compared to a single tree
- Trained on **unscaled** features
- Hyperparameters: `n_estimators=100`, `max_depth=6`, `random_state=42`, `n_jobs=-1`

---

### Step 6 — Model Evaluation & Comparison

#### 6.1 Confusion Matrices
Side-by-side confusion matrices for all three models, visualized using `ConfusionMatrixDisplay`.

#### 6.2 ROC Curves
All three ROC curves plotted together with AUC scores in the legend. A random classifier baseline (diagonal) is included for reference.

#### 6.3 5-Fold Stratified Cross-Validation
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
Mean ± standard deviation of accuracy scores are reported for each model. This ensures the evaluation is not over-optimistic due to a lucky train/test split.

#### 6.4 Feature Importance (Random Forest)
A horizontal bar chart showing which features contribute most to predictions. The top feature is highlighted in a distinct color.

#### 6.5 Model Comparison Summary Table
A consolidated table of all metrics:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted positives, how many are actually positive |
| **Recall** | Of actual positives, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC-ROC** | Discriminative power across all classification thresholds |
| **CV Mean** | Mean cross-validation accuracy |

A bar chart visually compares model accuracy scores.

---

### Step 7 — Predict on New Data

The notebook automatically identifies the best model (by test accuracy) and uses it to predict a sample applicant's outcome:

```python
best_model_name = summary['Accuracy'].idxmax()
# Outputs: ✅ Approved (probability: XX%) or ❌ Rejected
```

The sample is taken from the held-out test set, and prediction probability is also displayed.

---

### Step 8 — Conclusion

A formatted summary is printed at the end of the notebook, including:
- All key preprocessing and training steps
- Best model name, accuracy, and AUC-ROC
- Top 3 most important features from Random Forest
- Suggestions for future improvements

---

## 🤖 Models Used

### Logistic Regression
A simple and interpretable linear classifier that models the **log-odds** of the target class. Works best when features are linearly separable and benefits significantly from feature scaling.

### Decision Tree
A flowchart-like model that splits the dataset on feature thresholds. Easy to visualize and interpret, but prone to overfitting without depth constraints (`max_depth=5` used here).

### Random Forest
An ensemble method that trains multiple Decision Trees on random data subsets and averages their predictions. More robust and generalizable than a single tree — typically the strongest performer among the three.

---

## 📏 Evaluation Metrics

| Metric | Formula | Why It Matters |
|---|---|---|
| Accuracy | (TP + TN) / Total | Simple overall performance measure |
| Precision | TP / (TP + FP) | Measures false positive rate — avoids approving bad loans |
| Recall | TP / (TP + FN) | Measures false negative rate — avoids rejecting good loans |
| F1-Score | 2 × (P × R) / (P + R) | Balances precision and recall |
| AUC-ROC | Area under ROC curve | Measures ranking quality at all thresholds |
| CV Accuracy | Mean of k-fold scores | Measures generalization beyond a single split |

---

## 📊 Results

The model comparison summary is auto-generated within the notebook. Typical expected results on this dataset:

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | ~80–82% | ~0.82–0.85 |
| Decision Tree | ~76–80% | ~0.74–0.80 |
| Random Forest | **~82–85%** | **~0.85–0.88** |

> ⚠️ Exact values depend on the dataset split and random seed. Run the notebook to see your results.

**Random Forest typically achieves the highest performance** due to its ensemble nature and resistance to overfitting.

---

## 🔑 Feature Importance

Based on Random Forest feature importances, the most influential factors in loan approval decisions are typically:

1. **Credit History** — strongest predictor by far
2. **ApplicantIncome** — higher income increases approval chances
3. **LoanAmount** — affects debt-to-income ratio
4. **CoapplicantIncome** — combined income improves eligibility
5. **Loan_Amount_Term** — repayment duration affects risk

---

## 🚀 Future Improvements

| Improvement | Description |
|---|---|
| Hyperparameter Tuning | Use `GridSearchCV` or `RandomizedSearchCV` to find optimal model parameters |
| Advanced Models | Try `XGBoost`, `LightGBM`, or `SVM` for potentially higher accuracy |
| Class Imbalance Handling | Apply `SMOTE` or class-weight adjustments if the dataset is imbalanced |
| Feature Engineering | Create interaction features (e.g., total income, EMI ratio) |
| Deployment | Package the best model as a REST API using **Flask** or **FastAPI** |
| Web App | Build an interactive front-end using **Streamlit** for live predictions |
| SHAP Explainability | Use SHAP values for granular, per-prediction explainability |

---

## ⚡ Getting Started

### 1. Clone or Download the Repository
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install kagglehub scikit-learn pandas numpy matplotlib seaborn jupyter
```

### 3. Set Up Kaggle API Credentials
To download the dataset automatically via `kagglehub`, you need a Kaggle account and API token:

1. Go to [kaggle.com](https://www.kaggle.com) → Account → Create New Token
2. Place the downloaded `kaggle.json` in `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json` (Linux/macOS)

### 4. Launch the Notebook
```bash
jupyter notebook Loan_Approval_Prediction.ipynb
```

Run all cells from top to bottom — the notebook handles everything automatically.

---

## 📦 Requirements

```
kagglehub
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

> Python 3.8+ recommended.

---

## 📄 License

This project is intended for educational purposes. The dataset is sourced from Kaggle and subject to its original license terms.

---
