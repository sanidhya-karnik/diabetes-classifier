# Diabetes-Classifier
## Classification Model from scratch

This repository contains a collection of machine learning models implemented from scratch in Python. The highlight of the project is a **Support Vector Machine (SVM)** classifier using **Random Fourier Features (RFF)** for kernel approximation, enabling efficient non-linear classification. The models are applied for a binary classification task: To predict if a person has Diabetes or not based on Health Metrics.

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── Diabetes_Classifier_EDA.ipynb           # Exploratory Data Analysis and Statistical Analysis
│   ├── diabetes_classifier_v_0_0_3.ipynb       # Initial Analysis
│   ├── discrete_naive_bayes.ipynb              # Naive Bayes from scratch
│   ├── logistic_regression.ipynb               # Logistic Regression from scratch
│   ├── support_vector_machine.ipynb            # SVM Implementation and Hyperparameter Tuning
│   ├── svm_rff_tuned.ipynb                     # Final tuned SVM-RFF notebook
│   └── xg_boost_final.ipynb                    # Final tuned SVM-RFF notebook
│
├── src/
│   ├── __init__.py
│   ├── load_data.py                            # Dataset loading with validation
│   ├── preprocess.py                           # Feature scaling & splitting
│   └── svm_rff.py                              # SVM using Random Fourier Features
│
├── LICENSE
├── .gitignore
└── README.md
```

---

### 1. Clone the repository
```bash
git clone https://github.com/sanidhya-karnik/diabetes-classifier.git
cd diabetes-classifier
```

### 2. Install dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 3. Run the notebook
```bash
jupyter notebook notebooks/svm_rff_tuned.ipynb
```

---

## 🧩 Module Descriptions

### `src/load_data.py`
- Loads a dataset from CSV.
- Optionally validates required column names.

### `src/preprocess.py`
- Robust scaling of numerical features.
- Optional inversion of ordinal columns.
- Stratified train-test split.
- Outputs rounded, integer-valued features.

### `src/svm_rff.py`
- Custom SVM using Random Fourier Features (RFF).
- L2-regularized hinge loss optimization.
- Manual gradient descent with convergence checks.
- Evaluation tools: confusion matrix, ROC curve, and Recall.


---

## 📂 Dataset

The dataset used in this project is the **CDC Diabetes Health Indicators Dataset** from the UCI Machine Learning Repository.

📎 [CDC Diabetes Health Indicators – UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

---

## ⚙️ Model Hyperparameters

Tunable inside the notebook or via `svm_rff.py` class:
- `learning_rate`: Gradient descent step size
- `_lambda`: L2 regularization strength
- `epochs`: Maximum iterations
- `gamma`: Bandwidth parameter for the RBF kernel
- `D`: Number of random Fourier components

---
