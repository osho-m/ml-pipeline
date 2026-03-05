# ML Pipeline using Scikit-Learn

An end-to-end **Machine Learning pipeline** built with Scikit-Learn to predict item outlet sales using the BigMart Sales dataset. The project covers the complete ML workflow — from data preprocessing to model evaluation — comparing Linear Regression and Random Forest Regressor.

---

## 📌 Project Overview

This project demonstrates how to build a structured ML pipeline including:
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Model training and evaluation
- Comparing multiple regression models using RMSE

**Target Variable:** `Item_Outlet_Sales` (continuous — regression problem)

---

## 📊 Dataset

**BigMart Sales Dataset** (`train_v9rqX0R.csv`)

| Feature | Description |
|---|---|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Weight of product |
| `Item_Fat_Content` | Fat content category |
| `Item_Visibility` | Display area % in store |
| `Item_Type` | Category of product |
| `Item_MRP` | Maximum Retail Price |
| `Outlet_Identifier` | Unique store ID |
| `Outlet_Establishment_Year` | Year store was established |
| `Outlet_Size` | Size of the store |
| `Outlet_Location_Type` | City tier of the store |
| `Outlet_Type` | Type of outlet |
| `Item_Outlet_Sales` | **Target** — sales of the product |

**Missing Values:**
- `Item_Weight` — 1,463 missing → imputed with **mean**
- `Outlet_Size` — 2,410 missing → imputed with **mode**

---

## 🔧 Pipeline Steps

### 1. Data Loading & Exploration
```python
train_data = pd.read_csv('train_v9rqX0R.csv')
train_data.isna().sum()
```

### 2. Missing Value Imputation
```python
train_data.Item_Weight.fillna(train_data.Item_Weight.mean(), inplace=True)
train_data.Outlet_Size.fillna(train_data.Outlet_Size.mode()[0], inplace=True)
```

### 3. Categorical Encoding (One-Hot Encoding)
Applied to: `Item_Fat_Content`, `Item_Type`, `Outlet_Identifier`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`

### 4. Feature Scaling (StandardScaler)
Applied to: `Item_MRP`

### 5. Train-Test Split
- **Train:** 6,392 samples
- **Test:** 2,131 samples
- Split ratio: 75% / 25%

---

## 🤖 Models & Results

| Model | RMSE (Train) | RMSE (Test) |
|---|---|---|
| Linear Regression | 1120.42 | 1148.71 |
| Random Forest Regressor (max_depth=10) | **895.90** | **1119.78** |

The **Random Forest Regressor** outperforms Linear Regression on both train and test sets, indicating better capture of non-linear relationships in the data.

---

## 🛠️ Tech Stack

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- category_encoders

---

## 📁 Project Structure

```
ml-pipeline/
│
└── ML Pipeline using Scikit Learn.ipynb   # Full pipeline notebook
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib scikit-learn category_encoders
```

### Run the Notebook

```bash
jupyter notebook "ML Pipeline using Scikit Learn.ipynb"
```

> **Note:** Place `train_v9rqX0R.csv` in the same directory as the notebook before running.

---

## 🔍 Key Concepts Demonstrated

- **Missing Value Imputation** — Mean for numerical, mode for categorical
- **One-Hot Encoding** — Using `category_encoders` library
- **Feature Scaling** — StandardScaler on continuous features
- **Model Comparison** — Linear Regression vs Random Forest using RMSE
- **Train-Test Split** — Proper holdout evaluation

---

## 👤 Author

**Osho Muralidaran**  
[LinkedIn](https://www.linkedin.com/in/osho-m) · [GitHub](https://github.com/osho-m)
