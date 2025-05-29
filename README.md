# Credit Risk Classification with Random Forest

This project explores credit risk prediction using a real-world dataset from Kaggle. I cleaned the data, did some exploratory analysis, and built a Random Forest classifier to predict loan default. I also experimented with hyperparameter tuning using both Randomized Search and Bayesian Optimization to improve model performance.

## Dataset

- **Source:** [Kaggle - Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Size:** ~32,000 observations
- **Features include:**
  - Personal info like `person_age`, `person_income`, `person_home_ownership`
  - Loan info such as `loan_amnt`, `loan_int_rate`, `loan_grade`
  - Credit history variables: `cb_person_default_on_file`, `cb_person_cred_hist_length`
  - Target variable: `loan_status` (1 = default, 0 = repaid)

## Data Cleaning

I cleaned the data by:
- Removing duplicates and outliers (e.g. people with unrealistic ages or employment lengths)
- Filling in missing `loan_int_rate` values using the median interest rate for each `loan_grade`
- Dropping rows with missing `person_emp_length` values
- Binning `person_age` into ranges (inspired by FICO charts)
- Converting relevant columns to `category` dtype

## Exploratory Data Analysis

I used one-hot encoding for some categorical variables and mapped others to numeric codes. Then I created a correlation heatmap to get a better sense of which features were most related to loan status.

## Model Training

I used a `RandomForestClassifier` from scikit-learn with balanced class weights to deal with the class imbalance (fewer defaults than repayments). I standardized the numeric features and split the data 80/20 for training and testing.

**Baseline performance:**
- Accuracy: ~93%
- F1 score on the default class: ~0.80

## Hyperparameter Tuning

### Randomized Search

I used `RandomizedSearchCV` with 5-fold cross-validation and ran 50 iterations. Parameters tuned:
- `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`

### Bayesian Optimization

I also tried `optuna` for more efficient tuning, running 50 trials and optimizing for the F1 score. The final model was trained using the best parameters from the study.

**Best result (Bayesian Optimization):**
- Accuracy: ~93%
- F1 score (default class): ~0.81

## Results Summary

| Metric           | Random Search | Bayesian Optimization |
|------------------|----------------|------------------------|
| Accuracy         | 93%            | 93%                    |
| F1 (Default = 1) | 0.81           | 0.81                   |
| Recall (Default) | 0.71           | 0.71                   |

## Libraries & Tools

- Python 3.12
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for modeling
- `optuna` for Bayesian optimization
- `tqdm` for progress tracking

## What I'd Do Next

- Try interpretability tools like SHAP or LIME to understand what the model's learning
- Test out XGBoost or LightGBM for comparison
- Maybe add time-based borrower behavior features if available
