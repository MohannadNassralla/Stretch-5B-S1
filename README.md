# Hyperparameter Tuning & Nested Cross-Validation
**Honors Track - Module 5 Week B Stretch Assignment**

## Project Overview
This project explores systematic hyperparameter optimization and the critical importance of unbiased model evaluation. While `GridSearchCV` is a powerful tool for finding optimal model parameters, it often yields "optimistically biased" scores. This assignment implements **Nested Cross-Validation** to calculate the true selection bias and obtain an honest estimate of model performance on unseen data.

## Methodology

### Part 1: GridSearchCV (The Inner Loop)
We performed an exhaustive search over a parameter grid for a **Random Forest Classifier** using:
- **Scoring Metric:** F1-Score (to handle class imbalance in telecom churn data).
- **Cross-Validation:** 5-fold Stratified CV.
- **Parameters Tuned:** `n_estimators`, `max_depth`, and `min_samples_split`.



### Part 2: Nested Cross-Validation (The Outer Loop)
To account for selection bias, we implemented a nested CV structure:
1. **Inner Loop:** Performs `GridSearchCV` to select the best hyperparameters.
2. **Outer Loop:** Evaluates the entire tuning process on a separate test fold that was not involved in the parameter selection.

We compared two model families: **Random Forest** vs. **Decision Tree**.

---

##  Results & Analysis

### Part 1 Analysis: Model Complexity
* **Impactful Hyperparameters:** `max_depth` typically showed the most significant impact on the F1 score. 
* **The "Sweet Spot":** Performance often plateaus after a certain depth (e.g., depth 10 or 20), suggesting that adding more complexity doesn't necessarily yield better generalization.
* **Risk Profile:** If the F1 score continues to climb on training folds but drops on test folds, the model is at risk of **overfitting**.
* **Expansion:** Based on the heatmap, I would consider expanding the grid to include `min_samples_leaf` or `max_features` to further regularize the model if overfitting is detected.

### Part 2 Analysis: Selection Bias (Nested CV)

| Metric | Random Forest | Decision Tree |
| :--- | :--- | :--- |
| Inner best_score_ (Biased) | [Your Value] | [Your Value] |
| Outer Nested CV (Honest) | [Your Value] | [Your Value] |
| **Gap (Selection Bias)** | **[Your Gap]** | **[Your Gap]** |

#### Why the Gap Exists:
The **Decision Tree** family typically shows a larger gap between inner and outer scores. This is because Decision Trees have **high variance**; they are highly sensitive to the specific data they are trained on. Consequently, the "best" hyperparameters chosen by the inner loop are often over-specialized to those specific folds. 

**Random Forest** reduces this gap because it is an ensemble method that reduces variance through bagging. The choice of hyperparameters is therefore more stable across different folds.



#### The "Golden Rule" of ML:
Just as we use a held-out test set to evaluate a final model, **Nested CV** acts as the held-out test set for the *tuning process*. The `GridSearchCV.best_score_` is not fully trustworthy because the data "leaked" into the decision-making process of picking parameters. Nested CV provides the only honest assessment of how the model will perform on truly new data.

---

##  How to Run
1. Clone the repository.
2. Ensure you have the required libraries: `pandas`, `sklearn`, `seaborn`, `matplotlib`.
3. Run the main script:
   ```bash
   python hyperparameter_tuning_nested_cv.py