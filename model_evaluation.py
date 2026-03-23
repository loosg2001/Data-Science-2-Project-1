import statsmodels.api as sm
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from get_qof import get_qof
from get_cv_qof import get_cv_qof
from lambda_tuning import tune_ridge_lasso_alpha, tune_box_cox_lambda
from save_plots import save_sorted_plot


def lin_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name):
    """
    Evaluates a standard Multiple Linear Regression model.
    
    Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 
    5-Fold Cross-Validation. Automatically generates and saves high-resolution 
    plots comparing actual vs. predicted values.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The full target response vector.
        X_test (pd.DataFrame): The testing input feature matrix (Out-of-Sample).
        X_train (pd.DataFrame): The training input feature matrix (Out-of-Sample).
        y_test (pd.Series): The testing target response vector.
        y_train (pd.Series): The training target response vector.
        data_name (str): The name of the dataset (used for plot titles).
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(y, X).fit()
    yp_is = model_is.predict(X)
    k = X.shape[1]  # Extract number of predictors
    
    qof_is = get_qof(y, yp_is, k, model_is)
    save_sorted_plot(y, yp_is, data_name, folder_name, "Linear Regression", "Reg", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(y_train, X_train).fit()
    yp_oos = model_oos.predict(X_test)
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test, yp_oos, k, None)
    save_sorted_plot(y_test, yp_oos, data_name, folder_name, "Linear Regression", "Reg", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="linreg")

    return (qof_is, qof_oos, cv_stats)


def ridge_reg(X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, data_name, folder_name):
    """
    Evaluates a Ridge Regression model.
    
    First tunes the regularization hyperparameter (alpha) using a grid search. 
    Then performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), 
    and 5-Fold Cross-Validation. Automatically generates and saves plots.

    Args:
        X_z_score_is (pd.DataFrame): The standardized full input feature matrix.
        y_centered_is (pd.Series): The centered full target response vector.
        X_test_z_score (pd.DataFrame): The standardized testing feature matrix.
        X_train_z_score (pd.DataFrame): The standardized training feature matrix.
        y_test_centered (pd.Series): The centered testing target vector.
        y_train_centered (pd.Series): The centered training target vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
            - best_alpha (float): The optimal regularization parameter found via grid search.
    """
    # Tune regularization hyperparameter first
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X_z_score_is, y_centered_is, method='ridge')
    
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(y_centered_is, X_z_score_is).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_is = model_is.predict(X_z_score_is)
    k = X_z_score_is.shape[1]
    
    qof_is = get_qof(y_centered_is, yp_is, k, model_is)
    save_sorted_plot(y_centered_is, yp_is, data_name, folder_name, "Ridge Regression", "Ridge", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(y_train_centered, X_train_z_score).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_oos = model_oos.predict(X_test_z_score)
    k = X_train_z_score.shape[1]
    
    qof_oos = get_qof(y_test_centered, yp_oos, k, None)
    save_sorted_plot(y_test_centered, yp_oos, data_name, folder_name, "Ridge Regression", "Ridge", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X_z_score_is, y_centered_is, method="ridge", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha)


def lasso_reg(X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, data_name, folder_name):
    """
    Evaluates a Lasso Regression model.
    
    First tunes the regularization hyperparameter (alpha) using a grid search. 
    Then performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), 
    and 5-Fold Cross-Validation. Automatically generates and saves plots.

    Args:
        X_z_score_is (pd.DataFrame): The standardized full input feature matrix.
        y_centered_is (pd.Series): The centered full target response vector.
        X_test_z_score (pd.DataFrame): The standardized testing feature matrix.
        X_train_z_score (pd.DataFrame): The standardized training feature matrix.
        y_test_centered (pd.Series): The centered testing target vector.
        y_train_centered (pd.Series): The centered training target vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
            - best_alpha (float): The optimal regularization parameter found via grid search.
            - non_zero_coeffs (list of str): The names of the features that survived the L1 penalty. 
              These are derived from the final model trained on the full dataset using the 
              `best_alpha`, excluding the intercept, and accounting for floating-point inaccuracies 
              (e.g., applying a zero-threshold of 1e-6).
    """
    # Tune regularization hyperparameter first
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X_z_score_is, y_centered_is, method='lasso')
    
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(y_centered_is, X_z_score_is).fit_regularized(alpha=best_alpha, L1_wt=1.0)
    yp_is = model_is.predict(X_z_score_is)
    k = X_z_score_is.shape[1]
    
    qof_is = get_qof(y_centered_is, yp_is, k, model_is)
    save_sorted_plot(y_centered_is, yp_is, data_name, folder_name, "Lasso", "Lasso", False)

    # --- Extract the non-zero features ---
    # Extract all the coefficients
    coefficients = model_is.params

    # Get the column names for the non-zero coefficients
    # using a small threshold (1e-6) to account for floating-point inaccuracies
    non_zero_coeffs = coefficients[coefficients.abs() > 1e-6].index.tolist()

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(y_train_centered, X_train_z_score).fit_regularized(alpha=best_alpha, L1_wt=1.0)
    yp_oos = model_oos.predict(X_test_z_score)
    k = X_train_z_score.shape[1]
    
    qof_oos = get_qof(y_test_centered, yp_oos, k, None)
    save_sorted_plot(y_test_centered, yp_oos, data_name, folder_name, "Lasso", "Lasso", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X_z_score_is, y_centered_is, method="lasso", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha, non_zero_coeffs)


def sqrt_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name):
    """
    Evaluates a Transformed Regression model applying a Square Root transformation.
    
    Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 
    5-Fold Cross-Validation. Automatically generates and saves plots comparing 
    actual vs. predicted values.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The full target response vector.
        X_test (pd.DataFrame): The testing input feature matrix.
        X_train (pd.DataFrame): The training input feature matrix.
        y_test (pd.Series): The testing target response vector.
        y_train (pd.Series): The training target response vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(np.sqrt(y), X).fit()
    yp_is = (model_is.predict(X)) ** 2
    k = X.shape[1]
    
    qof_is = get_qof(y, yp_is, k, model_is)
    save_sorted_plot(y, yp_is, data_name, folder_name, "Sqrt Transformation", "Sqrt", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(np.sqrt(y_train), X_train).fit()
    yp_oos = (model_oos.predict(X_test)) ** 2
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test, yp_oos, k, None)
    save_sorted_plot(y_test, yp_oos, data_name, folder_name, "Sqrt Transformation", "Sqrt", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="sqrt")

    return (qof_is, qof_oos, cv_stats)


def log1p_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name):
    """
    Evaluates a Transformed Regression model applying a Log1p transformation.
    
    Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 
    5-Fold Cross-Validation. Automatically generates and saves plots comparing 
    actual vs. predicted values.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The full target response vector.
        X_test (pd.DataFrame): The testing input feature matrix.
        X_train (pd.DataFrame): The training input feature matrix.
        y_test (pd.Series): The testing target response vector.
        y_train (pd.Series): The training target response vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
    """
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(np.log1p(y), X).fit()
    yp_is = np.expm1(model_is.predict(X))
    k = X.shape[1]
    
    qof_is = get_qof(y, yp_is, k, model_is)
    save_sorted_plot(y, yp_is, data_name, folder_name, "Log1p Transformation", "Log1p", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(np.log1p(y_train), X_train).fit()
    yp_oos = np.expm1(model_oos.predict(X_test))
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test, yp_oos, k, None)
    save_sorted_plot(y_test, yp_oos, data_name, folder_name, "Log1p Transformation", "Log1p", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X, y, method="log1p")

    return (qof_is, qof_oos, cv_stats)


def boxcox_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name):
    """
    Evaluates a Transformed Regression model utilizing the Box-Cox transformation.
    
    First tunes the transformation lambda power using a grid search. Then performs 
    In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 5-Fold 
    Cross-Validation. Automatically generates and saves plots.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The full target response vector.
        X_test (pd.DataFrame): The testing input feature matrix.
        X_train (pd.DataFrame): The training input feature matrix.
        y_test (pd.Series): The testing target response vector.
        y_train (pd.Series): The training target response vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
            - best_lambda (float): The optimal lambda parameter found via grid search.
    """
    # Tune Box-Cox transformation parameter first
    (best_lambda, best_r_sq) = tune_box_cox_lambda(X, y)

    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(boxcox(y, best_lambda), X).fit()
    yp_is = inv_boxcox(model_is.predict(X), best_lambda)
    k = X.shape[1]
    
    qof_is = get_qof(y, yp_is, k, model_is)
    save_sorted_plot(y, yp_is, data_name, folder_name, "Box-Cox Transformation", "BoxCox", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(boxcox(y_train, best_lambda), X_train).fit()
    yp_oos = inv_boxcox(model_oos.predict(X_test), best_lambda)
    k = X_train.shape[1]
    
    qof_oos = get_qof(y_test, yp_oos, k, None)
    save_sorted_plot(y_test, yp_oos, data_name, folder_name, "Box-Cox Transformation", "BoxCox", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    # Added best_lambda mapping (bugfix)
    cv_stats = get_cv_qof(X, y, method="boxcox", lambda_=best_lambda)

    return (qof_is, qof_oos, cv_stats, best_lambda)


def order2_reg(X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, data_name, folder_name):
    """
    Evaluates an Order-2 (quadratic/polynomial) Ridge Regression model.
    
    First tunes the regularization hyperparameter (alpha) using a grid search. 
    Then performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), 
    and 5-Fold Cross-Validation. Automatically generates and saves plots.

    Args:
        X_z_score_is (pd.DataFrame): The standardized Order-2 input feature matrix.
        y_centered_is (pd.Series): The centered full target response vector.
        X_test_z_score (pd.DataFrame): The standardized Order-2 testing feature matrix.
        X_train_z_score (pd.DataFrame): The standardized Order-2 training feature matrix.
        y_test_centered (pd.Series): The centered testing target vector.
        y_train_centered (pd.Series): The centered training target vector.
        data_name (str): The name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        tuple: A 5-element tuple containing the evaluation results:
            - qof_is: The Quality of Fit metrics for the full dataset (In-Sample).
            - qof_oos: The Quality of Fit metrics for the 80-20 test split (Out-of-Sample).
            - cv_stats: The cross-validation evaluation metrics.
            - best_alpha (float): The optimal regularization parameter found via grid search.
    """
    # Tune regularization hyperparameter first
    (best_alpha, best_r_sq) = tune_ridge_lasso_alpha(X_z_score_is, y_centered_is, method='ridge')
    
    # ==========================================
    # --- In-Sample Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ In-Sample ------------")
    print("------------------------------------")
    model_is = sm.OLS(y_centered_is, X_z_score_is).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_is = model_is.predict(X_z_score_is)
    k = X_z_score_is.shape[1]
    
    qof_is = get_qof(y_centered_is, yp_is, k, model_is)
    save_sorted_plot(y_centered_is, yp_is, data_name, folder_name, "Order 2 Regression", "Order2Reg", False)

    # ==========================================
    # --- 80-20 Split Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"----------- 80-20 Split -----------")
    print("------------------------------------")
    model_oos = sm.OLS(y_train_centered, X_train_z_score).fit_regularized(alpha=best_alpha, L1_wt=0.0)
    yp_oos = model_oos.predict(X_test_z_score)
    k = X_train_z_score.shape[1]
    
    qof_oos = get_qof(y_test_centered, yp_oos, k, None)
    save_sorted_plot(y_test_centered, yp_oos, data_name, folder_name, "Order 2 Regression", "Order2Reg", True)

    # ==========================================
    # --- 5-Fold Cross-Validation ---
    # ==========================================
    print("------------------------------------")
    print(f"------------ 5-fold CV ------------")
    print("------------------------------------")
    cv_stats = get_cv_qof(X_z_score_is, y_centered_is, method="ridge", alpha=best_alpha)

    return (qof_is, qof_oos, cv_stats, best_alpha)