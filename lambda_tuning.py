import numpy as np
from get_cv_qof import get_cv_qof

def tune_ridge_lasso_alpha(X, y, method='ridge'):
    """
    Tunes the regularization hyperparameter alpha for Ridge or Lasso regression.
    
    This function performs a multi-stage grid search to find the optimal alpha value
    that maximizes the mean R-squared (R^2) score via 5-fold cross-validation. The search
    progresses through:
    1. A wide-range logarithmic coarse search.
    2. An intermediate-range search relative to the first-stage winner.
    3. A fine-grained linear search tailored to the magnitude of the current best alpha.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target response vector.
        method (str): The regression method to tune ('ridge' or 'lasso'). Defaults to 'ridge'.

    Returns:
        tuple: A tuple containing `(best_alpha, best_r_sq)`.
    """
    print("-----------------------------------")
    print(f"--- Tuning alpha for {method} ---")
    print("-----------------------------------")
    
    if method not in ['ridge', 'lasso']:
        raise ValueError(f"method must be one of 'ridge' or 'lasso'. Received {method}")

    best_alpha = 0.0
    best_r_sq = -float('inf')
    
    # ==========================================
    # --- STAGE 1: Coarse Logarithmic Search ---
    # ==========================================
    alpha_list_0 = [10**i for i in range(-7, 7)]
    
    for alpha in alpha_list_0:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        cur_r_sq_mean = np.mean(qof[0])  # Extract the mean R-squared value across all folds

        # Update best parameters if the current mean improves the score
        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = alpha

    # ==========================================
    # --- STAGE 2: Intermediate Search ---
    # ==========================================
    # Shift the search space based on the best alpha found in Stage 1
    left_1 = best_alpha / 10
    alpha_list_1 = [left_1, 2*left_1, 4*left_1, 6*left_1, 8*left_1, 10*left_1, 20*left_1, 40*left_1, 60*left_1, 80*left_1, 100*left_1]

    for alpha in alpha_list_1:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        cur_r_sq_mean = np.mean(qof[0])

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = alpha

    # ==========================================
    # --- STAGE 3: Targeted Fine-Grained Search ---
    # ==========================================
    # Dynamically build a highly specific search grid based on the magnitude of the new best_alpha
    if best_alpha == left_1:
        alpha_0 = left_1 / 10
        alpha_list_2 = [alpha_0, 2*alpha_0, 4*alpha_0, 6*alpha_0, 8*alpha_0, 10*alpha_0, 20*alpha_0, 40*alpha_0, 60*alpha_0, 80*alpha_0, 100*alpha_0]
        
    elif best_alpha == 100 * left_1:
        alpha_0 = left_1 / 10
        alpha_list_2 = [alpha_0, 2*alpha_0, 4*alpha_0, 6*alpha_0, 8*alpha_0, 10*alpha_0, 20*alpha_0, 40*alpha_0, 60*alpha_0, 80*alpha_0, 100*alpha_0]
        
    elif best_alpha == 2 * left_1:
        dif_1 = left_1 / 4
        lam_0 = left_1
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (2 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]
        
    elif best_alpha == 10 * left_1:
        dif_1 = (2 * left_1) / 4
        lam_0 = best_alpha - (2 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (10 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]
        
    elif best_alpha == 20 * left_1:
        dif_1 = (10 * left_1) / 4
        lam_0 = best_alpha - (10 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        dif_2 = (20 * left_1) / 4
        lam_5 = lam_4 + dif_2; lam_6 = lam_5 + dif_2; lam_7 = lam_6 + dif_2; lam_8 = lam_7 + dif_2
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]
        
    elif best_alpha < 10 * left_1:
        dif_1 = (2 * left_1) / 4
        lam_0 = best_alpha - (2 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        lam_5 = lam_4 + dif_1; lam_6 = lam_5 + dif_1; lam_7 = lam_6 + dif_1; lam_8 = lam_7 + dif_1
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]    
              
    elif best_alpha > 10 * left_1:
        dif_1 = (20 * left_1) / 4
        lam_0 = best_alpha - (20 * left_1)
        lam_1 = lam_0 + dif_1; lam_2 = lam_1 + dif_1; lam_3 = lam_2 + dif_1; lam_4 = lam_3 + dif_1
        lam_5 = lam_4 + dif_1; lam_6 = lam_5 + dif_1; lam_7 = lam_6 + dif_1; lam_8 = lam_7 + dif_1
        alpha_list_2 = [lam_0, lam_1, lam_2, lam_3, lam_4, lam_5, lam_6, lam_7, lam_8]

    for alpha in alpha_list_2:
        qof = get_cv_qof(X, y, method=method, alpha=alpha)
        cur_r_sq_mean = np.mean(qof[0])

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_alpha = alpha

    print(f"Best Alpha for {method}: {best_alpha} (CV R2: {best_r_sq:.4f})")
    return (best_alpha, best_r_sq)


def tune_box_cox_lambda(X, y):
    """
    Tunes the power transformation lambda for a Box-Cox model.
    
    Iterates through a predefined grid of standard power transformations (e.g., inverse, 
    log, square root, square) to determine which lambda yields the highest mean 
    R-squared via 5-fold cross-validation.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target response vector to be transformed.

    Returns:
        tuple: A tuple containing `(best_lambda, best_r_sq)`.
    """
    print("-----------------------------------")
    print(f"--- Tuning lambda for Box-Cox ---")
    print("-----------------------------------")

    best_lambda = 0.0
    best_r_sq = -float('inf')

    # Standard candidate values for the Box-Cox transformation parameter
    lambdas = [-4, -3, -2, -1, -1.0/2, -1.0/3, -1.0/4, -1.0/5, -1.0/6, -1.0/7, -1.0/8, 
               -1.0/9, -1.0/10, -1.0/11, -1.0/12, 0, 1.0/12, 1.0/11, 1.0/10, 1.0/9, 
               1.0/8, 1.0/7, 1.0/6, 1.0/5, 1.0/4, 1.0/3, 1.0/2, 1, 2, 3, 4]

    for lambda_ in lambdas:
        qof = get_cv_qof(X, y, method='boxcox', lambda_=lambda_)
        cur_r_sq_mean = np.mean(qof[0])

        if cur_r_sq_mean > best_r_sq:
            best_r_sq = cur_r_sq_mean
            best_lambda = lambda_

    print(f"Best Lambda for Box-Cox: {best_lambda} (CV R2: {best_r_sq:.4f})")
    return (best_lambda, best_r_sq)