import numpy as np
from save_plots import save_rsq_plot, save_aic_bic_plot
from feature_selection_methods import forward_select_all, backward_eliminate_all, stepwise_selection

def feature_selection(key, X, y, X_z_score_is, y_centered_is, X_2_z_score_is, data_name, folder_name, ridge_alpha, lasso_alpha, boxcox_lambda, order2reg_alpha):
    """
    Performs feature selection (Forward, Backward, or Stepwise) across multiple regression models.
    
    Evaluates Linear, Ridge, Lasso, Square Root, Log1p, Box-Cox, and Order-2 Polynomial models.
    Calculates and tracks Quality of Fit (QoF) metrics at each stage of the selection process.
    Automatically generates and saves high-resolution plots for R^2 vs. number of features, 
    as well as AIC/BIC vs. number of features.

    Args:
        key (str): The feature selection method to use ('Forward', 'Backward', or 'Stepwise').
        X (pd.DataFrame): The standard input feature matrix.
        y (pd.Series): The target response vector.
        X_z_score_is (pd.DataFrame): The standardized input feature matrix.
        y_centered_is (pd.Series): The centered target response vector.
        X_2_z_score_is (pd.DataFrame): The standardized quadratic/interaction feature matrix (Order 2).
        data_name (str): The name of the dataset (used for plot titles and logging).
        folder_name (str): The directory path where output plots will be saved.
        ridge_alpha (float): The optimized regularization hyperparameter for Ridge Regression.
        lasso_alpha (float): The optimized regularization hyperparameter for Lasso Regression.
        boxcox_lambda (float): The optimized transformation hyperparameter for Box-Cox Regression.
        order2reg_alpha (float): The optimized regularization hyperparameter for Order 2 Regression.

    Returns:
        tuple: A tuple containing lists of the selected feature names for each model: 
               `(reg_features, ridge_features, lasso_features, sqrt_features, 
                 log1p_features, boxcox_features, order2reg_features)`.
    """
    if key == 'Forward':
        method = 'Forward Selection'
    elif key == 'Backward':
        method = 'Backward Elimination'
    elif key == 'Stepwise':
        method = 'Stepwise Selection'
    else:
        raise ValueError(f"key must be one of 'Forward', 'Backward', or 'Stepwise'. Received {key}")
    
    # ==========================================
    # --- Linear Regression ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Regression")
    print("------------------------------------")
    
    if key == 'Forward':
        (reg_features, reg_qof_list, reg_cv_stats_list) = forward_select_all(X, y, start_cols=None, method='linreg', metric=0)
    elif key == 'Backward':
        (reg_features, reg_qof_list, reg_cv_stats_list) = backward_eliminate_all(X, y, start_cols=None, method='linreg', metric=0)
    elif key == 'Stepwise':
        (reg_features, reg_qof_list, reg_cv_stats_list) = stepwise_selection(X, y, start_cols=None, method='linreg', metric=1)
    
    # Extract tracking metrics across the feature selection process
    reg_x = list(range(len(reg_features)))
    reg_r_sq = []
    reg_adj_r_sq = []
    reg_smape = []
    reg_r_sq_cv = []
    reg_aic = []
    reg_bic = []
    
    for i in reg_x:
        reg_r_sq.append(100 * reg_qof_list[i][0])              # Convert R^2 to percentage
        reg_adj_r_sq.append(100 * reg_qof_list[i][1])          # Convert Adj R^2 to percentage
        reg_smape.append(reg_qof_list[i][8])
        reg_r_sq_cv.append(100 * np.mean(reg_cv_stats_list[i][0])) # Convert Mean CV R^2 to percentage
        reg_aic.append(reg_qof_list[i][13])
        reg_bic.append(reg_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, reg_x, reg_r_sq, reg_adj_r_sq, reg_smape, reg_r_sq_cv, method, data_name, folder_name, "Linear Regression", "Reg")
    save_aic_bic_plot(key, reg_x, reg_aic, reg_bic, method, data_name, folder_name, "Linear Regression", "Reg")

    # ==========================================
    # --- Ridge Regression ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Ridge Regression")
    print("------------------------------------")
    
    if key == 'Forward':
        (ridge_features, ridge_qof_list, ridge_cv_stats_list) = forward_select_all(X_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=0, alpha=ridge_alpha)
    elif key == 'Backward':
        (ridge_features, ridge_qof_list, ridge_cv_stats_list) = backward_eliminate_all(X_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=0, alpha=ridge_alpha)
    elif key == 'Stepwise':
        (ridge_features, ridge_qof_list, ridge_cv_stats_list) = stepwise_selection(X_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=1, alpha=ridge_alpha)
    
    # Extract tracking metrics
    ridge_x = list(range(len(ridge_features)))
    ridge_r_sq = []
    ridge_adj_r_sq = []
    ridge_smape = []
    ridge_r_sq_cv = []
    ridge_aic = []
    ridge_bic = []
    
    for i in ridge_x:
        ridge_r_sq.append(100 * ridge_qof_list[i][0])
        ridge_adj_r_sq.append(100 * ridge_qof_list[i][1])
        ridge_smape.append(ridge_qof_list[i][8])
        ridge_r_sq_cv.append(100 * np.mean(ridge_cv_stats_list[i][0]))
        ridge_aic.append(ridge_qof_list[i][13])
        ridge_bic.append(ridge_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, ridge_x, ridge_r_sq, ridge_adj_r_sq, ridge_smape, ridge_r_sq_cv, method, data_name, folder_name, "Ridge Regression", "Ridge")
    save_aic_bic_plot(key, ridge_x, ridge_aic, ridge_bic, method, data_name, folder_name, "Ridge Regression", "Ridge")

    # ==========================================
    # --- Lasso Regression ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Lasso Regression")
    print("------------------------------------")
    
    if key == 'Forward':
        (lasso_features, lasso_qof_list, lasso_cv_stats_list) = forward_select_all(X_z_score_is, y_centered_is, start_cols=None, method='lasso', metric=0, alpha=lasso_alpha)
    elif key == 'Backward':
        (lasso_features, lasso_qof_list, lasso_cv_stats_list) = backward_eliminate_all(X_z_score_is, y_centered_is, start_cols=None, method='lasso', metric=0, alpha=lasso_alpha)
    elif key == 'Stepwise':
        (lasso_features, lasso_qof_list, lasso_cv_stats_list) = stepwise_selection(X_z_score_is, y_centered_is, start_cols=None, method='lasso', metric=1, alpha=lasso_alpha)
    
    # Extract tracking metrics
    lasso_x = list(range(len(lasso_features)))
    lasso_r_sq = []
    lasso_adj_r_sq = []
    lasso_smape = []
    lasso_r_sq_cv = []
    lasso_aic = []
    lasso_bic = []
    
    for i in lasso_x:
        lasso_r_sq.append(100 * lasso_qof_list[i][0])
        lasso_adj_r_sq.append(100 * lasso_qof_list[i][1])
        lasso_smape.append(lasso_qof_list[i][8])
        lasso_r_sq_cv.append(100 * np.mean(lasso_cv_stats_list[i][0]))
        lasso_aic.append(lasso_qof_list[i][13])
        lasso_bic.append(lasso_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, lasso_x, lasso_r_sq, lasso_adj_r_sq, lasso_smape, lasso_r_sq_cv, method, data_name, folder_name, "Lasso Regression", "Lasso")
    save_aic_bic_plot(key, lasso_x, lasso_aic, lasso_bic, method, data_name, folder_name, "Lasso Regression", "Lasso")

    # ==========================================
    # --- Square Root Transformation ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Sqrt Transformation")
    print("------------------------------------")
    
    if key == 'Forward':
        (sqrt_features, sqrt_qof_list, sqrt_cv_stats_list) = forward_select_all(X, y, start_cols=None, method='sqrt', metric=0)
    elif key == 'Backward':
        (sqrt_features, sqrt_qof_list, sqrt_cv_stats_list) = backward_eliminate_all(X, y, start_cols=None, method='sqrt', metric=0)
    elif key == 'Stepwise':
        (sqrt_features, sqrt_qof_list, sqrt_cv_stats_list) = stepwise_selection(X, y, start_cols=None, method='sqrt', metric=1)
    
    # Extract tracking metrics
    sqrt_x = list(range(len(sqrt_features)))
    sqrt_r_sq = []
    sqrt_adj_r_sq = []
    sqrt_smape = []
    sqrt_r_sq_cv = []
    sqrt_aic = []
    sqrt_bic = []
    
    for i in sqrt_x:
        sqrt_r_sq.append(100 * sqrt_qof_list[i][0])
        sqrt_adj_r_sq.append(100 * sqrt_qof_list[i][1])
        sqrt_smape.append(sqrt_qof_list[i][8])
        sqrt_r_sq_cv.append(100 * np.mean(sqrt_cv_stats_list[i][0]))
        sqrt_aic.append(sqrt_qof_list[i][13])
        sqrt_bic.append(sqrt_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, sqrt_x, sqrt_r_sq, sqrt_adj_r_sq, sqrt_smape, sqrt_r_sq_cv, method, data_name, folder_name, "Sqrt Transformation", "Sqrt")
    save_aic_bic_plot(key, sqrt_x, sqrt_aic, sqrt_bic, method, data_name, folder_name, "Sqrt Transformation", "Sqrt")

    # ==========================================
    # --- Log1p Transformation ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Log1p Transformation")
    print("------------------------------------")
    
    if key == 'Forward':
        (log1p_features, log1p_qof_list, log1p_cv_stats_list) = forward_select_all(X, y, start_cols=None, method='log1p', metric=0)
    elif key == 'Backward':
        (log1p_features, log1p_qof_list, log1p_cv_stats_list) = backward_eliminate_all(X, y, start_cols=None, method='log1p', metric=0)
    elif key == 'Stepwise':
        (log1p_features, log1p_qof_list, log1p_cv_stats_list) = stepwise_selection(X, y, start_cols=None, method='log1p', metric=1)
    
    # Extract tracking metrics
    log1p_x = list(range(len(log1p_features)))
    log1p_r_sq = []
    log1p_adj_r_sq = []
    log1p_smape = []
    log1p_r_sq_cv = []
    log1p_aic = []
    log1p_bic = []
    
    for i in log1p_x:
        log1p_r_sq.append(100 * log1p_qof_list[i][0])
        log1p_adj_r_sq.append(100 * log1p_qof_list[i][1])
        log1p_smape.append(log1p_qof_list[i][8])
        log1p_r_sq_cv.append(100 * np.mean(log1p_cv_stats_list[i][0]))
        log1p_aic.append(log1p_qof_list[i][13])
        log1p_bic.append(log1p_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, log1p_x, log1p_r_sq, log1p_adj_r_sq, log1p_smape, log1p_r_sq_cv, method, data_name, folder_name, "Log1p Transformation", "Log1p")
    save_aic_bic_plot(key, log1p_x, log1p_aic, log1p_bic, method, data_name, folder_name, "Log1p Transformation", "Log1p")

    # ==========================================
    # --- Box-Cox Transformation ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Box-Cox Transformation")
    print("------------------------------------")
    
    if key == 'Forward':
        (boxcox_features, boxcox_qof_list, boxcox_cv_stats_list) = forward_select_all(X, y, start_cols=None, method='boxcox', metric=0, lambda_=boxcox_lambda)
    elif key == 'Backward':
        (boxcox_features, boxcox_qof_list, boxcox_cv_stats_list) = backward_eliminate_all(X, y, start_cols=None, method='boxcox', metric=0, lambda_=boxcox_lambda)
    elif key == 'Stepwise':
        (boxcox_features, boxcox_qof_list, boxcox_cv_stats_list) = stepwise_selection(X, y, start_cols=None, method='boxcox', metric=1, lambda_=boxcox_lambda)
    
    # Extract tracking metrics
    boxcox_x = list(range(len(boxcox_features)))
    boxcox_r_sq = []
    boxcox_adj_r_sq = []
    boxcox_smape = []
    boxcox_r_sq_cv = []
    boxcox_aic = []
    boxcox_bic = []
    
    for i in boxcox_x:
        boxcox_r_sq.append(100 * boxcox_qof_list[i][0])
        boxcox_adj_r_sq.append(100 * boxcox_qof_list[i][1])
        boxcox_smape.append(boxcox_qof_list[i][8])
        boxcox_r_sq_cv.append(100 * np.mean(boxcox_cv_stats_list[i][0]))
        boxcox_aic.append(boxcox_qof_list[i][13])
        boxcox_bic.append(boxcox_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, boxcox_x, boxcox_r_sq, boxcox_adj_r_sq, boxcox_smape, boxcox_r_sq_cv, method, data_name, folder_name, "Box-Cox Transformation", "BoxCox")
    save_aic_bic_plot(key, boxcox_x, boxcox_aic, boxcox_bic, method, data_name, folder_name, "Box-Cox Transformation", "BoxCox")

    # ==========================================
    # --- Order 2 Polynomial Regression ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} {method} for Order 2 Regression")
    print("------------------------------------")
    
    if key == 'Forward':
        (order2reg_features, order2reg_qof_list, order2reg_cv_stats_list) = forward_select_all(X_2_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=0, alpha=order2reg_alpha)
    elif key == 'Backward':
        (order2reg_features, order2reg_qof_list, order2reg_cv_stats_list) = backward_eliminate_all(X_2_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=0, alpha=order2reg_alpha)
    elif key == 'Stepwise':
        (order2reg_features, order2reg_qof_list, order2reg_cv_stats_list) = stepwise_selection(X_2_z_score_is, y_centered_is, start_cols=None, method='ridge', metric=1, alpha=order2reg_alpha)
    
    # Extract tracking metrics
    order2reg_x = list(range(len(order2reg_features)))
    order2reg_r_sq = []
    order2reg_adj_r_sq = []
    order2reg_smape = []
    order2reg_r_sq_cv = []
    order2reg_aic = []
    order2reg_bic = []
    
    for i in order2reg_x:
        order2reg_r_sq.append(100 * order2reg_qof_list[i][0])
        order2reg_adj_r_sq.append(100 * order2reg_qof_list[i][1])
        order2reg_smape.append(order2reg_qof_list[i][8])
        order2reg_r_sq_cv.append(100 * np.mean(order2reg_cv_stats_list[i][0]))
        order2reg_aic.append(order2reg_qof_list[i][13])
        order2reg_bic.append(order2reg_qof_list[i][14])

    # Generate and save plots
    save_rsq_plot(key, order2reg_x, order2reg_r_sq, order2reg_adj_r_sq, order2reg_smape, order2reg_r_sq_cv, method, data_name, folder_name, "Order 2 Regression", "Order2Reg")
    save_aic_bic_plot(key, order2reg_x, order2reg_aic, order2reg_bic, method, data_name, folder_name, "Order 2 Regression", "Order2Reg")

    # ==========================================
    # --- Return Selected Feature Lists ---
    # ==========================================
    return (reg_features, ridge_features, lasso_features, sqrt_features, log1p_features, boxcox_features, order2reg_features)