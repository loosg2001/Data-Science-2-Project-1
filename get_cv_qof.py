import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from get_qof import get_qof


def get_cv_qof(X, y, method="linreg", alpha=0.0, lambda_=0.0, n_splits=5):
    """
    Performs k-fold cross-validation for a specified regression model and calculates 
    Quality of Fit (QoF) metrics for each fold.
    
    This function handles standard linear regression, regularized regression (Ridge, Lasso), 
    and target variable transformations (Square Root, Log1p, Box-Cox).

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target response vector.
        method (str): The regression method to evaluate ('linreg', 'ridge', 'lasso', 
                      'sqrt', 'log1p', 'boxcox'). Defaults to 'linreg'.
        alpha (float): The regularization penalty strength used for Ridge and Lasso models. 
                       Defaults to 0.0.
        lambda_ (float): The transformation parameter used specifically for the Box-Cox model. 
                         Defaults to 0.0.
        n_splits (int): The number of cross-validation folds. Defaults to 5.

    Returns:
        list: A list of 15 lists (`cv_stats`), where each inner list contains the metric 
              values evaluated across all k-folds. For example, `cv_stats[0]` holds 
              the R-squared values for every fold.
    """
    # ==========================================
    # --- Cross-Validation Setup ---
    # ==========================================
    # Initialize the K-Fold splitter with shuffling enabled for randomized data distribution
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    # Initialize a list of 15 empty lists to store the 15 QoF metrics for each fold
    cv_stats = [[] for _ in range(15)]
    
    # Iterate through each fold
    for train_idx, val_idx in kf.split(X):
        # --- Data Splitting ---
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # ==========================================
        # --- Model Training and Prediction ---
        # ==========================================
        if method == 'linreg':
            model = sm.OLS(y_tr, X_tr).fit()
            y_pred = model.predict(X_val)
            
        elif method == 'ridge':
            # L1_wt=0.0 strictly defines Ridge (L2 penalty)
            model = sm.OLS(y_tr, X_tr).fit_regularized(alpha=alpha, L1_wt=0.0)
            y_pred = model.predict(X_val)
            
        elif method == 'lasso':
            # L1_wt=1.0 strictly defines Lasso (L1 penalty)
            model = sm.OLS(y_tr, X_tr).fit_regularized(alpha=alpha, L1_wt=1.0)
            y_pred = model.predict(X_val)
            
        elif method == 'sqrt':
            # Transform target with sqrt, then square predictions to return to original scale
            model = sm.OLS(np.sqrt(y_tr), X_tr).fit()
            y_pred = (model.predict(X_val)) ** 2
            
        elif method == 'log1p':
            # Transform target with log(1 + y), then apply exp(pred) - 1 to return to original scale
            model = sm.OLS(np.log1p(y_tr), X_tr).fit()
            y_pred = np.expm1(model.predict(X_val))
            
        elif method == 'boxcox':
            # Transform target using Box-Cox lambda, then apply inverse Box-Cox to predictions
            model = sm.OLS(boxcox(y_tr, lambda_), X_tr).fit()
            y_pred = inv_boxcox(model.predict(X_val), lambda_)
            
        else:
            raise ValueError(f"method must be one of 'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', or 'boxcox'. Received {method}")
        
        # ==========================================
        # --- Model Evaluation ---
        # ==========================================
        k = X.shape[1]  # Extract the number of predictor features
        
        # Calculate the 15 QoF metrics for the current validation fold
        temp_qof = get_qof(y_val, y_pred, k)
        
        # Append each calculated metric to its respective tracking list
        for i in range(15):
            cv_stats[i].append(temp_qof[i])

    # ==========================================
    # --- Return Results ---
    # ==========================================
    return cv_stats