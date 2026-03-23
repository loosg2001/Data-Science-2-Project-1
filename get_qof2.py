import statsmodels.api as sm
import numpy as np
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from get_qof import get_qof
from get_cv_qof import get_cv_qof

def get_qof2(X, y, method='ridge', alpha=0.0, lambda_=0.0, cv=False):
    """
    Fits a specified regression model and calculates Quality of Fit (QoF) metrics.
    Optionally performs k-fold cross-validation to gather out-of-sample performance statistics.

    This function serves as a unified interface for evaluating standard linear regression, 
    regularized regression (Ridge, Lasso), and models with transformed targets 
    (Square Root, Log1p, Box-Cox).

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target response vector.
        method (str): The regression method to evaluate ('linreg', 'ridge', 'lasso', 
                      'sqrt', 'log1p', 'boxcox'). Defaults to 'ridge'.
        alpha (float): The regularization penalty strength used for Ridge and Lasso models. 
                       Defaults to 0.0.
        lambda_ (float): The transformation parameter used specifically for the Box-Cox model. 
                         Defaults to 0.0.
        cv (bool): Flag indicating whether to perform cross-validation. Defaults to False.

    Returns:
        tuple: A tuple containing `(qof, cv_stats)`:
               - qof (list): A list of 15 metrics evaluating the model on the full provided dataset.
               - cv_stats (list or None): If `cv=True`, a list of 15 lists containing metrics across 
                 each fold. Otherwise, returns None.
    """
    # ==========================================
    # --- Cross-Validation ---
    # ==========================================
    if cv:
        # Perform cross-validation and collect fold statistics
        cv_stats = get_cv_qof(X, y, method=method, alpha=alpha, lambda_=lambda_)
    else:
        # Bypass cross-validation if flag is set to False
        cv_stats = None
    
    # ==========================================
    # --- Model Training and Prediction ---
    # ==========================================
    if method == 'linreg':
        model = sm.OLS(y, X).fit()
        y_pred = model.predict(X)
        
    elif method == 'ridge':
        # L1_wt=0.0 strictly defines Ridge (L2 penalty)
        model = sm.OLS(y, X).fit_regularized(alpha=alpha, L1_wt=0.0)
        y_pred = model.predict(X)
        
    elif method == 'lasso':
        # L1_wt=1.0 strictly defines Lasso (L1 penalty)
        model = sm.OLS(y, X).fit_regularized(alpha=alpha, L1_wt=1.0)
        y_pred = model.predict(X)
        
    elif method == 'sqrt':
        # Transform target with sqrt, then square predictions to return to original scale
        model = sm.OLS(np.sqrt(y), X).fit()
        y_pred = (model.predict(X)) ** 2
        
    elif method == 'log1p':
        # Transform target with log(1 + y), then apply exp(pred) - 1 to return to original scale
        model = sm.OLS(np.log1p(y), X).fit()
        y_pred = np.expm1(model.predict(X))
        
    elif method == 'boxcox':
        # Transform target using Box-Cox lambda, then apply inverse Box-Cox to predictions
        model = sm.OLS(boxcox(y, lambda_), X).fit()
        y_pred = inv_boxcox(model.predict(X), lambda_)
        
    else:
        raise ValueError(f"method must be one of 'linreg', 'ridge', 'lasso', 'sqrt', 'log1p', or 'boxcox'. Received {method}")
    
    # ==========================================
    # --- Metrics Calculation & Return ---
    # ==========================================
    k = X.shape[1]  # Extract the number of predictor features
    
    # Calculate the 15 Quality of Fit metrics on the full dataset
    qof = get_qof(y, y_pred, k)

    return (qof, cv_stats)