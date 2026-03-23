import pandas as pd
from sklearn.model_selection import train_test_split
from model_evaluation import lin_reg, ridge_reg, lasso_reg, sqrt_reg, log1p_reg, boxcox_reg, order2_reg
from feature_selection import feature_selection
from latex_tables import is_oos_comparison, model_comparison, cv_table

def get_tables(X, y, X_test, X_train, y_test, y_train, X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, X_2_z_score_is, X_2_test_z_score, X_2_train_z_score, data_name, folder_name):
    """
    Orchestrator function to evaluate multiple regression models, perform feature selection,
    and generate formatted outputs (LaTeX tables and plain text lists) for a given dataset.

    Args:
        X (pd.DataFrame): The full input feature matrix (with intercept).
        y (pd.Series): The full target response vector.
        X_test (pd.DataFrame): The testing input feature matrix (with intercept).
        X_train (pd.DataFrame): The training input feature matrix (with intercept).
        y_test (pd.Series): The testing target response vector.
        y_train (pd.Series): The training target response vector.
        X_z_score_is (pd.DataFrame): The standardized full input feature matrix.
        y_centered_is (pd.Series): The centered full target response vector.
        X_test_z_score (pd.DataFrame): The standardized testing feature matrix.
        X_train_z_score (pd.DataFrame): The standardized training feature matrix.
        y_test_centered (pd.Series): The centered testing target vector.
        y_train_centered (pd.Series): The centered training target vector.
        X_2_z_score_is (pd.DataFrame): The standardized quadratic/interaction feature matrix (Order 2).
        X_2_test_z_score (pd.DataFrame): The standardized testing Order 2 feature matrix.
        X_2_train_z_score (pd.DataFrame): The standardized training Order 2 feature matrix.
        data_name (str): The descriptive name of the dataset.
        folder_name (str): The directory path where output plots will be saved.

    Returns:
        None: Executes the entire modeling pipeline, prints tables, and saves plots.
    """
    # ==========================================
    # --- Data Formatting ---
    # ==========================================
    # Ensure all target variables are properly formatted as 1D Series for Statsmodels
    y = y.squeeze()
    y_test = y_test.squeeze()
    y_train = y_train.squeeze()
    y_centered_is = y_centered_is.squeeze()
    y_test_centered = y_test_centered.squeeze()
    y_train_centered = y_train_centered.squeeze()

    # ==========================================
    # --- Model Evaluation ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} Regression")
    print("------------------------------------")
    (reg_qof_is, reg_qof_oos, reg_cv_stats) = lin_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Ridge")
    print("------------------------------------")
    (ridge_qof_is, ridge_qof_oos, ridge_cv_stats, ridge_alpha) = ridge_reg(X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Lasso")
    print("------------------------------------")
    (lasso_qof_is, lasso_qof_oos, lasso_cv_stats, lasso_alpha, lasso_non_zero_features) = lasso_reg(X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Sqrt")
    print("------------------------------------")
    (sqrt_qof_is, sqrt_qof_oos, sqrt_cv_stats) = sqrt_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Log1p")
    print("------------------------------------")
    (log1p_qof_is, log1p_qof_oos, log1p_cv_stats) = log1p_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Box-Cox")
    print("------------------------------------")
    (boxcox_qof_is, boxcox_qof_oos, boxcox_cv_stats, boxcox_lambda) = boxcox_reg(X, y, X_test, X_train, y_test, y_train, data_name, folder_name)

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial")
    print("------------------------------------")
    (order2reg_qof_is, order2reg_qof_oos, order2reg_cv_stats, order2reg_alpha) = order2_reg(X_2_z_score_is, y_centered_is, X_2_test_z_score, X_2_train_z_score, y_test_centered, y_train_centered, data_name, folder_name)

    # ==========================================
    # --- Feature Selection ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} Forward Selection")
    print("------------------------------------")
    (reg_col_names_fs, ridge_col_names_fs, lasso_col_names_fs, sqrtreg_col_names_fs, log1p_col_names_fs, boxcox_col_names_fs, order2reg_col_names_fs) = feature_selection("Forward", X, y, X_z_score_is, y_centered_is, X_2_z_score_is, data_name, folder_name, ridge_alpha, lasso_alpha, boxcox_lambda, order2reg_alpha)

    print("------------------------------------")
    print(f"{data_name} Backward Elimination")
    print("------------------------------------")
    (reg_col_names_be, ridge_col_names_be, lasso_col_names_be, sqrtreg_col_names_be, log1p_col_names_be, boxcox_col_names_be, order2reg_col_names_be) = feature_selection("Backward", X, y, X_z_score_is, y_centered_is, X_2_z_score_is, data_name, folder_name, ridge_alpha, lasso_alpha, boxcox_lambda, order2reg_alpha)

    print("------------------------------------")
    print(f"{data_name} Stepwise Selection")
    print("------------------------------------")
    (reg_col_names_ss, ridge_col_names_ss, lasso_col_names_ss, sqrtreg_col_names_ss, log1p_col_names_ss, boxcox_col_names_ss, order2reg_col_names_ss) = feature_selection("Stepwise", X, y, X_z_score_is, y_centered_is, X_2_z_score_is, data_name, folder_name, ridge_alpha, lasso_alpha, boxcox_lambda, order2reg_alpha)

    # ==========================================
    # --- Cross-Validation Tables ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} Regression CV Table")
    print("------------------------------------")
    cv_table(reg_cv_stats, data_name, "Linear Regression")

    print("------------------------------------")
    print(f"{data_name} Ridge CV Table")
    print("------------------------------------")
    print(f"{data_name} Ridge alpha Used: {ridge_alpha}")
    cv_table(ridge_cv_stats, data_name, "Ridge Regression")

    print("------------------------------------")
    print(f"{data_name} Lasso CV Table")
    print("------------------------------------")
    print(f"{data_name} Lasso alpha Used: {lasso_alpha}")
    print(f"{data_name} Lasso Selected Features: {lasso_non_zero_features}")
    cv_table(lasso_cv_stats, data_name, "Lasso Regression")

    print("------------------------------------")
    print(f"{data_name} Sqrt CV Table")
    print("------------------------------------")
    cv_table(sqrt_cv_stats, data_name, "Sqrt Transformation")

    print("------------------------------------")
    print(f"{data_name} Log1p CV Table")
    print("------------------------------------")
    cv_table(log1p_cv_stats, data_name, "Log1p Transformation")

    print("------------------------------------")
    print(f"{data_name} Box-Cox CV Table")
    print("------------------------------------")
    print(f"{data_name} Box-Cox lambda Used: {boxcox_lambda}")
    cv_table(boxcox_cv_stats, data_name, "Box-Cox Transformation")

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial CV Table")
    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial alpha Used: {order2reg_alpha}")
    cv_table(order2reg_cv_stats, data_name, "Order 2 Polynomial Regression")

    # ==========================================
    # --- LaTeX Comparison Tables ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} Regression Comparison Table")
    print("------------------------------------")
    is_oos_comparison(reg_qof_is, reg_qof_oos, data_name, "Linear Regression")

    print("------------------------------------")
    print(f"{data_name} Ridge Comparison Table")
    print("------------------------------------")
    is_oos_comparison(ridge_qof_is, ridge_qof_oos, data_name, "Ridge Regression")

    print("------------------------------------")
    print(f"{data_name} Lasso Comparison Table")
    print("------------------------------------")
    is_oos_comparison(lasso_qof_is, lasso_qof_oos, data_name, "Lasso Regression")

    print("------------------------------------")
    print(f"{data_name} Sqrt Comparison Table")
    print("------------------------------------")
    is_oos_comparison(sqrt_qof_is, sqrt_qof_oos, data_name, "Sqrt Transformation")

    print("------------------------------------")
    print(f"{data_name} Log1p Comparison Table")
    print("------------------------------------")
    is_oos_comparison(log1p_qof_is, log1p_qof_oos, data_name, "Log1p Transformation")

    print("------------------------------------")
    print(f"{data_name} Box-Cox Comparison Table")
    print("------------------------------------")
    is_oos_comparison(boxcox_qof_is, boxcox_qof_oos, data_name, "Box-Cox Transformation")

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial Comparison Table")
    print("------------------------------------")
    is_oos_comparison(order2reg_qof_is, order2reg_qof_oos, data_name, "Order 2 Polynomial Regression")

    print("------------------------------------")
    print(f"{data_name} Model In-Sample Comparison Table")
    print("------------------------------------")
    model_comparison(reg_qof_is, ridge_qof_is, lasso_qof_is, sqrt_qof_is, log1p_qof_is, boxcox_qof_is, order2reg_qof_is, data_name, False)

    print("------------------------------------")
    print(f"{data_name} Model Out-of-Sample Comparison Table")
    print("------------------------------------")
    model_comparison(reg_qof_oos, ridge_qof_oos, lasso_qof_oos, sqrt_qof_oos, log1p_qof_oos, boxcox_qof_oos, order2reg_qof_oos, data_name, True)

    # ==========================================
    # --- Feature Selection Results Printing ---
    # ==========================================
    print("------------------------------------")
    print(f"{data_name} Regression Forward Selection Order")
    print("------------------------------------")
    print(reg_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Regression Backward Elimination Reversed Order")
    print("------------------------------------")
    print(reg_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Regression Stepwise Selection Order")
    print("------------------------------------")
    print(reg_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Ridge Forward Selection Order")
    print("------------------------------------")
    print(ridge_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Ridge Backward Elimination Reversed Order")
    print("------------------------------------")
    print(ridge_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Ridge Stepwise Selection Order")
    print("------------------------------------")
    print(ridge_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Lasso Forward Selection Order")
    print("------------------------------------")
    print(lasso_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Lasso Backward Elimination Reversed Order")
    print("------------------------------------")
    print(lasso_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Lasso Stepwise Selection Order")
    print("------------------------------------")
    print(lasso_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Sqrt Forward Selection Order")
    print("------------------------------------")
    print(sqrtreg_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Sqrt Backward Elimination Reversed Order")
    print("------------------------------------")
    print(sqrtreg_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Sqrt Stepwise Selection Order")
    print("------------------------------------")
    print(sqrtreg_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Log1p Forward Selection Order")
    print("------------------------------------")
    print(log1p_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Log1p Backward Elimination Reversed Order")
    print("------------------------------------")
    print(log1p_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Log1p Stepwise Selection Order")
    print("------------------------------------")
    print(log1p_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Box-Cox Forward Selection Order")
    print("------------------------------------")
    print(boxcox_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Box-Cox Backward Elimination Reversed Order")
    print("------------------------------------")
    print(boxcox_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Box-Cox Stepwise Selection Order")
    print("------------------------------------")
    print(boxcox_col_names_ss)

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial Forward Selection Order")
    print("------------------------------------")
    print(order2reg_col_names_fs)

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial Backward Elimination Reversed Order")
    print("------------------------------------")
    print(order2reg_col_names_be)

    print("------------------------------------")
    print(f"{data_name} Order 2 Polynomial Stepwise Selection Order")
    print("------------------------------------")
    print(order2reg_col_names_ss)


def p1_auto_mpg():
    """
    Main entry point for evaluating regression models on the Auto MPG dataset.
    
    * Loads and preprocesses data (splitting and standardizing).
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("Auto_MPG/cleaned_auto_mpg_with_intercept.csv")
    ox = oxy.drop('mpg', axis=1)
    y = oxy[['mpg']]

    oxy_2 = pd.read_csv("Auto_MPG/cleaned_order_2_auto_mpg_with_intercept.csv")
    ox_2 = oxy_2.drop('mpg', axis=1)

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    ox_train, ox_test, ox_2_train, ox_2_test, y_train, y_test = train_test_split(ox, ox_2, y, test_size=0.2, random_state=0)

    X = ox
    X_test = ox_test
    X_train = ox_train

    # ==========================================
    # --- Standardization (Z-Score Normalization) ---
    # ==========================================
    x = ox.drop('intercept', axis=1)
    X_z_score_is = (x - x.mean()) / x.std()
    y_centered_is = y - y.mean()

    x_train = ox_train.drop('intercept', axis=1)
    x_test = ox_test.drop('intercept', axis=1)

    X_train_z_score = (x_train - x_train.mean()) / x_train.std()
    X_test_z_score = (x_test - x_train.mean()) / x_train.std()

    y_train_centered = y_train - y_train.mean()
    y_test_centered = y_test - y_train.mean()

    x_2 = ox_2.drop('intercept', axis=1)
    x_2_train = ox_2_train.drop('intercept', axis=1)
    x_2_test = ox_2_test.drop('intercept', axis=1)

    X_2_z_score_is = (x_2 - x_2.mean()) / x_2.std()
    X_2_train_z_score = (x_2_train - x_2_train.mean()) / x_2_train.std()
    X_2_test_z_score = (x_2_test - x_2_train.mean()) / x_2_train.std()

    # ==========================================
    # --- Run Pipeline ---
    # ==========================================
    get_tables(X, y, X_test, X_train, y_test, y_train, X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, X_2_z_score_is, X_2_test_z_score, X_2_train_z_score, "Auto MPG", "Auto_MPG_Statsmodels_Plots")

    print("------------------------------------")
    print("Finished")
    print("------------------------------------")


def p1_housing():
    """
    Main entry point for evaluating regression models on the California Housing dataset.
    
    * Loads and preprocesses data (splitting and standardizing).
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("House_Prices/cleaned_housing_with_intercept.csv")
    ox = oxy.drop('median_house_value', axis=1)
    y = oxy[['median_house_value']]

    oxy_2 = pd.read_csv("House_Prices/cleaned_order_2_housing_with_intercept.csv")
    ox_2 = oxy_2.drop('median_house_value', axis=1)

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    ox_train, ox_test, ox_2_train, ox_2_test, y_train, y_test = train_test_split(ox, ox_2, y, test_size=0.2, random_state=0)

    X = ox
    X_test = ox_test
    X_train = ox_train

    # ==========================================
    # --- Standardization (Z-Score Normalization) ---
    # ==========================================
    x = ox.drop('intercept', axis=1)
    X_z_score_is = (x - x.mean()) / x.std()
    y_centered_is = y - y.mean()

    x_train = ox_train.drop('intercept', axis=1)
    x_test = ox_test.drop('intercept', axis=1)

    X_train_z_score = (x_train - x_train.mean()) / x_train.std()
    X_test_z_score = (x_test - x_train.mean()) / x_train.std()

    y_train_centered = y_train - y_train.mean()
    y_test_centered = y_test - y_train.mean()

    x_2 = ox_2.drop('intercept', axis=1)
    x_2_train = ox_2_train.drop('intercept', axis=1)
    x_2_test = ox_2_test.drop('intercept', axis=1)

    X_2_z_score_is = (x_2 - x_2.mean()) / x_2.std()
    X_2_train_z_score = (x_2_train - x_2_train.mean()) / x_2_train.std()
    X_2_test_z_score = (x_2_test - x_2_train.mean()) / x_2_train.std()

    # ==========================================
    # --- Run Pipeline ---
    # ==========================================
    get_tables(X, y, X_test, X_train, y_test, y_train, X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, X_2_z_score_is, X_2_test_z_score, X_2_train_z_score, "House Prices", "House_Prices_Statsmodels_Plots")

    print("------------------------------------")
    print("Finished")
    print("------------------------------------")


def p1_insurance():
    """
    Main entry point for evaluating regression models on the Medical Insurance dataset.
    
    * Loads and preprocesses data (splitting and standardizing).
    * Calls the `get_tables` orchestrator to generate metrics, plots, and LaTeX summaries.
    """
    # ==========================================
    # --- Data Loading ---
    # ==========================================
    oxy = pd.read_csv("Insurance_Charges/cleaned_insurance_with_intercept.csv")
    ox = oxy.drop('charges', axis=1)
    y = oxy[['charges']]

    oxy_2 = pd.read_csv("Insurance_Charges/cleaned_order_2_insurance_with_intercept.csv")
    ox_2 = oxy_2.drop('charges', axis=1)

    # ==========================================
    # --- Train-Test Split (80-20) ---
    # ==========================================
    ox_train, ox_test, ox_2_train, ox_2_test, y_train, y_test = train_test_split(ox, ox_2, y, test_size=0.2, random_state=0)

    X = ox
    X_test = ox_test
    X_train = ox_train

    # ==========================================
    # --- Standardization (Z-Score Normalization) ---
    # ==========================================
    x = ox.drop('intercept', axis=1)
    X_z_score_is = (x - x.mean()) / x.std()
    y_centered_is = y - y.mean()

    x_train = ox_train.drop('intercept', axis=1)
    x_test = ox_test.drop('intercept', axis=1)

    X_train_z_score = (x_train - x_train.mean()) / x_train.std()
    X_test_z_score = (x_test - x_train.mean()) / x_train.std()

    y_train_centered = y_train - y_train.mean()
    y_test_centered = y_test - y_train.mean()

    x_2 = ox_2.drop('intercept', axis=1)
    x_2_train = ox_2_train.drop('intercept', axis=1)
    x_2_test = ox_2_test.drop('intercept', axis=1)

    X_2_z_score_is = (x_2 - x_2.mean()) / x_2.std()
    X_2_train_z_score = (x_2_train - x_2_train.mean()) / x_2_train.std()
    X_2_test_z_score = (x_2_test - x_2_train.mean()) / x_2_train.std()

    # ==========================================
    # --- Run Pipeline ---
    # ==========================================
    get_tables(X, y, X_test, X_train, y_test, y_train, X_z_score_is, y_centered_is, X_test_z_score, X_train_z_score, y_test_centered, y_train_centered, X_2_z_score_is, X_2_test_z_score, X_2_train_z_score, "Insurance Charges", "Insurance_Charges_Statsmodels_Plots")

    print("------------------------------------")
    print("Finished")
    print("------------------------------------")