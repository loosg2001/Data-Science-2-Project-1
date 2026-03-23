from get_qof2 import get_qof2

def select_single_feature(X, y, in_cols, out_cols, method='linreg', alpha=0.0, lambda_=0.0, metric=0):
    """
    Evaluates all candidate features and selects the single best feature to add to the model.
    
    This function iterates through the features currently not in the model (`out_cols`), 
    temporarily adds each one, and calculates the Quality of Fit (QoF). It selects the 
    feature that improves the specified target metric the most.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The target response vector.
        in_cols (list): A list of column names currently included in the model.
        out_cols (list): A list of candidate column names not yet in the model.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        metric (int): The index of the QoF metric to optimize (e.g., 0 for R^2, 13 for AIC). 
                      Defaults to 0.

    Returns:
        tuple: A tuple containing `(new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats)`.
    """
    # ==========================================
    # --- Metric Initialization ---
    # ==========================================
    # Metrics [0=R^2, 1=Adj_R^2, 12=F-stat] are to be maximized
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    # Metrics [3=SSE, 4=SDE, 5=MSE, 6=RMSE, 7=MAE, 8=sMAPE, 13=AIC, 14=BIC] are to be minimized
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_add = ''
    
    # ==========================================
    # --- Feature Evaluation ---
    # ==========================================
    for col in out_cols:
        new_cols = in_cols + [col]
        temp_X = X[new_cols].copy()

        # Evaluate the temporary model (without Cross-Validation for speed)
        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=False)

        cur_metric = temp_qof[metric]

        # Check for maximization improvements
        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_add = col
        # Check for minimization improvements
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_add = col

    # ==========================================
    # --- Finalize Selection ---
    # ==========================================
    new_in_cols = in_cols + [feature_to_add]
    new_out_cols = [col for col in out_cols if col != feature_to_add]

    temp_X = X[new_in_cols].copy()

    # Re-evaluate the best model, this time securing Cross-Validation statistics
    (best_qof, best_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)

    return (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats)


def forward_select_all(X, y, start_cols=None, method='linreg', alpha=0.0, lambda_=0.0, metric=0):
    """
    Performs Forward Selection feature engineering across the entire dataset.
    
    Iteratively adds the single best feature to the model until all available features 
    have been included. Tracks and stores the progression of QoF metrics.

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The target response vector.
        start_cols (list, optional): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        metric (int): The index of the QoF metric to optimize. Defaults to 0 (R^2).

    Returns:
        tuple: A tuple containing `(for_sel_features, qof_list, cv_stats_list)` mapping the
               order of addition and corresponding historical metrics.
    """
    # ==========================================
    # --- Initialization ---
    # ==========================================
    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy = ['intercept']
            in_cols = ['intercept']
            for_sel_features = ['intercept']
        else:
            start_cols_copy = []
            in_cols = []
            for_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        for_sel_features = [start_cols]

    # Handle Null/Intercept-only baseline if starting with empty predictors
    if len(in_cols) == 0:
        qof_list = []
        cv_stats_list = []
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
            for_sel_features.append('Null')
            qof_list.append(int_qof.copy())
            cv_stats_list.append(int_cv_stats.copy())
    else:
        temp_X = X[in_cols].copy()
        (qof, cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
        qof_list = [qof]
        cv_stats_list = [cv_stats]

    out_cols = [col for col in X.columns if col not in start_cols_copy]

    # ==========================================
    # --- Forward Selection Loop ---
    # ==========================================
    while True:
        (new_in_cols, new_out_cols, feature_to_add, best_qof, best_cv_stats) = select_single_feature(
            X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
        )
        
        in_cols = new_in_cols.copy()
        out_cols = new_out_cols.copy()
        for_sel_features.append(feature_to_add)
        qof_list.append(best_qof.copy())
        cv_stats_list.append(best_cv_stats.copy())

        if len(out_cols) == 0:
            break

    return (for_sel_features, qof_list, cv_stats_list)


def eliminate_single_feature(X, y, in_cols, method='linreg', alpha=0.0, lambda_=0.0, metric=0):
    """
    Evaluates all included features and selects the single worst feature to remove.
    
    This function iterates through the features currently in the model (`in_cols`), 
    temporarily removes each one, and calculates the resulting Quality of Fit (QoF). 
    It selects the feature whose removal yields the best metric (or hurts it the least).

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The target response vector.
        in_cols (list): A list of column names currently included in the model.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        metric (int): The index of the QoF metric to optimize. Defaults to 0.

    Returns:
        tuple: A tuple containing `(new_in_cols, feature_to_remove, best_qof, best_cv_stats)`.
    """
    # ==========================================
    # --- Metric Initialization ---
    # ==========================================
    if metric in [0, 1, 12]:
        best_metric = -float('inf')
    elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
        best_metric = float('inf')
    else:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")

    feature_to_remove = ''
    
    # Exclude the intercept from being evaluated for elimination
    if 'intercept' in in_cols:
        in_cols_copy = [col for col in in_cols if col != 'intercept']
    else:
        in_cols_copy = in_cols.copy()

    # ==========================================
    # --- Feature Evaluation ---
    # ==========================================
    for col in in_cols_copy:
        if 'intercept' in X.columns:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col] + ['intercept']
        else:
            new_cols = [col2 for col2 in in_cols_copy if col2 != col]
        
        temp_X = X[new_cols].copy()

        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=False)

        cur_metric = temp_qof[metric]

        if metric in [0, 1, 12]:
            if cur_metric > best_metric:
                best_metric = cur_metric
                feature_to_remove = col
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            if cur_metric < best_metric:
                best_metric = cur_metric
                feature_to_remove = col

    # ==========================================
    # --- Finalize Elimination ---
    # ==========================================
    new_in_cols = [col2 for col2 in in_cols if col2 != feature_to_remove]

    temp_X = X[new_in_cols].copy()

    # Re-evaluate the model after dropping the worst feature, this time securing CV stats
    (best_qof, best_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)

    return (new_in_cols, feature_to_remove, best_qof, best_cv_stats)


def backward_eliminate_all(X, y, start_cols=None, method='linreg', alpha=0.0, lambda_=0.0, metric=0):
    """
    Performs Backward Elimination feature engineering across the entire dataset.
    
    Starts with a full model containing all features and iteratively drops the weakest 
    feature one by one until only the intercept (or empty model) remains. 

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The target response vector.
        start_cols (list, optional): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        metric (int): The index of the QoF metric to optimize. Defaults to 0 (R^2).

    Returns:
        tuple: A tuple containing `(bac_eli_features, qof_list, cv_stats_list)` mapping the
               reverse order of elimination and corresponding historical metrics.
    """
    # ==========================================
    # --- Initialization ---
    # ==========================================
    if start_cols is None:
        in_cols = X.columns.tolist().copy()
    elif len(start_cols) == 0:
        raise ValueError(f"start_cols must be non-empty")
    else:
        in_cols = start_cols.copy()
    
    temp_X = X[in_cols].copy()

    (qof, cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)

    bac_eli_features = []
    qof_list = [qof]
    cv_stats_list = [cv_stats]

    # ==========================================
    # --- Backward Elimination Loop ---
    # ==========================================
    while True:
        (new_in_cols, feature_to_remove, best_qof, best_cv_stats) = eliminate_single_feature(
            X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
        )
        
        in_cols = new_in_cols.copy()
        bac_eli_features.append(feature_to_remove)
        qof_list.append(best_qof.copy())
        cv_stats_list.append(best_cv_stats.copy())

        # Stop when only the intercept (or final single feature) is left
        if len(in_cols) == 1:
            break
    
    # Log the final remaining column
    bac_eli_features.append(in_cols[0])

    # Handle the Null model calculation if necessary
    if in_cols[0] != 'intercept':
        temp_X = X.copy()
        temp_X['intercept'] = 1
        X_int = temp_X[['intercept']].copy()
        (int_qof, int_cv_stats) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
        bac_eli_features.append('Null')
        qof_list.append(int_qof.copy())
        cv_stats_list.append(int_cv_stats.copy())

    # Reverse outputs to show selection order (from best standalone to least important)
    bac_eli_features.reverse()
    qof_list.reverse()
    cv_stats_list.reverse()

    return (bac_eli_features, qof_list, cv_stats_list)


def stepwise_selection(X, y, start_cols=None, method='linreg', alpha=0.0, lambda_=0.0, metric=1):
    """
    Performs Stepwise Selection feature engineering.
    
    At each step, this algorithm evaluates both adding a new feature (Forward Step) 
    and dropping an existing feature (Backward Step). It greedily chooses the action 
    that maximizes the targeted QoF metric (default is Adjusted R^2).

    Args:
        X (pd.DataFrame): The full input feature matrix.
        y (pd.Series): The target response vector.
        start_cols (list, optional): A list of starting column names. Defaults to None.
        method (str): The regression method to evaluate. Defaults to 'linreg'.
        alpha (float): The regularization penalty for Ridge/Lasso. Defaults to 0.0.
        lambda_ (float): The transformation parameter for Box-Cox. Defaults to 0.0.
        metric (int): The index of the QoF metric to optimize. Defaults to 1 (Adjusted R^2).

    Returns:
        tuple: A tuple containing `(step_sel_features, qof_list, cv_stats_list)`.
    """
    if metric not in [0, 1, 3, 4, 5, 6, 7, 8, 12, 13, 14]:
        raise ValueError(f"metric must be one of [0,1,3,4,5,6,7,8,12,13,14]. Received {metric}")
    
    # Dictionaries to maintain metric history keyed by added/remaining features
    qof_dict = {}
    cv_stats_dict = {}

    # ==========================================
    # --- Initialization ---
    # ==========================================
    if start_cols is None:
        if 'intercept' in X.columns:
            start_cols_copy = ['intercept']
            in_cols = ['intercept']
            step_sel_features = ['intercept']

            temp_X = X[in_cols].copy()
            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
    
            qof_dict['intercept'] = temp_qof.copy()
            cv_stats_dict['intercept'] = temp_cv_stats.copy()
            cur_metric = temp_qof[metric]
        else:
            start_cols_copy = []
            in_cols = []
            step_sel_features = []
    else:
        start_cols_copy = start_cols.copy()
        in_cols = start_cols.copy()
        step_sel_features = start_cols.copy()
        
        # Initialize dictionary metrics for predefined starting columns
        for i in range(len(in_cols)):
            temp_in_cols = in_cols[:i+1]
            temp_X = X[temp_in_cols].copy()

            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)

            qof_dict[in_cols[i]] = temp_qof.copy()
            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
        cur_metric = temp_qof[metric]
    
    # Handle Null/Intercept-only baseline
    if len(in_cols) == 0:
        if 'intercept' not in X.columns:
            temp_X = X.copy()
            temp_X['intercept'] = 1
            X_int = temp_X[['intercept']].copy()
            (int_qof, int_cv_stats) = get_qof2(X_int, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
            step_sel_features.append('Null')
            qof_dict['Null'] = int_qof.copy()
            cv_stats_dict['Null'] = int_cv_stats.copy()
            cur_metric = int_qof[metric]
        elif metric in [0, 1, 12]:
            cur_metric = -float('inf')
        elif metric in [3, 4, 5, 6, 7, 8, 13, 14]:
            cur_metric = float('inf')
        else:
            raise ValueError(f"cur_metric was not assigned a value: {cur_metric}")

    out_cols = [col for col in X.columns if col not in start_cols_copy]
    num_cols = X.shape[1]
 
    # ==========================================
    # --- Stepwise Selection Loop ---
    # ==========================================
    while True:

        # Scenario A: The model is effectively empty; we can only add features (Forward step)
        if len(in_cols) <= 1:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
            )

            # Maximize metric logic
            if metric in [0, 1, 12] and (sel_best_qof[metric] >= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof.copy()
                cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                cur_metric = sel_best_qof[metric]

            # Minimize metric logic
            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (sel_best_qof[metric] <= cur_metric):
                in_cols = sel_new_in_cols.copy()
                out_cols = sel_new_out_cols.copy()
                step_sel_features.append(feature_to_add)
                qof_dict[feature_to_add] = sel_best_qof.copy()
                cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                cur_metric = sel_best_qof[metric]

            else:
                break

        # Scenario B: The model is full; we can only remove features (Backward step)
        elif len(in_cols) == num_cols:
            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
            )

            # Maximize metric logic
            if metric in [0, 1, 12] and (bac_best_qof[metric] >= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features_copy = step_sel_features.copy()
                step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                # Update dictionary metrics post-removal
                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
                        qof_dict[in_cols[i]] = temp_qof.copy()
                        cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                
                cur_metric = bac_best_qof[metric]

            # Minimize metric logic
            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and (bac_best_qof[metric] <= cur_metric):
                old_in_cols = in_cols.copy()
                in_cols = bac_new_in_cols.copy()
                temp_in_cols = bac_new_in_cols.copy()
                out_cols = [col for col in X.columns if col not in temp_in_cols]
                step_sel_features_copy = step_sel_features.copy()
                step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                # Update dictionary metrics post-removal
                flag = False
                for i in range(len(old_in_cols)):
                    if old_in_cols[i] == feature_to_remove:
                        flag = True
                    if flag and (i < len(in_cols)):
                        temp_in_cols = in_cols[:i+1]
                        temp_X = X[temp_in_cols].copy()
                        (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
                        qof_dict[in_cols[i]] = temp_qof.copy()
                        cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                
                cur_metric = bac_best_qof[metric]

            else:
                break

        # Scenario C: Intermediate model; evaluate BOTH adding and removing features
        else:
            (sel_new_in_cols, sel_new_out_cols, feature_to_add, sel_best_qof, sel_best_cv_stats) = select_single_feature(
                X, y, in_cols, out_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
            )

            (bac_new_in_cols, feature_to_remove, bac_best_qof, bac_best_cv_stats) = eliminate_single_feature(
                X, y, in_cols, method=method, alpha=alpha, lambda_=lambda_, metric=metric
            )

            # Maximize metric logic
            if metric in [0, 1, 12] and ((sel_best_qof[metric] >= cur_metric) or (bac_best_qof[metric] >= cur_metric)):
                # Adding is better or equal to dropping
                if sel_best_qof[metric] >= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof.copy()
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                    cur_metric = sel_best_qof[metric]

                # Dropping is better
                elif sel_best_qof[metric] < bac_best_qof[metric]:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features_copy = step_sel_features.copy()
                    step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                    # Update dictionary metrics post-removal
                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
                            qof_dict[in_cols[i]] = temp_qof.copy()
                            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                    
                    cur_metric = bac_best_qof[metric]

            # Minimize metric logic
            elif metric in [3, 4, 5, 6, 7, 8, 13, 14] and ((sel_best_qof[metric] <= cur_metric) or (bac_best_qof[metric] <= cur_metric)):
                # Adding is better or equal to dropping
                if sel_best_qof[metric] <= bac_best_qof[metric]:
                    in_cols = sel_new_in_cols.copy()
                    out_cols = sel_new_out_cols.copy()
                    step_sel_features.append(feature_to_add)
                    qof_dict[feature_to_add] = sel_best_qof.copy()
                    cv_stats_dict[feature_to_add] = sel_best_cv_stats.copy()
                    cur_metric = sel_best_qof[metric]

                # Dropping is better
                elif sel_best_qof[metric] > bac_best_qof[metric]:
                    old_in_cols = in_cols.copy()
                    in_cols = bac_new_in_cols.copy()
                    temp_in_cols = bac_new_in_cols.copy()
                    out_cols = [col for col in X.columns if col not in temp_in_cols]
                    step_sel_features_copy = step_sel_features.copy()
                    step_sel_features = [feat for feat in step_sel_features_copy if feat != feature_to_remove]

                    # Update dictionary metrics post-removal
                    flag = False
                    for i in range(len(old_in_cols)):
                        if old_in_cols[i] == feature_to_remove:
                            flag = True
                        if flag and (i < len(in_cols)):
                            temp_in_cols = in_cols[:i+1]
                            temp_X = X[temp_in_cols].copy()
                            (temp_qof, temp_cv_stats) = get_qof2(temp_X, y, method=method, alpha=alpha, lambda_=lambda_, cv=True)
                            qof_dict[in_cols[i]] = temp_qof.copy()
                            cv_stats_dict[in_cols[i]] = temp_cv_stats.copy()
                    
                    cur_metric = bac_best_qof[metric]

            else:
                break

    # ==========================================
    # --- Format Results Return ---
    # ==========================================
    qof_list = []
    cv_stats_list = []

    # Map output matrices to match the generated Stepwise flow order
    for col in step_sel_features:
        qof_list.append(qof_dict[col].copy())
        cv_stats_list.append(cv_stats_dict[col].copy())

    return (step_sel_features, qof_list, cv_stats_list)