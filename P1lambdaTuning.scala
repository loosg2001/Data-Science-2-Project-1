// Place in modeling directory

package scalation
package modeling

import scalation.mathstat._
import modeling.TranRegression.{box_cox, cox_box}

/**
 * Tunes the lambda (shrinkage) hyperparameter for either Ridge or Lasso regression.
 * * This function performs a three-stage grid search to find the optimal lambda value
 * that maximizes the mean R-squared ($R^2$) score via cross-validation.
 * 1. Coarse logarithmic search.
 * 2. Intermediate search based on the best result from stage 1.
 * 3. Fine-grained linear search around the best result from stage 2.
 *
 * @param key         The type of regression to tune: either "ridge" or "lasso".
 * @param xZScoreIS   The standardized (z-scored) feature matrix.
 * @param yCenteredIS The centered target vector.
 * @param xFname      The array of feature names.
 * @return            A tuple containing `(bestLambda, bestRSq)`.
 */
def tuneRidgeLassoLambda(key: "ridge" | "lasso", xZScoreIS: MatrixD, yCenteredIS: VectorD, xFname: Array[String]): (Double, Double) =
    banner(s"Tuning lambda for $key")
    var bestLambda = 0.0
    var bestRSq = Double.NegativeInfinity

    // ==========================================
    // STAGE 1: Coarse Logarithmic Search
    // ==========================================
    for lambda <- List(0.0000001,  0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000) do
        banner("lambda = " + lambda)
        modeling.RidgeRegression.hp("lambda") = lambda                      // Adjust shrinkage hyperparameter
        val stats = 
            if key == "ridge" then
                val ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)  // Ridge Regression model
                ridge.crossValidate()                                            // Run cross-validation and get statistics
            else if key == "lasso" then
                val lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)  // Lasso Regression model
                lasso.crossValidate()                                            // Run cross-validation and get statistics
            else
                Array(Statistic(1, 1.0, 1.0, 1.0, 1.0, 1.0))
            end if
        
        val currentRSqMean = stats(0).mean                                       // Extract the mean R-squared value
    
        // Update best parameters if the current mean improves the score
        if currentRSqMean > bestRSq then
            bestRSq = currentRSqMean
            bestLambda = lambda
        end if

    end for

    // ==========================================
    // STAGE 2: Intermediate Search
    // ==========================================
    // Shift the search space based on the best lambda found in Stage 1
    val left1 = bestLambda / 10
    val lambdaList1 = List(left1, 2*left1, 4*left1, 6*left1, 8*left1, 10*left1, 20*left1, 40*left1, 60*left1, 80*left1, 100*left1)

    for lambda <- lambdaList1 do
        banner("lambda = " + lambda)
        modeling.RidgeRegression.hp("lambda") = lambda
        val stats = 
            if key == "ridge" then
                val ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)
                ridge.crossValidate()
            else if key == "lasso" then
                val lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)
                lasso.crossValidate()
            else
                Array(Statistic(1, 1.0, 1.0, 1.0, 1.0, 1.0))
            end if
        val currentRSqMean = stats(0).mean
    
        if currentRSqMean > bestRSq then
            bestRSq = currentRSqMean
            bestLambda = lambda
        end if

    end for

    // ==========================================
    // STAGE 3: Fine-grained Linear Search
    // ==========================================
    // Dynamically build a highly specific search grid based on the magnitude of the new bestLambda
    val lambdaList2 = 
        if bestLambda == left1 then
            val lam0 = bestLambda / 10
            List(lam0, 2*lam0, 4*lam0, 6*lam0, 8*lam0, 10*lam0, 20*lam0, 40*lam0, 60*lam0, 80*lam0, 100*lam0)

        else if bestLambda == 100 * left1 then
            val lam0 = bestLambda / 10
            List(lam0, 2*lam0, 4*lam0, 6*lam0, 8*lam0, 10*lam0, 20*lam0, 40*lam0, 60*lam0, 80*lam0, 100*lam0)
        
        else if bestLambda == 2 * left1 then
            val diff1 = left1 / 4
            val lam0 = left1
            val lam1 = lam0 + diff1; val lam2 = lam1 + diff1; val lam3 = lam2 + diff1; val lam4 = lam3 + diff1
            val diff2 = (2 * left1) / 4
            val lam5 = lam4 + diff2; val lam6 = lam5 + diff2; val lam7 = lam6 + diff2; val lam8 = lam7 + diff2
            List(lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8)

        else if bestLambda == 10 * left1 then
            val diff1 = (2 * left1) / 4
            val lam0 = bestLambda - (2 * left1)
            val lam1 = lam0 + diff1; val lam2 = lam1 + diff1; val lam3 = lam2 + diff1; val lam4 = lam3 + diff1
            val diff2 = (10 * left1) / 4
            val lam5 = lam4 + diff2; val lam6 = lam5 + diff2; val lam7 = lam6 + diff2; val lam8 = lam7 + diff2
            List(lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8)

        else if bestLambda == 20 * left1 then
            val diff1 = (10 * left1) / 4
            val lam0 = bestLambda - (10 * left1)
            val lam1 = lam0 + diff1; val lam2 = lam1 + diff1; val lam3 = lam2 + diff1; val lam4 = lam3 + diff1
            val diff2 = (20 * left1) / 4
            val lam5 = lam4 + diff2; val lam6 = lam5 + diff2; val lam7 = lam6 + diff2; val lam8 = lam7 + diff2
            List(lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8)

        else if bestLambda < 10 * left1 then
            val diff1 = (2 * left1) / 4
            val lam0 = bestLambda - (2 * left1)
            val lam1 = lam0 + diff1; val lam2 = lam1 + diff1; val lam3 = lam2 + diff1; val lam4 = lam3 + diff1
            val lam5 = lam4 + diff1; val lam6 = lam5 + diff1; val lam7 = lam6 + diff1; val lam8 = lam7 + diff1
            List(lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8)
        
        else if bestLambda > 10 * left1 then
            val diff1 = (20 * left1) / 4
            val lam0 = bestLambda - (20 * left1)
            val lam1 = lam0 + diff1; val lam2 = lam1 + diff1; val lam3 = lam2 + diff1; val lam4 = lam3 + diff1
            val lam5 = lam4 + diff1; val lam6 = lam5 + diff1; val lam7 = lam6 + diff1; val lam8 = lam7 + diff1
            List(lam0, lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8)
        
        else
            List(bestLambda)
        end if

    for lambda <- lambdaList2 do
        banner("lambda = " + lambda)
        modeling.RidgeRegression.hp("lambda") = lambda
        val stats = 
            if key == "ridge" then
                val ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)
                ridge.crossValidate()
            else if key == "lasso" then
                val lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)
                lasso.crossValidate()
            else
                Array(Statistic(1, 1.0, 1.0, 1.0, 1.0, 1.0))
            end if
        val currentRSqMean = stats(0).mean
    
        if currentRSqMean > bestRSq then
            bestRSq = currentRSqMean
            bestLambda = lambda
        end if

    end for

    banner("best R^2 and lambda")
    println(bestRSq)
    println(bestLambda)

    (bestLambda, bestRSq)
end tuneRidgeLassoLambda


/**
 * Tunes the lambda parameter for a Box-Cox transformed regression model.
 * * Evaluates a predefined grid of common transformation powers (including roots,
 * inverse roots, zero for log transform, and whole numbers) to find the optimal 
 * lambda that maximizes the mean R-squared ($R^2$) score via cross-validation.
 *
 * @param ox       The feature matrix.
 * @param y        The target vector to be transformed.
 * @param oxFname  The array of feature names.
 * @return         A tuple containing `(bestLambda, bestRSq)`.
 */
def tuneBoxCoxLambda(ox: MatrixD, y: VectorD, oxFname: Array[String]): (Double, Double) = 
    banner("Tuning lambda for Box-Cox")
    
    var bestLambda = 0.0
    var bestRSq = Double.NegativeInfinity

    // List of standard power transformations (e.g., -1 for inverse, 0 for log, 0.5 for square root, 2 for square)
    val boxCoxLambdas = List(-4, -3, -2, -1, -1.0/2, -1.0/3, -1.0/4, -1.0/5, -1.0/6, -1.0/7, -1.0/8, -1.0/9, -1.0/10, -1.0/11, -1.0/12, 0, 1.0/12, 1.0/11, 1.0/10, 1.0/9, 1.0/8, 1.0/7, 1.0/6, 1.0/5, 1.0/4, 1.0/3, 1.0/2, 1, 2, 3, 4)

    for lambda <- boxCoxLambdas do
        banner("lambda = " + lambda)
        modeling.TranRegression.λ = lambda                                       // Adjust transformation hyperparameter
        val boxCoxModel = new TranRegression(ox, y, oxFname, tran=box_cox, itran=cox_box) // Instantiate Box-Cox model
        val stats = boxCoxModel.crossValidate()                                  // Run cross-validation and get statistics
        val currentRSqMean = stats(0).mean                                       // Extract the mean R-squared value
    
        // Update best parameters if the current mean improves the score
        if currentRSqMean > bestRSq then
            bestRSq = currentRSqMean
            bestLambda = lambda
        end if
    end for

    banner("best R^2 and lambda")
    println(bestRSq)
    println(bestLambda)

    (bestLambda, bestRSq)
end tuneBoxCoxLambda