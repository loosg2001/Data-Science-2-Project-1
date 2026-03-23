// Place in modeling directory

package scalation
package modeling

import scalation.mathstat._
import scala.math.sqrt
import modeling.TranRegression.{box_cox, cox_box}
import scalation.scala2d.savePlot
import java.awt.Window

/**
 * Evaluates a standard Multiple Linear Regression model.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param ox          The full input data matrix.
 * @param y           The full response vector.
 * @param oxTest      The testing input data matrix (Out-of-Sample).
 * @param oxTrain     The training input data matrix (Out-of-Sample).
 * @param yTest       The testing response vector.
 * @param yTrain      The training response vector.
 * @param oxFname     The array of feature names.
 * @param folderName  The directory path where output plots will be saved.
 * @param scale       The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ          The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 */
def linReg(ox: MatrixD,  y: VectorD, oxTest: MatrixD, oxTrain: MatrixD, yTest: VectorD, yTrain: VectorD, oxFname: Array[String], folderName: String, scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic]) =
    // --- In-Sample Evaluation ---
    banner("In-Sample")                 
    var reg = new Regression(ox, y, oxFname)                                     // Instantiate Regression model
    val (ypIS, qof1Full) = reg.trainNtest()()                                    // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      // Extract core QoF metrics
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      // Order by actual response for visualization
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${reg.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameIS = s"$folderName/Scalation_Reg_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    // Save high-resolution plot
    println(reg.summary())                                                       // Print parameter/coefficient statistics
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    reg = new Regression(ox, y, oxFname)                                         // Re-instantiate Regression model
    val (ypOOS, qof2Full) = reg.trainNtest(oxTrain, yTrain)(oxTest, yTest)       // Train on training data, test on validation data
    val qof2 = qof2Full(rQ)                                                      // Extract core QoF metrics
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            // Order by actual test response for visualization
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${reg.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameOOS = s"$folderName/Scalation_Reg_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       // Save high-resolution plot
    println(reg.summary())                                                       // Print parameter/coefficient statistics

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    reg = new Regression(ox, y, oxFname)                                         // Re-instantiate for CV
    val cvStats = reg.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats)
end linReg


/**
 * Evaluates a Ridge Regression model.
 * * First tunes the lambda (shrinkage) hyperparameter using a grid search.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param xZScoreIS      The standardized full input data matrix.
 * @param yCenteredIS    The centered full response vector.
 * @param xZScoreOOS     The standardized full data matrix used for OOS initialization.
 * @param yCenteredOOS   The centered full response vector used for OOS initialization.
 * @param xTestZScore    The standardized testing input data matrix.
 * @param yTestCentered  The centered testing response vector.
 * @param xTrainZScore   The standardized training input data matrix.
 * @param yTrainCentered The centered training response vector.
 * @param xFname         The array of feature names.
 * @param folderName     The directory path where output plots will be saved.
 * @param scale          The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ             The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 *                          - `bestLambda`: The optimal shrinkage parameter found via grid search.
 */
def ridgeReg(xZScoreIS: MatrixD, yCenteredIS: VectorD, xZScoreOOS: MatrixD, yCenteredOOS: VectorD, xTestZScore: MatrixD, yTestCentered: VectorD, xTrainZScore: MatrixD, yTrainCentered: VectorD, xFname: Array[String], folderName: String, scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic], Double) =
    val (bestLambda, _) = tuneRidgeLassoLambda("ridge", xZScoreIS, yCenteredIS, xFname)
    modeling.RidgeRegression.hp("lambda") = bestLambda                           // Set optimized shrinkage hyperparameter

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)              // Instantiate Ridge Regression model
    val (ypIS, qof1Full) = ridge.trainNtest()()                                  // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      // Extract core QoF metrics
    val (yOrd, ypOrdIS) = orderByY(yCenteredIS, ypIS)                            // Order by actual response
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${ridge.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_Ridge_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(ridge.summary())                                                     // Print parameter/coefficient statistics
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")                                                        
    ridge = new RidgeRegression(xZScoreOOS, yCenteredOOS, xFname)                // Re-instantiate Ridge Regression model
    val (ypOOS, qof2Full) = ridge.trainNtest(xTrainZScore, yTrainCentered)(xTestZScore, yTestCentered) 
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTestCentered, ypOOS)                    
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${ridge.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_Ridge_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(ridge.summary())                                                     

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)                  // Re-instantiate for CV
    val cvStats = ridge.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats, bestLambda)
end ridgeReg


/**
 * Evaluates a Lasso Regression model.
 * * First tunes the lambda (shrinkage) hyperparameter using a grid search.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param xZScoreIS      The standardized full input data matrix.
 * @param yCenteredIS    The centered full response vector.
 * @param xZScoreOOS     The standardized full data matrix used for OOS initialization.
 * @param yCenteredOOS   The centered full response vector used for OOS initialization.
 * @param xTestZScore    The standardized testing input data matrix.
 * @param yTestCentered  The centered testing response vector.
 * @param xTrainZScore   The standardized training input data matrix.
 * @param yTrainCentered The centered training response vector.
 * @param xFname         The array of feature names.
 * @param folderName     The directory path where output plots will be saved.
 * @param scale          The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ             The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 *                          - `bestLambda`: The optimal shrinkage parameter found via grid search.
 *                          - `nonZeroFeaturesList`: An List[String] of the feature names that survived the L1 penalty 
 *                                                  (excluding the intercept). These are derived from the final model trained on the 
 *                                                  full dataset using the `bestLambda`, applying a zero-threshold of 1e-6.
 */
def lassoReg(xZScoreIS: MatrixD, yCenteredIS: VectorD, xZScoreOOS: MatrixD, yCenteredOOS: VectorD, xTestZScore: MatrixD, yTestCentered: VectorD, xTrainZScore: MatrixD, yTrainCentered: VectorD, xFname: Array[String], folderName: String, scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic], Double, List[String]) =
    val (bestLambda, _) = tuneRidgeLassoLambda("lasso", xZScoreIS, yCenteredIS, xFname)
    modeling.RidgeRegression.hp("lambda") = bestLambda                           // Set optimized shrinkage hyperparameter

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)              // Instantiate Lasso Regression model
    val (ypIS, qof1Full) = lasso.trainNtest()()                                  // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      
    val (yOrd, ypOrdIS) = orderByY(yCenteredIS, ypIS)                            
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${lasso.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_Lasso_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(lasso.summary())

    // --- Extract the non-zero features ---
    val coefficients = lasso.parameter // Get all the coefficients

    // Use a small threshold (1e-6) to account for floating-point inaccuracies
    val threshold = 1e-6

    // Get the column names for the non-zero coefficients
    val nonZeroFeatures = for {
    i <- xFname.indices
    if math.abs(coefficients(i)) > threshold
    } yield xFname(i)

    val nonZeroFeaturesList = nonZeroFeatures.toList
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    lasso = new LassoRegression(xZScoreOOS, yCenteredOOS, xFname)                // Re-instantiate Lasso Regression model
    val (ypOOS, qof2Full) = lasso.trainNtest(xTrainZScore, yTrainCentered)(xTestZScore, yTestCentered) 
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTestCentered, ypOOS)                    
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${lasso.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_Lasso_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(lasso.summary())                                                     

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)                  // Re-instantiate for CV
    val cvStats = lasso.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats, bestLambda, nonZeroFeaturesList)
end lassoReg


/**
 * Evaluates a Transformed Regression model applying a square root transformation.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation via index validation, and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param ox          The full input data matrix.
 * @param y           The full response vector.
 * @param oxTest      The testing input data matrix (passed to identity function).
 * @param oxTrain     The training input data matrix (passed to identity function).
 * @param yTest       The testing response vector.
 * @param yTrain      The training response vector (passed to identity function).
 * @param oxFname     The array of feature names.
 * @param folderName  The directory path where output plots will be saved.
 * @param idx         The indices mapping to the testing data.
 * @param scale       The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ          The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 */
def sqrtReg(ox: MatrixD,  y: VectorD, oxTest: MatrixD, oxTrain: MatrixD, yTest: VectorD, yTrain: VectorD, oxFname: Array[String], folderName: String, idx: scala.collection.mutable.IndexedSeq[Int], scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic]) =
    // Consume unused parameters to bypass compiler warnings without altering logic
    identity(oxTest)
    identity(oxTrain)
    identity(yTrain)

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var sqrtModel = new TranRegression(ox, y, oxFname, tran=sqrt, itran=sq)      // Instantiate Transformed Regression (Sqrt)
    val (ypIS, qof1Full) = sqrtModel.trainNtest()()                              // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${sqrtModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_Sqrt_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(sqrtModel.summary())                                                 
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    sqrtModel = new TranRegression(ox, y, oxFname, tran=sqrt, itran=sq)          // Re-instantiate Transformed Regression
    val (ypOOS, qof2Full) = sqrtModel.validate()(idx = idx)                      // Train on complement data, validate using idx
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${sqrtModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_Sqrt_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(sqrtModel.summary())                                                 

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    sqrtModel = new TranRegression(ox, y, oxFname, tran=sqrt, itran=sq)          // Re-instantiate for CV
    val cvStats = sqrtModel.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats)
end sqrtReg


/**
 * Evaluates a Transformed Regression model.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation via index validation, and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param ox          The full input data matrix.
 * @param y           The full response vector.
 * @param oxTest      The testing input data matrix (passed to identity function).
 * @param oxTrain     The training input data matrix (passed to identity function).
 * @param yTest       The testing response vector.
 * @param yTrain      The training response vector (passed to identity function).
 * @param oxFname     The array of feature names.
 * @param folderName  The directory path where output plots will be saved.
 * @param idx         The indices mapping to the testing data (passed to identity function).
 * @param scale       The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ          The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 */
def log1pReg(ox: MatrixD,  y: VectorD, oxTest: MatrixD, oxTrain: MatrixD, yTest: VectorD, yTrain: VectorD, oxFname: Array[String], folderName: String, idx: scala.collection.mutable.IndexedSeq[Int], scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic]) =
    // Consume unused parameters to bypass compiler warnings without altering logic
    identity(oxTest)
    identity(oxTrain)
    identity(yTrain)
    identity(idx)

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var log1pModel = new TranRegression(ox, y, oxFname)                          // Instantiate default Transformed Regression
    val (ypIS, qof1Full) = log1pModel.trainNtest()()                             // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${log1pModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_Log1p_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(log1pModel.summary())                                                
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    log1pModel = new TranRegression(ox, y, oxFname)                              // Re-instantiate Transformed Regression
    val (ypOOS, qof2Full) = log1pModel.validate()(idx = idx)                     // Train on complement data, validate using idx
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${log1pModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_Log1p_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(log1pModel.summary())                                                

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    log1pModel = new TranRegression(ox, y, oxFname)                              // Re-instantiate for CV
    val cvStats = log1pModel.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats)
end log1pReg


/**
 * Evaluates a Transformed Regression model utilizing the Box-Cox transformation.
 * * First tunes the transformation lambda power using a predefined grid search.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation via index validation, and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param ox          The full input data matrix.
 * @param y           The full response vector.
 * @param oxTest      The testing input data matrix (passed to identity function).
 * @param oxTrain     The training input data matrix (passed to identity function).
 * @param yTest       The testing response vector.
 * @param yTrain      The training response vector (passed to identity function).
 * @param oxFname     The array of feature names.
 * @param folderName  The directory path where output plots will be saved.
 * @param idx         The indices mapping to the testing data.
 * @param scale       The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ          The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 *                          - `bestLambda`: The optimal lambda parameter found via grid search.
 */
def boxCoxReg(ox: MatrixD,  y: VectorD, oxTest: MatrixD, oxTrain: MatrixD, yTest: VectorD, yTrain: VectorD, oxFname: Array[String], folderName: String, idx: scala.collection.mutable.IndexedSeq[Int], scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic], Double) =
    // Consume unused parameters to bypass compiler warnings without altering logic
    identity(oxTest)
    identity(oxTrain)
    identity(yTrain)

    val (bestLambda, _) = tuneBoxCoxLambda(ox, y, oxFname)
    modeling.TranRegression.λ = bestLambda                                       // Adjust transformation hyperparameter

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var boxCoxModel = new TranRegression(ox, y, oxFname, tran=box_cox, itran=cox_box) // Instantiate Box-Cox model
    val (ypIS, qof1Full) = boxCoxModel.trainNtest()()                            // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${boxCoxModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_BoxCox_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(boxCoxModel.summary())                                               
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    boxCoxModel = new TranRegression(ox, y, oxFname, tran=box_cox, itran=cox_box) // Re-instantiate Box-Cox model
    val (ypOOS, qof2Full) = boxCoxModel.validate()(idx = idx)                    // Train on complement data, validate using idx
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${boxCoxModel.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_BoxCox_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(boxCoxModel.summary())                                               

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    boxCoxModel = new TranRegression(ox, y, oxFname, tran=box_cox, itran=cox_box) // Re-instantiate for CV
    val cvStats = boxCoxModel.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats, bestLambda)
end boxCoxReg


/**
 * Evaluates an Order-2 (quadratic/polynomial) Ridge Regression model.
 * * First tunes the lambda (shrinkage) hyperparameter using a grid search.
 * * Performs In-Sample evaluation, Out-of-Sample evaluation (80-20 split), and 5-Fold Cross-Validation.
 * * Automatically generates and saves plots comparing actual vs. predicted values.
 *
 * @param xZScoreIS      The standardized full input data matrix.
 * @param yCenteredIS    The centered full response vector.
 * @param xZScoreOOS     The standardized full data matrix used for OOS initialization.
 * @param yCenteredOOS   The centered full response vector used for OOS initialization.
 * @param xTestZScore    The standardized testing input data matrix.
 * @param yTestCentered  The centered testing response vector.
 * @param xTrainZScore   The standardized training input data matrix.
 * @param yTrainCentered The centered training response vector.
 * @param xFname         The array of feature names.
 * @param folderName     The directory path where output plots will be saved.
 * @param scale          The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rQ             The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 * @return               A tuple containing the evaluation results:
 *                          - `In-Sample QoF`: The Quality of Fit metrics for the full dataset.
 *                          - `Out-of-Sample QoF`: The Quality of Fit metrics for the 80-20 test split.
 *                          - `CV Statistics`: The cross-validation evaluation metrics.
 *                          - `bestLambda`: The optimal shrinkage parameter found via grid search.
 */
def order2Reg(xZScoreIS: MatrixD, yCenteredIS: VectorD, xZScoreOOS: MatrixD, yCenteredOOS: VectorD, xTestZScore: MatrixD, yTestCentered: VectorD, xTrainZScore: MatrixD, yTrainCentered: VectorD, xFname: Array[String], folderName: String, scale: Double = 2, rQ: Range = 0 until 15): (VectorD, VectorD, Array[Statistic], Double) =
    val (bestLambda, _) = tuneRidgeLassoLambda("ridge", xZScoreIS, yCenteredIS, xFname)
    modeling.RidgeRegression.hp("lambda") = bestLambda                           // Set optimized shrinkage hyperparameter

    // --- In-Sample Evaluation ---
    banner("In-Sample")                  
    var ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)              // Instantiate Ridge Regression model
    val (ypIS, qof1Full) = ridge.trainNtest()()                                  // Train and test on the full dataset
    val qof1 = qof1Full(rQ)                                                      
    val (yOrd, ypOrdIS) = orderByY(yCenteredIS, ypIS)                            
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${ridge.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameIS = s"$folderName/Scalation_Order2Reg_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot, scale)                                    
    println(ridge.summary())                                                     
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")                                                        
    ridge = new RidgeRegression(xZScoreOOS, yCenteredOOS, xFname)                // Re-instantiate Ridge Regression model
    val (ypOOS, qof2Full) = ridge.trainNtest(xTrainZScore, yTrainCentered)(xTestZScore, yTestCentered) 
    val qof2 = qof2Full(rQ)                                                      
    val (yTestOrd, ypOrdOOS) = orderByY(yTestCentered, ypOOS)                    
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${ridge.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    
    Thread.sleep(1000)                                                           
    val fileNameOOS = s"$folderName/Scalation_Order2Reg_80_20.png"
    savePlot(fileNameOOS, plot8020, scale)                                       
    println(ridge.summary())                                                     

    // --- 5-Fold Cross-Validation ---
    banner("5-Fold CV")
    banner("Cross-Validation")
    ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)                  // Re-instantiate for CV
    val cvStats = ridge.crossValidate()

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    (qof1, qof2, cvStats, bestLambda)
end order2Reg