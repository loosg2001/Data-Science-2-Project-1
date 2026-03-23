// Place in modeling directory

package scalation
package modeling

import scalation.mathstat._
import scalation.modeling._
import scalation.mathstat.Scala2LaTeX._

/**
 * Orchestrator function to evaluate multiple regression models, perform feature selection,
 * and generate formatted outputs (LaTeX tables and plain text lists) for a given dataset.
 *
 * @param ox                 The full input data matrix (with intercept).
 * @param y                  The full response vector.
 * @param oxTest             The testing input data matrix (with intercept).
 * @param oxTrain            The training input data matrix (with intercept).
 * @param yTest              The testing response vector.
 * @param yTrain             The training response vector.
 * @param oxFname            The array of feature names (with intercept).
 * @param xZScoreIS          The standardized full input data matrix.
 * @param yCenteredIS        The centered full response vector.
 * @param xZScoreOOS         The standardized full input data matrix used for OOS initialization.
 * @param yCenteredOOS       The centered full response vector used for OOS initialization.
 * @param xTestZScore        The standardized testing input data matrix.
 * @param yTestCentered      The centered testing response vector.
 * @param xTrainZScore       The standardized training input data matrix.
 * @param yTrainCentered     The centered training response vector.
 * @param xFname             The array of feature names (without intercept).
 * @param x2ZScoreIS         The standardized quadratic/interaction input data matrix (Order 2).
 * @param x2ZScoreOOS        The standardized full Order 2 data matrix used for OOS initialization.
 * @param x2TestZScore       The standardized testing Order 2 input data matrix.
 * @param x2TrainZScore      The standardized training Order 2 input data matrix.
 * @param x2Fname            The array of feature names for the Order 2 matrix.
 * @param dataName           The descriptive name of the dataset (used for logging and titles).
 * @param folderName         The directory path where output plots will be saved.
 * @param idx                The indices mapping to the testing data.
 * @param scale              The scaling multiplier for high-resolution plotting (default is 2.0).
 * @param rq                 The range of Quality of Fit (QoF) metrics to extract (default is 0 until 15).
 */
def getTables (ox: MatrixD,  y: VectorD, oxTest: MatrixD, oxTrain: MatrixD, yTest: VectorD, yTrain: VectorD, oxFname: Array[String], xZScoreIS: MatrixD, yCenteredIS: VectorD, xZScoreOOS: MatrixD, yCenteredOOS: VectorD, xTestZScore: MatrixD, yTestCentered: VectorD, xTrainZScore: MatrixD, yTrainCentered: VectorD, xFname: Array[String], x2ZScoreIS: MatrixD, x2ZScoreOOS: MatrixD, x2TestZScore: MatrixD, x2TrainZScore: MatrixD, x2Fname: Array[String], dataName: String, folderName: String, idx: scala.collection.mutable.IndexedSeq[Int], scale: Double = 2, rq: Range = 0 until 15): Unit =

    // ==========================================
    // --- Model Evaluation ---
    // ==========================================
    banner(s"$dataName Regression")
    val (regIsQof, regOosQof, regCvStats) = linReg(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, folderName, scale, rq)

    banner(s"$dataName Ridge")
    val (ridgeIsQof, ridgeOosQof, ridgeCvStats, ridgeLambda) = ridgeReg(xZScoreIS, yCenteredIS, xZScoreOOS, yCenteredOOS, xTestZScore, yTestCentered, xTrainZScore, yTrainCentered, xFname, folderName, scale, rq)

    banner(s"$dataName Lasso")
    val (lassoIsQof, lassoOosQof, lassoCvStats, lassoLambda, lassoNonZeroFeatures) = lassoReg(xZScoreIS, yCenteredIS, xZScoreOOS, yCenteredOOS, xTestZScore, yTestCentered, xTrainZScore, yTrainCentered, xFname, folderName, scale, rq)

    banner(s"$dataName Sqrt")
    val (sqrtIsQof, sqrtOosQof, sqrtCvStats) = sqrtReg(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, folderName, idx, scale, rq)

    banner(s"$dataName Log1p")
    val (log1pIsQof, log1pOosQof, log1pCvStats) = log1pReg(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, folderName, idx, scale, rq)

    banner(s"$dataName Box-Cox")
    val (boxcoxIsQof, boxcoxOosQof, boxcoxCvStats, boxcoxLambda) = boxCoxReg(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, folderName, idx, scale, rq)

    banner(s"$dataName Order 2 Polynomial")
    val (order2regIsQof, order2regOosQof, order2regCvStats, order2regLambda) = order2Reg(x2ZScoreIS, yCenteredIS, x2ZScoreOOS, yCenteredOOS, x2TestZScore, yTestCentered, x2TrainZScore, yTrainCentered, x2Fname, folderName, scale, rq)

    // ==========================================
    // --- Feature Selection ---
    // ==========================================
    banner(s"$dataName Forward Selection")
    val (regColNamesFS, ridgeColNamesFS, lassoColNamesFS, sqrtregColNamesFS, log1pColNamesFS, boxcoxColNamesFS, order2regColNamesFS) = featureSelection ("Forward", ox,  y, oxFname, xZScoreIS, yCenteredIS, xFname, x2ZScoreIS, x2Fname, dataName, folderName, ridgeLambda, lassoLambda, boxcoxLambda, order2regLambda)

    banner(s"$dataName Backward Elimination")
    val (regColNamesBE, ridgeColNamesBE, lassoColNamesBE, sqrtregColNamesBE, log1pColNamesBE, boxcoxColNamesBE, order2regColNamesBE) = featureSelection ("Backward", ox,  y, oxFname, xZScoreIS, yCenteredIS, xFname, x2ZScoreIS, x2Fname, dataName, folderName, ridgeLambda, lassoLambda, boxcoxLambda, order2regLambda)

    banner(s"$dataName Stepwise Selection")
    val (regColNamesSS, ridgeColNamesSS, lassoColNamesSS, sqrtregColNamesSS, log1pColNamesSS, boxcoxColNamesSS, order2regColNamesSS) = featureSelection ("Stepwise", ox,  y, oxFname, xZScoreIS, yCenteredIS, xFname, x2ZScoreIS, x2Fname, dataName, folderName, ridgeLambda, lassoLambda, boxcoxLambda, order2regLambda)

    // ==========================================
    // --- Cross-Validation Tables ---
    // ==========================================
    banner(s"$dataName Regression CV Table")
    FitM.showQofStatTable (regCvStats)

    banner(s"$dataName Ridge CV Table")
    println (s"$dataName Ridge Lambda Used: $ridgeLambda")
    FitM.showQofStatTable (ridgeCvStats)

    banner(s"$dataName Lasso CV Table")
    println (s"$dataName Lasso Lambda Used: $lassoLambda")
    println (s"$dataName Lasso Selected Features: $lassoNonZeroFeatures")
    FitM.showQofStatTable (lassoCvStats)

    banner(s"$dataName Sqrt CV Table")
    FitM.showQofStatTable (sqrtCvStats)

    banner(s"$dataName Log1p CV Table")
    FitM.showQofStatTable (log1pCvStats)

    banner(s"$dataName Box-Cox CV Table")
    println (s"$dataName Box-Cox Lambda Used: $boxcoxLambda")
    FitM.showQofStatTable (boxcoxCvStats)

    banner(s"$dataName Order 2 Polynomial CV Table")
    println (s"$dataName Order 2 Polynomial Lambda Used: $order2regLambda")
    FitM.showQofStatTable (order2regCvStats)

    // ==========================================
    // --- LaTeX Table Generation ---
    // ==========================================
    val nQ = 15
    val rowName = modeling.qoF_names.take(nQ)

    val regColName = "Metric, In-Sample, 80-20 Split"
    val regCaption = s"Scalation - $dataName Linear Regression"
    val regName    = s"Scalation - $dataName Linear Regression"
    val regQofs    = MatrixD (regIsQof, regOosQof).transpose             // Create metrics for both point and interval predictions
    val regLatex   = make_doc (make_table (regCaption, regName, regQofs, regColName, rowName))
    println (regLatex)

    val ridgeColName = "Metric, In-Sample, 80-20 Split"
    val ridgeCaption = s"Scalation - $dataName Ridge Regression"
    val ridgeName    = s"Scalation - $dataName Ridge Regression"
    val ridgeQofs    = MatrixD (ridgeIsQof, ridgeOosQof).transpose      
    val ridgeLatex   = make_doc (make_table (ridgeCaption, ridgeName, ridgeQofs, ridgeColName, rowName))
    println (ridgeLatex)

    val lassoColName = "Metric, In-Sample, 80-20 Split"
    val lassoCaption = s"Scalation - $dataName Lasso Regression"
    val lassoName    = s"Scalation - $dataName Lasso Regression"
    val lassoQofs    = MatrixD (lassoIsQof, lassoOosQof).transpose      
    val lassoLatex   = make_doc (make_table (lassoCaption, lassoName, lassoQofs, lassoColName, rowName))
    println (lassoLatex)

    val sqrtColName = "Metric, In-Sample, 80-20 Split"
    val sqrtCaption = s"Scalation - $dataName Sqrt Transformation"
    val sqrtName    = s"Scalation - $dataName Sqrt Transformation"
    val sqrtQofs    = MatrixD (sqrtIsQof, sqrtOosQof).transpose         
    val sqrtLatex   = make_doc (make_table (sqrtCaption, sqrtName, sqrtQofs, sqrtColName, rowName))
    println (sqrtLatex)

    val log1pColName = "Metric, In-Sample, 80-20 Split"
    val log1pCaption = s"Scalation - $dataName Log1p Transformation"
    val log1pName    = s"Scalation - $dataName Log1p Transformation"
    val log1pQofs    = MatrixD (log1pIsQof, log1pOosQof).transpose      
    val log1pLatex   = make_doc (make_table (log1pCaption, log1pName, log1pQofs, log1pColName, rowName))
    println (log1pLatex)

    val boxcoxColName = "Metric, In-Sample, 80-20 Split"
    val boxcoxCaption = s"Scalation - $dataName Box-Cox Transformation"
    val boxcoxName    = s"Scalation - $dataName Box-Cox Transformation"
    val boxcoxQofs    = MatrixD (boxcoxIsQof, boxcoxOosQof).transpose   
    val boxcoxLatex   = make_doc (make_table (boxcoxCaption, boxcoxName, boxcoxQofs, boxcoxColName, rowName))
    println (boxcoxLatex)

    val order2regColName = "Metric, In-Sample, 80-20 Split"
    val order2regCaption = s"Scalation - $dataName Order 2 Polynomial Regression"
    val order2regName    = s"Scalation - $dataName Order 2 Polynomial Regression"
    val order2regQofs    = MatrixD (order2regIsQof, order2regOosQof).transpose 
    val order2regLatex   = make_doc (make_table (order2regCaption, order2regName, order2regQofs, order2regColName, rowName))
    println (order2regLatex)

    // --- Aggregate Comparison Tables ---
    val isColName = "Metric, Regression, Ridge, Lasso, Sqrt, Log1p,  Box-Cox, Order 2 Polynomial"
    val isCaption = s"Scalation: - $dataName In-Sample QoF Comparison"
    val isName = s"Scalation: - $dataName In-Sample QoF Comparison"
    val isQofMatrix = MatrixD (regIsQof, ridgeIsQof, lassoIsQof, sqrtIsQof, log1pIsQof, boxcoxIsQof, order2regIsQof).transpose // Transpose to align metrics as rows, models as columns
    val isLatex   = make_doc (make_table (isCaption, isName, isQofMatrix, isColName, rowName))
    println (isLatex)

    val oosColName = "Metric, Regression, Ridge, Lasso, Sqrt, Log1p, Box-Cox, Order 2 Polynomial"
    val oosCaption = s"Scalation: - $dataName Out-of-Sample QoF Comparison"
    val oosName = s"Scalation: - $dataName Out-of-Sample QoF Comparison"
    val oosQofMatrix = MatrixD (regOosQof, ridgeOosQof, lassoOosQof, sqrtOosQof, log1pOosQof, boxcoxOosQof, order2regOosQof).transpose 
    val oosLatex   = make_doc (make_table (oosCaption, oosName, oosQofMatrix, oosColName, rowName))
    println (oosLatex)

    // ==========================================
    // --- Feature Selection Results Printing ---
    // ==========================================
    banner(s"$dataName Regression Forward Selection Order")
    println (regColNamesFS)

    banner(s"$dataName Regression Backward Elimination Reversed Order")
    println (regColNamesBE)

    banner(s"$dataName Regression Stepwise Selection Order")
    println (regColNamesSS)

    banner(s"$dataName Ridge Forward Selection Order")
    println (ridgeColNamesFS)

    banner(s"$dataName Ridge Backward Elimination Reversed Order")
    println (ridgeColNamesBE)

    banner(s"$dataName Ridge Stepwise Selection Order")
    println (ridgeColNamesSS)

    banner(s"$dataName Lasso Forward Selection Order")
    println (lassoColNamesFS)

    banner(s"$dataName Lasso Backward Elimination Reversed Order")
    println (lassoColNamesBE)

    banner(s"$dataName Lasso Stepwise Selection Order")
    println (lassoColNamesSS)

    banner(s"$dataName Sqrt Forward Selection Order")
    println (sqrtregColNamesFS)

    banner(s"$dataName Sqrt Backward Elimination Reversed Order")
    println (sqrtregColNamesBE)

    banner(s"$dataName Sqrt Stepwise Selection Order")
    println (sqrtregColNamesSS)

    banner(s"$dataName Log1p Forward Selection Order")
    println (log1pColNamesFS)

    banner(s"$dataName Log1p Backward Elimination Reversed Order")
    println (log1pColNamesBE)

    banner(s"$dataName Log1p Stepwise Selection Order")
    println (log1pColNamesSS)

    banner(s"$dataName Box-Cox Forward Selection Order")
    println (boxcoxColNamesFS)

    banner(s"$dataName Box-Cox Backward Elimination Reversed Order")
    println (boxcoxColNamesBE)

    banner(s"$dataName Box-Cox Stepwise Selection Order")
    println (boxcoxColNamesSS)

    banner(s"$dataName Order 2 Polynomial Forward Selection Order")
    println (order2regColNamesFS)

    banner(s"$dataName Order 2 Polynomial Backward Elimination Reversed Order")
    println (order2regColNamesBE)

    banner(s"$dataName Order 2 Polynomial Stepwise Selection Order")
    println (order2regColNamesSS)

end getTables


/**
 * Main entry point for evaluating regression models on the Auto MPG dataset.
 * * Loads and preprocesses data (splitting, standardizing, and creating Order 2 features).
 * * Calls the `getTables` orchestrator to generate metrics, plots, and LaTeX summaries.
 */
@main def P1AutoMPG (): Unit =
    val oxFname = Array ("intercept", "displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")
    val xFname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_auto_mpg_with_intercept.csv", 1, sp=',')  // Load the dataset, skipping the header row
    val ox = oxy.not(?, 9)                                                       // Get the first 9 columns as the feature matrix
    val x = ox.not(?, 0)                                                         // Remove the intercept
    val y = oxy(?, 9)                                                            // Get the 10th column as the response vector
    val yy = MatrixD.fromVector (y)                                              // Turn the m-vector y into an m-by-1 matrix

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              // Make a permutation generator
    val nTest = (ox.dim * 0.2).toInt                                             // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           // Get test indices for 80-20 split

    val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  // TnT split the dataset ox (row split)
    val (xTest, xTrain) = TnT_Split (x, idx)                                     // TnT split the dataset x (row split)
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  // TnT split the response vector y (row split)
    val yTrain = yyTrain.col(0)                                                  // Get the train response vector from the test response matrix
    val yTest = yyTest.col(0)                                                    // Get the test response vector from the test response matrix

    // --- Standardization (Z-Score Normalization) ---
    val xZScoreIS = (x - x.mean(0)) / x.stdev(0)                                 // Center and scale the full input data
    val yCenteredIS  = (y - y.mean)                                              // Center the full target variable
    
    val xZScoreOOS = (x - xTrain.mean(0)) / xTrain.stdev(0)                      // Scale full data using training statistics (prevents leakage)
    val yCenteredOOS  = y - yTrain.mean                                          // Center full response using training mean
    val xTrainZScore = (xTrain - xTrain.mean(0)) / xTrain.stdev(0)               // Scale training data
    val yTrainCentered  = yTrain - yTrain.mean                                   // Center training response
    val xTestZScore = (xTest - xTrain.mean(0)) / xTrain.stdev(0)                 // Scale test data using training statistics
    val yTestCentered  = yTest - yTrain.mean                                     // Center test response using training mean

    // --- Order 2 Polynomial Feature Engineering ---
    val x2Fname = Array (
        "displacement", "cylinders", "horsepower", "weight", "acceleration", "model_year",
        "origin_1", "origin_2", "origin_3", "displacement x displacement", "displacement x cylinders",
        "displacement x horsepower", "displacement x weight", "displacement x acceleration",
        "displacement x model_year", "displacement x origin_1", "displacement x origin_2",
        "displacement x origin_3", "cylinders x cylinders", "cylinders x horsepower",
        "cylinders x weight", "cylinders x acceleration", "cylinders x model_year",
        "cylinders x origin_1", "cylinders x origin_2", "cylinders x origin_3",
        "horsepower x horsepower", "horsepower x weight", "horsepower x acceleration",
        "horsepower x model_year", "horsepower x origin_1", "horsepower x origin_2",
        "horsepower x origin_3", "weight x weight", "weight x acceleration", "weight x model_year",
        "weight x origin_1", "weight x origin_2", "weight x origin_3", "acceleration x acceleration",
        "acceleration x model_year", "acceleration x origin_1", "acceleration x origin_2",
        "acceleration x origin_3", "model_year x model_year", "model_year x origin_1",
        "model_year x origin_2", "model_year x origin_3", "mpg"
    )   

    val oxy2 = MatrixD.load ("cleaned_order_2_auto_mpg_with_intercept.csv", 1, sp=',')  
    val ox2 = oxy2.not(?, 49)                                                           
    val x2 = ox2.not(?, 0)                                                              

    val (x2Test, x2Train) = TnT_Split (x2, idx)                                         
    val x2ZScoreIS = (x2 - x2.mean(0)) / x2.stdev(0) 
    val x2ZScoreOOS = (x2 - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TrainZScore = (x2Train - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TestZScore = (x2Test - x2Train.mean(0)) / x2Train.stdev(0) 

    // --- Run Pipeline ---
    getTables(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, xZScoreIS, yCenteredIS, xZScoreOOS, yCenteredOOS, xTestZScore, yTestCentered, xTrainZScore, yTrainCentered, xFname, x2ZScoreIS, x2ZScoreOOS, x2TestZScore, x2TrainZScore, x2Fname, "Auto MPG", "Auto_MPG_Scalation_Plots", idx, 2.1)

    banner("Finished")

    System.exit(0)
end P1AutoMPG


/**
 * Main entry point for evaluating regression models on the California Housing dataset.
 * * Loads and preprocesses data (splitting, standardizing, and creating Order 2 features).
 * * Calls the `getTables` orchestrator to generate metrics, plots, and LaTeX summaries.
 */
@main def P1Housing (): Unit =
    val oxFname = Array("intercept", "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")
    val xFname = Array ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_housing_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 13)                                      
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 13)                                            
    val yy = MatrixD.fromVector (y)                                

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              
    val nTest = (ox.dim * 0.2).toInt                                             
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           

    val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  
    val (xTest, xTrain) = TnT_Split (x, idx)                                 
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  
    val yTrain = yyTrain.col(0)                                                      
    val yTest = yyTest.col(0)                                                        
    
    // --- Standardization (Z-Score Normalization) ---
    val xZScoreIS = (x - x.mean(0)) / x.stdev(0) 
    val yCenteredIS  = (y - y.mean)           
    
    val xZScoreOOS = (x - xTrain.mean(0)) / xTrain.stdev(0) 
    val yCenteredOOS  = y - yTrain.mean          
    val xTrainZScore = (xTrain - xTrain.mean(0)) / xTrain.stdev(0) 
    val yTrainCentered  = yTrain - yTrain.mean           
    val xTestZScore = (xTest - xTrain.mean(0)) / xTrain.stdev(0) 
    val yTestCentered  = yTest - yTrain.mean           

    // --- Order 2 Polynomial Feature Engineering ---
    val x2Fname = Array ("longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "ocean_proximity_<1H OCEAN", "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN", "longitude x longitude",
        "longitude x latitude", "longitude x housing_median_age", "longitude x total_rooms",
        "longitude x total_bedrooms", "longitude x population", "longitude x households",
        "longitude x median_income", "longitude x ocean_proximity_<1H OCEAN",
        "longitude x ocean_proximity_INLAND", "longitude x ocean_proximity_ISLAND",
        "longitude x ocean_proximity_NEAR BAY", "longitude x ocean_proximity_NEAR OCEAN",
        "latitude x latitude", "latitude x housing_median_age", "latitude x total_rooms",
        "latitude x total_bedrooms", "latitude x population", "latitude x households",
        "latitude x median_income", "latitude x ocean_proximity_<1H OCEAN",
        "latitude x ocean_proximity_INLAND", "latitude x ocean_proximity_ISLAND",
        "latitude x ocean_proximity_NEAR BAY", "latitude x ocean_proximity_NEAR OCEAN",
        "housing_median_age x housing_median_age", "housing_median_age x total_rooms",
        "housing_median_age x total_bedrooms", "housing_median_age x population",
        "housing_median_age x households", "housing_median_age x median_income",
        "housing_median_age x ocean_proximity_<1H OCEAN", "housing_median_age x ocean_proximity_INLAND",
        "housing_median_age x ocean_proximity_ISLAND", "housing_median_age x ocean_proximity_NEAR BAY",
        "housing_median_age x ocean_proximity_NEAR OCEAN", "total_rooms x total_rooms",
        "total_rooms x total_bedrooms", "total_rooms x population", "total_rooms x households",
        "total_rooms x median_income", "total_rooms x ocean_proximity_<1H OCEAN",
        "total_rooms x ocean_proximity_INLAND", "total_rooms x ocean_proximity_ISLAND",
        "total_rooms x ocean_proximity_NEAR BAY", "total_rooms x ocean_proximity_NEAR OCEAN",
        "total_bedrooms x total_bedrooms", "total_bedrooms x population",
        "total_bedrooms x households", "total_bedrooms x median_income",
        "total_bedrooms x ocean_proximity_<1H OCEAN", "total_bedrooms x ocean_proximity_INLAND",
        "total_bedrooms x ocean_proximity_ISLAND", "total_bedrooms x ocean_proximity_NEAR BAY",
        "total_bedrooms x ocean_proximity_NEAR OCEAN", "population x population",
        "population x households", "population x median_income",
        "population x ocean_proximity_<1H OCEAN", "population x ocean_proximity_INLAND",
        "population x ocean_proximity_ISLAND", "population x ocean_proximity_NEAR BAY",
        "population x ocean_proximity_NEAR OCEAN", "households x households",
        "households x median_income", "households x ocean_proximity_<1H OCEAN",
        "households x ocean_proximity_INLAND", "households x ocean_proximity_ISLAND",
        "households x ocean_proximity_NEAR BAY", "households x ocean_proximity_NEAR OCEAN",
        "median_income x median_income", "median_income x ocean_proximity_<1H OCEAN",
        "median_income x ocean_proximity_INLAND", "median_income x ocean_proximity_ISLAND",
        "median_income x ocean_proximity_NEAR BAY", "median_income x ocean_proximity_NEAR OCEAN"
    )   

    val oxy2 = MatrixD.load ("cleaned_order_2_housing_with_intercept.csv", 1, sp=',')      
    val ox2 = oxy2.not(?, 90)                                      
    val x2 = ox2.not(?, 0)                                       

    val (x2Test, x2Train) = TnT_Split (x2, idx)                                  
    val x2ZScoreIS = (x2 - x2.mean(0)) / x2.stdev(0) 
    val x2ZScoreOOS = (x2 - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TrainZScore = (x2Train - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TestZScore = (x2Test - x2Train.mean(0)) / x2Train.stdev(0) 

    // --- Run Pipeline ---
    getTables(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, xZScoreIS, yCenteredIS, xZScoreOOS, yCenteredOOS, xTestZScore, yTestCentered, xTrainZScore, yTrainCentered, xFname, x2ZScoreIS, x2ZScoreOOS, x2TestZScore, x2TrainZScore, x2Fname, "House Prices", "House_Prices_Scalation_Plots", idx, 2.3)

    banner("Finished")

    System.exit(0)
end P1Housing


/**
 * Main entry point for evaluating regression models on the Medical Insurance dataset.
 * * Loads and preprocesses data (splitting, standardizing, and creating Order 2 features).
 * * Calls the `getTables` orchestrator to generate metrics, plots, and LaTeX summaries.
 */
@main def P1Insurance (): Unit =
    val oxFname = Array("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val xFname = Array ("age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_insurance_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 9)                                       
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 9)                                             
    val yy = MatrixD.fromVector (y)                                

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              
    val nTest = (ox.dim * 0.2).toInt                                             
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           

    val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  
    val (xTest, xTrain) = TnT_Split (x, idx)                                 
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  
    val yTrain = yyTrain.col(0)                                                      
    val yTest = yyTest.col(0)                                                        
    
    // --- Standardization (Z-Score Normalization) ---
    val xZScoreIS = (x - x.mean(0)) / x.stdev(0) 
    val yCenteredIS  = (y - y.mean)           
    
    val xZScoreOOS = (x - xTrain.mean(0)) / xTrain.stdev(0) 
    val yCenteredOOS  = y - yTrain.mean          
    val xTrainZScore = (xTrain - xTrain.mean(0)) / xTrain.stdev(0) 
    val yTrainCentered  = yTrain - yTrain.mean           
    val xTestZScore = (xTest - xTrain.mean(0)) / xTrain.stdev(0) 
    val yTestCentered  = yTest - yTrain.mean           

    // --- Order 2 Polynomial Feature Engineering ---
    val x2Fname = Array ("intercept", "age", "bmi", "children", "sex_female", "sex_male", 
        "smoker_no", "smoker_yes", "region_northeast", "region_northwest", 
        "region_southeast", "region_southwest", "age x age", "age x bmi", 
        "age x children", "age x sex_female", "age x sex_male", "age x smoker_no", 
        "age x smoker_yes", "age x region_northeast", "age x region_northwest", 
        "age x region_southeast", "age x region_southwest", "bmi x bmi", 
        "bmi x children", "bmi x sex_female", "bmi x sex_male", "bmi x smoker_no", 
        "bmi x smoker_yes", "bmi x region_northeast", "bmi x region_northwest", 
        "bmi x region_southeast", "bmi x region_southwest", "children x children", 
        "children x sex_female", "children x sex_male", "children x smoker_no", 
        "children x smoker_yes", "children x region_northeast", "children x region_northwest", 
        "children x region_southeast", "children x region_southwest", "sex_female x smoker_no", 
        "sex_female x smoker_yes", "sex_female x region_northeast", "sex_female x region_northwest", 
        "sex_female x region_southeast", "sex_female x region_southwest", "sex_male x smoker_no", 
        "sex_male x smoker_yes", "sex_male x region_northeast", "sex_male x region_northwest", 
        "sex_male x region_southeast", "sex_male x region_southwest", "smoker_no x region_northeast", 
        "smoker_no x region_northwest", "smoker_no x region_southeast", "smoker_no x region_southwest", 
        "smoker_yes x region_northeast", "smoker_yes x region_northwest", "smoker_yes x region_southeast", 
        "smoker_yes x region_southwest"
    )   

    val oxy2 = MatrixD.load ("cleaned_order_2_insurance_with_intercept.csv", 1, sp=',')      
    val ox2 = oxy2.not(?, 62)                                      
    val x2 = ox2.not(?, 0)                                       

    val (x2Test, x2Train) = TnT_Split (x2, idx)                                  
    val x2ZScoreIS = (x2 - x2.mean(0)) / x2.stdev(0) 
    val x2ZScoreOOS = (x2 - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TrainZScore = (x2Train - x2Train.mean(0)) / x2Train.stdev(0) 
    val x2TestZScore = (x2Test - x2Train.mean(0)) / x2Train.stdev(0) 

    // --- Run Pipeline ---
    getTables(ox, y, oxTest, oxTrain, yTest, yTrain, oxFname, xZScoreIS, yCenteredIS, xZScoreOOS, yCenteredOOS, xTestZScore, yTestCentered, xTrainZScore, yTrainCentered, xFname, x2ZScoreIS, x2ZScoreOOS, x2TestZScore, x2TrainZScore, x2Fname, "Insurance Charges", "Insurance_Charges_Scalation_Plots", idx, 2.3)

    banner("Finished")

    System.exit(0)
end P1Insurance