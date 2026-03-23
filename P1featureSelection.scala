// Place in modeling directory

package scalation
package modeling

import scalation.mathstat._
import scala.math.sqrt
import modeling.TranRegression.{box_cox, cox_box}
import scala.collection.mutable.LinkedHashSet
import scalation.scala2d.savePlot
import java.awt.Window

/**
 * Performs feature selection (Forward, Backward, or Stepwise) across multiple regression models.
 * * Evaluates Linear, Ridge, Lasso, Square Root, Log1p, Box-Cox, and Order-2 Polynomial models.
 * * Calculates and tracks AIC and BIC metrics at each stage of the selection process.
 * * Automatically generates and saves high-resolution plots for R^2 vs. number of features, 
 * as well as AIC/BIC vs. number of features.
 *
 * @param key              The feature selection method to use: "Forward", "Backward", or "Stepwise".
 * @param ox               The standard input data matrix (usually with an intercept).
 * @param y                The response vector.
 * @param oxFname          The array of feature names for `ox`.
 * @param xZScoreIS        The standardized input data matrix.
 * @param yCenteredIS      The centered response vector.
 * @param xFname           The array of feature names for `xZScoreIS`.
 * @param x2ZScoreIS       The standardized quadratic/interaction input data matrix (Order 2).
 * @param x2Fname          The array of feature names for `x2ZScoreIS`.
 * @param dataName         The name of the dataset (used for plot titles and logging).
 * @param folderName       The directory path where output plots will be saved.
 * @param ridgeLambda      The optimized shrinkage hyperparameter for Ridge Regression.
 * @param lassoLambda      The optimized shrinkage hyperparameter for Lasso Regression.
 * @param boxcoxLambda     The optimized transformation hyperparameter for Box-Cox Regression.
 * @param order2regLambda  The optimized shrinkage hyperparameter for Order 2 Polynomial Regression.
 * @return                 A tuple containing lists of the selected feature names for each model: 
 * `(Linear, Ridge, Lasso, Sqrt, Log1p, BoxCox, Order2)`.
 */
def featureSelection(key: "Forward" | "Backward" | "Stepwise", ox: MatrixD, y: VectorD, oxFname: Array[String], xZScoreIS: MatrixD, yCenteredIS: VectorD, xFname: Array[String], x2ZScoreIS: MatrixD, x2Fname: Array[String], dataName: String, folderName: String, ridgeLambda: Double, lassoLambda: Double, boxcoxLambda: Double, order2regLambda: Double): (List[String], List[String], List[String], List[String], List[String], List[String], List[String]) =
    val method = 
        if key == "Forward" then
            "Forward Selection"
        else if key == "Backward" then
            "Backward Elimination"
        else if key == "Stepwise" then
            "Stepwise Selection"
        else
            ""
        end if

    // ==========================================
    // --- Linear Regression ---
    // ==========================================
    banner(s"$dataName $method for Regression")
    val reg = new Regression(ox, y, oxFname)                                       // Instantiate Regression model
    
    // Extract indices of selected features and R-squared values
    val (regCols, regRSq) = 
        if key == "Forward" then
            reg.forwardSelAll()
        else if key == "Backward" then
            reg.backwardElimAll()
        else if key == "Stepwise" then
            reg.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    // Calculate AIC and BIC for each stage of the selection process
    val regAicList = new VectorD(regCols.size)
    val regBicList = new VectorD(regCols.size)

    for k <- 0 until regCols.size do
        val regSubCols = regCols.slice(0, k + 1)                                   // Subset top k+1 features
        val regOxSub = ox(?, regSubCols)

        val subReg = new Regression(regOxSub, y)                                   // Fit model on subset
        val (_, regQof) = subReg.trainNtest()()
        
        regAicList(k) = regQof(13)                                                 // Store AIC
        regBicList(k) = regQof(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val regColsArray = regCols.toArray
    val regColsArrayD = regColsArray.map(_.toDouble)
    val regColsVecD = VectorD(regColsArrayD)                                       // Convert for display
    val regAicBicMatrix = MatrixD(regColsVecD, regAicList, regBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $regAicBicMatrix")

    // Generate and save plots
    if regCols.nonEmpty && regRSq.dim > 0 && regRSq.dim2 > 0 then
        val regRSqPlot = new PlotM(null, regRSq.ᵀ, Regression.metrics, s"R^2 vs n for Linear Regression $method on $dataName", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val regRSqFileName = s"$folderName/Scalation_Reg_rSq_$key.png"
        savePlot(regRSqFileName, regRSqPlot, 2.1)
        
        val regAicBicPlot = new PlotM(null, MatrixD(regAicList, regBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Linear Regression $method on $dataName", lines = true)
        Thread.sleep(1000)
        val regAicBicFileName = s"$folderName/Scalation_Reg_AIC_BIC_$key.png"
        savePlot(regAicBicFileName, regAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Linear Regression $method: No features were selected.")
    end if

    // Map column indices to feature names
    val regColsList = regColsArray.toList
    var regColNames = List.fill(regCols.size)("")
    for i <- 0 until regCols.size do
        regColNames = regColNames.updated(i, oxFname(regColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Ridge Regression ---
    // ==========================================
    banner(s"$dataName $method for Ridge Regression")
    modeling.RidgeRegression.hp("lambda") = ridgeLambda                            // Set shrinkage hyperparameter
    val ridge = new RidgeRegression(xZScoreIS, yCenteredIS, xFname)                // Instantiate Ridge Regression model
    
    val (ridgeCols, ridgeRSq) = 
        if key == "Forward" then
            ridge.forwardSelAll()
        else if key == "Backward" then
            ridge.backwardElimAll()
        else if key == "Stepwise" then
            ridge.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val ridgeAicList = new VectorD(ridgeCols.size)
    val ridgeBicList = new VectorD(ridgeCols.size)

    for k <- 0 until ridgeCols.size do
        val ridgeSubCols = ridgeCols.slice(0, k + 1)
        val ridgeXZScoreISSub = xZScoreIS(?, ridgeSubCols)

        val subRidge = new RidgeRegression(ridgeXZScoreISSub, yCenteredIS)
        val (_, ridgeQof) = subRidge.trainNtest()()
        
        ridgeAicList(k) = ridgeQof(13)
        ridgeBicList(k) = ridgeQof(14)
    end for

    banner("AIC/BIC History")
    val ridgeColsArray = ridgeCols.toArray
    val ridgeColsArrayD = ridgeColsArray.map(_.toDouble)
    val ridgeColsVecD = VectorD(ridgeColsArrayD)
    val ridgeAicBicMatrix = MatrixD(ridgeColsVecD, ridgeAicList, ridgeBicList)
    println(s"AIC/BIC History:      $ridgeAicBicMatrix")

    if ridgeCols.nonEmpty && ridgeRSq.dim > 0 && ridgeRSq.dim2 > 0 then
        val ridgeRSqPlot = new PlotM(null, ridgeRSq.ᵀ, Regression.metrics, s"R^2 vs n for Ridge Regression $method on $dataName", lines = true)
        Thread.sleep(1000)
        val ridgeRSqFileName = s"$folderName/Scalation_Ridge_rSq_$key.png"
        savePlot(ridgeRSqFileName, ridgeRSqPlot, 2.1)
        
        val ridgeAicBicPlot = new PlotM(null, MatrixD(ridgeAicList, ridgeBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Ridge Regression $method on $dataName", lines = true)
        Thread.sleep(1000)
        val ridgeAicBicFileName = s"$folderName/Scalation_Ridge_AIC_BIC_$key.png"
        savePlot(ridgeAicBicFileName, ridgeAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Ridge Regression $method: No features were selected.")
    end if

    val ridgeColsList = ridgeColsArray.toList
    var ridgeColNames = List.fill(ridgeCols.size)("")
    for i <- 0 until ridgeCols.size do
        ridgeColNames = ridgeColNames.updated(i, xFname(ridgeColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Lasso Regression ---
    // ==========================================
    banner(s"$dataName $method for Lasso Regression")
    modeling.RidgeRegression.hp("lambda") = lassoLambda                            // Set shrinkage hyperparameter
    val lasso = new LassoRegression(xZScoreIS, yCenteredIS, xFname)                // Instantiate Lasso Regression model
    
    val (lassoCols, lassoRSq) = 
        if key == "Forward" then
            lasso.forwardSelAll()
        else if key == "Backward" then
            lasso.backwardElimAll()
        else if key == "Stepwise" then
            lasso.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val lassoAicList = new VectorD(lassoCols.size)
    val lassoBicList = new VectorD(lassoCols.size)

    for k <- 0 until lassoCols.size do
        val lassoSubCols = lassoCols.slice(0, k + 1)
        val lassoXZScoreISSub = xZScoreIS(?, lassoSubCols)

        val subLasso = new LassoRegression(lassoXZScoreISSub, yCenteredIS)
        val (_, lassoQof) = subLasso.trainNtest()()
        
        lassoAicList(k) = lassoQof(13)
        lassoBicList(k) = lassoQof(14)
    end for

    banner("AIC/BIC History")
    val lassoColsArray = lassoCols.toArray
    val lassoColsArrayD = lassoColsArray.map(_.toDouble)
    val lassoColsVecD = VectorD(lassoColsArrayD)
    val lassoAicBicMatrix = MatrixD(lassoColsVecD, lassoAicList, lassoBicList)
    println(s"AIC/BIC History:      $lassoAicBicMatrix")

    if lassoCols.nonEmpty && lassoRSq.dim > 0 && lassoRSq.dim2 > 0 then
        val lassoRSqPlot = new PlotM(null, lassoRSq.ᵀ, Regression.metrics, s"R^2 vs n for Lasso Regression $method on $dataName", lines = true)
        Thread.sleep(1000)
        val lassoRSqFileName = s"$folderName/Scalation_Lasso_rSq_$key.png"
        savePlot(lassoRSqFileName, lassoRSqPlot, 2.1)
        
        val lassoAicBicPlot = new PlotM(null, MatrixD(lassoAicList, lassoBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Lasso Regression $method on $dataName", lines = true)
        Thread.sleep(1000)
        val lassoAicBicFileName = s"$folderName/Scalation_Lasso_AIC_BIC_$key.png"
        savePlot(lassoAicBicFileName, lassoAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Lasso Regression $method: No features were selected.")
    end if

    val lassoColsList = lassoColsArray.toList
    var lassoColNames = List.fill(lassoCols.size)("")
    for i <- 0 until lassoCols.size do
        lassoColNames = lassoColNames.updated(i, xFname(lassoColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Square Root Transformation ---
    // ==========================================
    banner(s"$dataName $method for Sqrt Transformation")
    val sqrtReg = new TranRegression(ox, y, oxFname, tran = sqrt, itran = sq)      // Instantiate Sqrt Transformed model
    
    val (sqrtRegCols, sqrtRegRSq) = 
        if key == "Forward" then
            sqrtReg.forwardSelAll()
        else if key == "Backward" then
            sqrtReg.backwardElimAll()
        else if key == "Stepwise" then
            sqrtReg.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val sqrtRegAicList = new VectorD(sqrtRegCols.size)
    val sqrtRegBicList = new VectorD(sqrtRegCols.size)

    for k <- 0 until sqrtRegCols.size do
        val sqrtRegSubCols = sqrtRegCols.slice(0, k + 1)
        val sqrtRegOxSub = ox(?, sqrtRegSubCols)

        val subSqrt = new TranRegression(sqrtRegOxSub, y, tran = sqrt, itran = sq)
        val (_, sqrtRegQof) = subSqrt.trainNtest()()
        
        sqrtRegAicList(k) = sqrtRegQof(13)
        sqrtRegBicList(k) = sqrtRegQof(14)
    end for

    banner("AIC/BIC History")
    val sqrtRegColsArray = sqrtRegCols.toArray
    val sqrtRegColsArrayD = sqrtRegColsArray.map(_.toDouble)
    val sqrtRegColsVecD = VectorD(sqrtRegColsArrayD)
    val sqrtRegAicBicMatrix = MatrixD(sqrtRegColsVecD, sqrtRegAicList, sqrtRegBicList)
    println(s"AIC/BIC History:      $sqrtRegAicBicMatrix")

    if sqrtRegCols.nonEmpty && sqrtRegRSq.dim > 0 && sqrtRegRSq.dim2 > 0 then
        val sqrtRSqPlot = new PlotM(null, sqrtRegRSq.ᵀ, Regression.metrics, s"R^2 vs n for Sqrt Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val sqrtRSqFileName = s"$folderName/Scalation_Sqrt_rSq_$key.png"
        savePlot(sqrtRSqFileName, sqrtRSqPlot, 2.1)
        
        val sqrtAicBicPlot = new PlotM(null, MatrixD(sqrtRegAicList, sqrtRegBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Sqrt Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val sqrtAicBicFileName = s"$folderName/Scalation_Sqrt_AIC_BIC_$key.png"
        savePlot(sqrtAicBicFileName, sqrtAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Sqrt Transformation $method: No features were selected.")
    end if

    val sqrtRegColsList = sqrtRegColsArray.toList
    var sqrtRegColNames = List.fill(sqrtRegCols.size)("")
    for i <- 0 until sqrtRegCols.size do
        sqrtRegColNames = sqrtRegColNames.updated(i, oxFname(sqrtRegColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Log1p Transformation ---
    // ==========================================
    banner(s"$dataName $method for Log1p Transformation")
    val log1pReg = new TranRegression(ox, y, oxFname)                              // Instantiate default (Log1p) Transformed model
    
    val (log1pCols, log1pRSq) = 
        if key == "Forward" then
            log1pReg.forwardSelAll()
        else if key == "Backward" then
            log1pReg.backwardElimAll()
        else if key == "Stepwise" then
            log1pReg.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val log1pAicList = new VectorD(log1pCols.size)
    val log1pBicList = new VectorD(log1pCols.size)

    for k <- 0 until log1pCols.size do
        val log1pSubCols = log1pCols.slice(0, k + 1)
        val log1pOxSub = ox(?, log1pSubCols)

        val subLog1p = new TranRegression(log1pOxSub, y)
        val (_, log1pQof) = subLog1p.trainNtest()()
        
        log1pAicList(k) = log1pQof(13)
        log1pBicList(k) = log1pQof(14)
    end for

    banner("AIC/BIC History")
    val log1pColsArray = log1pCols.toArray
    val log1pColsArrayD = log1pColsArray.map(_.toDouble)
    val log1pColsVecD = VectorD(log1pColsArrayD)
    val log1pAicBicMatrix = MatrixD(log1pColsVecD, log1pAicList, log1pBicList)
    println(s"AIC/BIC History:      $log1pAicBicMatrix")

    if log1pCols.nonEmpty && log1pRSq.dim > 0 && log1pRSq.dim2 > 0 then
        val log1pRSqPlot = new PlotM(null, log1pRSq.ᵀ, Regression.metrics, s"R^2 vs n for Log1p Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val log1pRSqFileName = s"$folderName/Scalation_Log1p_rSq_$key.png"
        savePlot(log1pRSqFileName, log1pRSqPlot, 2.1)
        
        val log1pAicBicPlot = new PlotM(null, MatrixD(log1pAicList, log1pBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Log1p Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val log1pAicBicFileName = s"$folderName/Scalation_Log1p_AIC_BIC_$key.png"
        savePlot(log1pAicBicFileName, log1pAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Log1p Transformation $method: No features were selected.")
    end if

    val log1pColsList = log1pColsArray.toList
    var log1pColNames = List.fill(log1pCols.size)("")
    for i <- 0 until log1pCols.size do
        log1pColNames = log1pColNames.updated(i, oxFname(log1pColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Box-Cox Transformation ---
    // ==========================================
    banner(s"$dataName $method for Box-Cox Transformation")
    modeling.TranRegression.λ = boxcoxLambda                                       // Set transformation hyperparameter                
    val boxCoxReg = new TranRegression(ox, y, oxFname, tran = box_cox, itran = cox_box) // Instantiate Box-Cox model
    
    val (boxCoxCols, boxCoxRSq) = 
        if key == "Forward" then
            boxCoxReg.forwardSelAll()
        else if key == "Backward" then
            boxCoxReg.backwardElimAll()
        else if key == "Stepwise" then
            boxCoxReg.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val boxCoxAicList = new VectorD(boxCoxCols.size)
    val boxCoxBicList = new VectorD(boxCoxCols.size)

    for k <- 0 until boxCoxCols.size do
        val boxCoxSubCols = boxCoxCols.slice(0, k + 1)
        val boxCoxOxSub = ox(?, boxCoxSubCols)

        val subBoxCox = new TranRegression(boxCoxOxSub, y, tran = box_cox, itran = cox_box)
        val (_, boxCoxQof) = subBoxCox.trainNtest()()
        
        boxCoxAicList(k) = boxCoxQof(13)
        boxCoxBicList(k) = boxCoxQof(14)
    end for

    banner("AIC/BIC History")
    val boxCoxColsArray = boxCoxCols.toArray
    val boxCoxColsArrayD = boxCoxColsArray.map(_.toDouble)
    val boxCoxColsVecD = VectorD(boxCoxColsArrayD)
    val boxCoxAicBicMatrix = MatrixD(boxCoxColsVecD, boxCoxAicList, boxCoxBicList)
    println(s"AIC/BIC History:      $boxCoxAicBicMatrix")

    if boxCoxCols.nonEmpty && boxCoxRSq.dim > 0 && boxCoxRSq.dim2 > 0 then
        val boxCoxRSqPlot = new PlotM(null, boxCoxRSq.ᵀ, Regression.metrics, s"R^2 vs n for Box-Cox Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val boxCoxRSqFileName = s"$folderName/Scalation_BoxCox_rSq_$key.png"
        savePlot(boxCoxRSqFileName, boxCoxRSqPlot, 2.1)
        
        val boxCoxAicBicPlot = new PlotM(null, MatrixD(boxCoxAicList, boxCoxBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Box-Cox Transformation $method on $dataName", lines = true)
        Thread.sleep(1000)
        val boxCoxAicBicFileName = s"$folderName/Scalation_BoxCox_AIC_BIC_$key.png"
        savePlot(boxCoxAicBicFileName, boxCoxAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Box-Cox Transformation $method: No features were selected.")
    end if

    val boxCoxColsList = boxCoxColsArray.toList
    var boxCoxColNames = List.fill(boxCoxCols.size)("")
    for i <- 0 until boxCoxCols.size do
        boxCoxColNames = boxCoxColNames.updated(i, oxFname(boxCoxColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Order 2 Polynomial ---
    // ==========================================
    banner(s"$dataName $method for Order 2 Polynomial")
    modeling.RidgeRegression.hp("lambda") = order2regLambda                        // Set shrinkage hyperparameter
    val order2Reg = new RidgeRegression(x2ZScoreIS, yCenteredIS, x2Fname)          // Instantiate Ridge Regression for Order 2
    
    val (order2RegCols, order2RegRSq) = 
        if key == "Forward" then
            order2Reg.forwardSelAll()
        else if key == "Backward" then
            order2Reg.backwardElimAll()
        else if key == "Stepwise" then
            order2Reg.stepwiseSelAll()
        else
            (LinkedHashSet[Int](1), MatrixD(1))
        end if

    val order2RegAicList = new VectorD(order2RegCols.size)
    val order2RegBicList = new VectorD(order2RegCols.size)

    for k <- 0 until order2RegCols.size do
        val order2RegSubCols = order2RegCols.slice(0, k + 1)
        val order2RegX2ZScoreISSub = x2ZScoreIS(?, order2RegSubCols)

        val subOrder2Reg = new RidgeRegression(order2RegX2ZScoreISSub, yCenteredIS)
        val (_, order2RegQof) = subOrder2Reg.trainNtest()()
        
        order2RegAicList(k) = order2RegQof(13)
        order2RegBicList(k) = order2RegQof(14)
    end for

    banner("AIC/BIC History")
    val order2RegColsArray = order2RegCols.toArray
    val order2RegColsArrayD = order2RegColsArray.map(_.toDouble)
    val order2RegColsVecD = VectorD(order2RegColsArrayD)
    val order2RegAicBicMatrix = MatrixD(order2RegColsVecD, order2RegAicList, order2RegBicList)
    println(s"AIC/BIC History:      $order2RegAicBicMatrix")

    if order2RegCols.nonEmpty && order2RegRSq.dim > 0 && order2RegRSq.dim2 > 0 then
        val order2RegRSqPlot = new PlotM(null, order2RegRSq.ᵀ, Regression.metrics, s"R^2 vs n for Order 2 Polynomial $method on $dataName", lines = true)
        Thread.sleep(1000)
        val order2RegRSqFileName = s"$folderName/Scalation_Order2Reg_rSq_$key.png"
        savePlot(order2RegRSqFileName, order2RegRSqPlot, 2.1)
        
        val order2RegAicBicPlot = new PlotM(null, MatrixD(order2RegAicList, order2RegBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for Order 2 Polynomial $method on $dataName", lines = true)
        Thread.sleep(1000)
        val order2RegAicBicFileName = s"$folderName/Scalation_Order2Reg_AIC_BIC_$key.png"
        savePlot(order2RegAicBicFileName, order2RegAicBicPlot, 2.1)
    else
        println(s"Skipping plots for Order 2 Polynomial $method: No features were selected.")
    end if

    val order2RegColsList = order2RegColsArray.toList
    var order2RegColNames = List.fill(order2RegCols.size)("")
    for i <- 0 until order2RegCols.size do
        order2RegColNames = order2RegColNames.updated(i, x2Fname(order2RegColsList(i)))
    end for

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // ==========================================
    // --- Return Selected Feature Lists ---
    // ==========================================
    (regColNames, ridgeColNames, lassoColNames, sqrtRegColNames, log1pColNames, boxCoxColNames, order2RegColNames)
end featureSelection