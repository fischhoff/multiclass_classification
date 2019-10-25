multiclass\_classification
================
Ilya
10/25/2019

\#\#based on code here: <https://rpubs.com/mharris/multiclass_xgboost>

\#\#\#\#\#install packages

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:xgboost':
    ## 
    ##     slice

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

Here is where we:

Load the RBGlass1 dataset convert the variable Site from a factor to
numeric Simulate a third class (furnace) from the data Bind the new
class to the original data Subtract 1 from the Site names so they start
at 0 Print out a summary() The XGBoost algorithm requires that the class
labels (Site names) start at 0 and increase sequentially to the maximum
number of classes. This is a bit of an inconvenience as you need to keep
track of what Site name goes with which label. Also, you need to be very
careful when you add or remove a 1 to go from the zero based labels to
the 1 based labels.

The simulated class is created by taking the data from Site == 1,
creating an offset ./10 and adding that back to the same value with some
random normal noise rnorm(1,.,.\*0.1) proportional to the value. You can
pretty much ignore this.

``` r
# set random seed
set.seed(717)
data(RBGlass1)
dat <- RBGlass1 
dat$Site <- as.numeric(dat$Site)
dat_add <- dat[which(dat$Site == 1),] %>%
  rowwise() %>%
  mutate_each(funs(./10 + rnorm(1,.,.*0.1))) %>%
  mutate_each(funs(round(.,2))) %>%
  mutate(Site = 3)
```

    ## Warning: funs() is soft deprecated as of dplyr 0.8.0
    ## Please use a list of either functions or lambdas: 
    ## 
    ##   # Simple named list: 
    ##   list(mean = mean, median = median)
    ## 
    ##   # Auto named with `tibble::lst()`: 
    ##   tibble::lst(mean, median)
    ## 
    ##   # Using lambdas
    ##   list(~ mean(., trim = .2), ~ median(., na.rm = TRUE))
    ## This warning is displayed once per session.

``` r
dat <- rbind(dat, dat_add) %>%
  mutate(Site = Site - 1)

summary(dat)
```

    ##       Site         Al              Fe               Mg        
    ##  Min.   :0   Min.   :2.060   Min.   :0.3200   Min.   :0.3900  
    ##  1st Qu.:0   1st Qu.:2.330   1st Qu.:0.4900   1st Qu.:0.5300  
    ##  Median :1   Median :2.450   Median :0.6550   Median :0.5500  
    ##  Mean   :1   Mean   :2.479   Mean   :0.6626   Mean   :0.5594  
    ##  3rd Qu.:2   3rd Qu.:2.592   3rd Qu.:0.8200   3rd Qu.:0.5800  
    ##  Max.   :2   Max.   :3.120   Max.   :1.3600   Max.   :0.7200  
    ##        Ca               Na              K                Ti        
    ##  Min.   : 4.890   Min.   :13.66   Min.   :0.4800   Min.   :0.0600  
    ##  1st Qu.: 6.360   1st Qu.:17.36   1st Qu.:0.6700   1st Qu.:0.0800  
    ##  Median : 6.825   Median :18.34   Median :0.7200   Median :0.1000  
    ##  Mean   : 6.998   Mean   :18.57   Mean   :0.7349   Mean   :0.0978  
    ##  3rd Qu.: 7.503   3rd Qu.:19.68   3rd Qu.:0.7725   3rd Qu.:0.1100  
    ##  Max.   :10.620   Max.   :24.58   Max.   :1.5400   Max.   :0.1600  
    ##        P                Mn               Sb               Pb         
    ##  Min.   :0.0900   Min.   :0.0700   Min.   :0.0000   Min.   :0.01000  
    ##  1st Qu.:0.1100   1st Qu.:0.2500   1st Qu.:0.0850   1st Qu.:0.02000  
    ##  Median :0.1300   Median :0.2800   Median :0.2700   Median :0.03000  
    ##  Mean   :0.1282   Mean   :0.3244   Mean   :0.2225   Mean   :0.03073  
    ##  3rd Qu.:0.1400   3rd Qu.:0.3825   3rd Qu.:0.3200   3rd Qu.:0.04000  
    ##  Max.   :0.2200   Max.   :0.9000   Max.   :0.5300   Max.   :0.09000

\#\#\#\#train and test split

``` r
# Make split index
train_index <- sample(1:nrow(dat), nrow(dat)*0.75)
# Full data set
data_variables <- as.matrix(dat[,-1])
data_label <- dat[,"Site"]
data_matrix <- xgb.DMatrix(data = as.matrix(dat), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)
```

\#\#\#\#K-folds Cross-validation to Estimate Error set the objective to
multi:softprob and the eval\_metric to mlogloss. These two parameters
tell the XGBoost algorithm that we want to to probabilistic
classification and use a multiclass logloss as our evaluation metric.
Use of the multi:softprob objective also requires that we tell is the
number of classes we have with num\_class

``` r
numberOfClasses <- length(unique(dat$Site))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)
```

\#\#\#\#Assess Out-of-Fold Prediction Error

``` r
OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
head(OOF_prediction)
```

    ##           X1          X2          X3 max_prob label
    ## 1 0.08967378 0.019084143 0.891242146        3     2
    ## 2 0.03876708 0.957451046 0.003781835        2     2
    ## 3 0.40387401 0.005780077 0.590345919        3     3
    ## 4 0.74101967 0.241375476 0.017604856        1     3
    ## 5 0.80737162 0.187277868 0.005350541        1     2
    ## 6 0.40329969 0.428810209 0.167890161        2     2

#### confusion matrix

``` r
# confusion matrix
confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3
    ##          1 34  6  5
    ##          2  7 25  1
    ##          3  4  2 39
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7967          
    ##                  95% CI : (0.7148, 0.8639)
    ##     No Information Rate : 0.3659          
    ##     P-Value [Acc > NIR] : <2e-16          
    ##                                           
    ##                   Kappa : 0.6922          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.9142          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity            0.7556   0.7576   0.8667
    ## Specificity            0.8590   0.9111   0.9231
    ## Pos Pred Value         0.7556   0.7576   0.8667
    ## Neg Pred Value         0.8590   0.9111   0.9231
    ## Precision              0.7556   0.7576   0.8667
    ## Recall                 0.7556   0.7576   0.8667
    ## F1                     0.7556   0.7576   0.8667
    ## Prevalence             0.3659   0.2683   0.3659
    ## Detection Rate         0.2764   0.2033   0.3171
    ## Detection Prevalence   0.3659   0.2683   0.3659
    ## Balanced Accuracy      0.8073   0.8343   0.8949

\#\#\#Train Full Model and Assess Test Set Error

``` r
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$max_prob),
                factor(test_prediction$label),
                mode = "everything")
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3
    ##          1 13  2  0
    ##          2  1 10  2
    ##          3  0  1 12
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8537          
    ##                  95% CI : (0.7083, 0.9443)
    ##     No Information Rate : 0.3415          
    ##     P-Value [Acc > NIR] : 1.862e-11       
    ##                                           
    ##                   Kappa : 0.7804          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity            0.9286   0.7692   0.8571
    ## Specificity            0.9259   0.8929   0.9630
    ## Pos Pred Value         0.8667   0.7692   0.9231
    ## Neg Pred Value         0.9615   0.8929   0.9286
    ## Precision              0.8667   0.7692   0.9231
    ## Recall                 0.9286   0.7692   0.8571
    ## F1                     0.8966   0.7692   0.8889
    ## Prevalence             0.3415   0.3171   0.3415
    ## Detection Rate         0.3171   0.2439   0.2927
    ## Detection Prevalence   0.3659   0.3171   0.3171
    ## Balanced Accuracy      0.9272   0.8310   0.9101

\#\#\#Variable Importance

``` r
# get the feature real names
names <-  colnames(dat[,-1])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)
```

    ##    Feature       Gain      Cover Frequency
    ## 1:      Na 0.19451014 0.16795320 0.1216407
    ## 2:      Ca 0.18628258 0.16455912 0.1456860
    ## 3:      Mg 0.18501053 0.14465977 0.1287129
    ## 4:      Fe 0.15362100 0.13760940 0.1216407
    ## 5:      Mn 0.10301000 0.10585287 0.1159830
    ## 6:      Al 0.08572768 0.09785265 0.1202263

``` r
# plot
gp = xgb.ggplot.importance(importance_matrix)
print(gp) 
```

![](multiclass_classification_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->
