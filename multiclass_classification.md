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

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

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
\#\#\#load and merge the 1) rodent trait data and 2) rodent - disease
data 3) disease – tx mode data 4) climate data

``` r
traits = read.csv("dfPanRodent_imputed.csv")
names(traits)
```

    ##  [1] "X"                             "X5.1_AdultBodyMass_g"         
    ##  [3] "X13.1_AdultHeadBodyLen_mm"     "X2.1_AgeatEyeOpening_d"       
    ##  [5] "X18.1_BasalMetRate_mLO2hr"     "X9.1_GestationLen_d"          
    ##  [7] "X22.1_HomeRange_km2"           "X15.1_LitterSize"             
    ##  [9] "X16.1_LittersPerYear"          "X5.3_NeonateBodyMass_g"       
    ## [11] "X21.1_PopulationDensity_n.km2" "X23.1_SexualMaturityAge_d"    
    ## [13] "X24.1_TeatNumber"              "X25.1_WeaningAge_d"           
    ## [15] "X26.1_GR_Area_km2"             "MSW05_Order"                  
    ## [17] "MSW05_Family"                  "MSW05_Genus"                  
    ## [19] "MSW05_Species"                 "MSW05_Binomial"

``` r
traits$Pan = paste(traits$MSW05_Genus,
                   traits$MSW05_Species,
                   sep = "_")

#host - rodent-disease pairs from GIDEON.csv
dx_host = read.csv("host - rodent-disease pairs from GIDEON.csv")

dx_host_traits = merge(traits, dx_host)

#host - rodent-disease transmission modes_20190413.csv
modes = read.csv("host - rodent-disease transmission modes_20190413.csv")
names(modes)[names(modes)=="Zoonoses.with.rodent.reservoirs"]="Matches"
modes = modes[,c("Matches",
                 "transmission.mode.to.humans.simplified")]
dx_modes_host_traits = merge(dx_host_traits,
                             modes, by = "Matches")

dx_modes_host_traits$Pan = str_replace(string = dx_modes_host_traits$Pan,pattern = "_",
            replacement = " ")

load("climate_envelope.rdata")

#these are the same length
length(intersect(xclim_all$binomial,dx_modes_host_traits$Pan))
```

    ## [1] 134

``` r
length(intersect(xclim_all$IUCN_binom,dx_modes_host_traits$Pan))
```

    ## [1] 134

``` r
names(xclim_all)[names(xclim_all)=="binomial"]="Pan"

dx_modes_host_traits_climate = merge(dx_modes_host_traits,xclim_all)
#remove fields no longer needed
rm = c("Matches", "Pan", "X",
       "MSW05_Order", "MSW05_Family","MSW05_Genus",                           
 "MSW05_Species","MSW05_Binomial","Flag" , "IUCN_binom")
keep = setdiff(names(dx_modes_host_traits_climate),rm)
dat = dx_modes_host_traits_climate[,keep]
```

\#\#change classes from 0 to n-1–
rodents

``` r
dat$transmission.mode.to.humans.simplified.numeric = as.numeric(factor(dat$transmission.mode.to.humans.simplified))
#direct: 1
#environmental: 2
#vector: 3
rm = c("transmission.mode.to.humans.simplified")
keep = setdiff(names(dat), rm)
dat = dat[,keep]
dat <- dat %>%
  mutate(transmission.mode.to.humans.simplified.numeric =  transmission.mode.to.humans.simplified.numeric- 1)

summary(dat)
```

    ##  X5.1_AdultBodyMass_g X13.1_AdultHeadBodyLen_mm X2.1_AgeatEyeOpening_d
    ##  Min.   :    6.99     Min.   : 59.0             Min.   : 0.00         
    ##  1st Qu.:   28.00     1st Qu.:100.7             1st Qu.:10.79         
    ##  Median :   67.99     Median :129.5             Median :14.19         
    ##  Mean   :  779.26     Mean   :173.3             Mean   :13.27         
    ##  3rd Qu.:  143.88     3rd Qu.:182.5             3rd Qu.:15.92         
    ##  Max.   :13406.27     Max.   :647.1             Max.   :31.50         
    ##                                                                       
    ##  X18.1_BasalMetRate_mLO2hr X9.1_GestationLen_d X22.1_HomeRange_km2
    ##  Min.   :  17.08           Min.   : 15.00      Min.   :0.0000667  
    ##  1st Qu.:  49.71           1st Qu.: 22.97      1st Qu.:0.0018191  
    ##  Median :  90.69           Median : 26.72      Median :0.0029109  
    ##  Mean   : 264.54           Mean   : 35.81      Mean   :0.0514994  
    ##  3rd Qu.: 154.91           3rd Qu.: 36.38      3rd Qu.:0.0066403  
    ##  Max.   :2746.80           Max.   :116.24      Max.   :0.7400000  
    ##                                                                   
    ##  X15.1_LitterSize X16.1_LittersPerYear X5.3_NeonateBodyMass_g
    ##  Min.   :0.970    Min.   : 1.000       Min.   :  0.770       
    ##  1st Qu.:2.520    1st Qu.: 2.098       1st Qu.:  2.500       
    ##  Median :3.392    Median : 3.019       Median :  3.799       
    ##  Mean   :3.505    Mean   : 3.261       Mean   : 42.092       
    ##  3rd Qu.:4.334    3rd Qu.: 3.867       3rd Qu.:  8.051       
    ##  Max.   :9.010    Max.   :10.000       Max.   :674.770       
    ##                                                              
    ##  X21.1_PopulationDensity_n.km2 X23.1_SexualMaturityAge_d X24.1_TeatNumber
    ##  Min.   :    6.32              Min.   : 13.57            Min.   : 3.790  
    ##  1st Qu.:  406.14              1st Qu.: 55.66            1st Qu.: 5.070  
    ##  Median :  674.58              Median : 84.81            Median : 6.160  
    ##  Mean   : 2020.46              Mean   :155.75            Mean   : 6.292  
    ##  3rd Qu.: 1837.13              3rd Qu.:152.55            3rd Qu.: 7.050  
    ##  Max.   :57067.85              Max.   :832.16            Max.   :16.000  
    ##                                                                          
    ##  X25.1_WeaningAge_d X26.1_GR_Area_km2       rf_Jan.V1     
    ##  Min.   :  8.45     Min.   :       2   Min.   :  1.29428  
    ##  1st Qu.: 23.17     1st Qu.:  424193   1st Qu.: 17.27286  
    ##  Median : 26.50     Median : 1097471   Median : 40.73360  
    ##  Mean   : 33.65     Mean   : 2456399   Mean   : 67.88814  
    ##  3rd Qu.: 32.11     3rd Qu.: 2493994   3rd Qu.: 93.44435  
    ##  Max.   :140.00     Max.   :18177352   Max.   :291.26840  
    ##                                        NA's   :1          
    ##       rf_Feb.V1          rf_Mar.V1           rf_Apr.V1     
    ##  Min.   :  0.91589   Min.   :  0.9902   Min.   :  0.21537  
    ##  1st Qu.: 15.39338   1st Qu.: 18.0890   1st Qu.: 23.48535  
    ##  Median : 36.16549   Median : 43.4557   Median : 44.03786  
    ##  Mean   : 61.46320   Mean   : 66.8400   Mean   : 67.55737  
    ##  3rd Qu.: 87.05641   3rd Qu.:104.9288   3rd Qu.:100.37049  
    ##  Max.   :301.02956   Max.   :319.9042   Max.   :266.12497  
    ##  NA's   :1           NA's   :1          NA's   :1          
    ##      rf_May.V1          rf_Jun.V1          rf_Jul.V1     
    ##  Min.   :  0.4042   Min.   :  0.1731   Min.   :  0.0171  
    ##  1st Qu.: 25.6636   1st Qu.: 26.5810   1st Qu.: 21.6361  
    ##  Median : 57.6512   Median : 58.6600   Median : 63.3298  
    ##  Mean   : 74.7024   Mean   : 81.7676   Mean   : 81.8082  
    ##  3rd Qu.:109.4927   3rd Qu.:120.0944   3rd Qu.:121.5557  
    ##  Max.   :347.3494   Max.   :414.2209   Max.   :395.5782  
    ##  NA's   :1          NA's   :1                            
    ##      rf_Aug.V1          rf_Sep.V1           rf_Oct.V1     
    ##  Min.   :  0.3537   Min.   :  0.3266   Min.   :  0.99362  
    ##  1st Qu.: 21.9235   1st Qu.: 26.6529   1st Qu.: 27.93551  
    ##  Median : 59.9326   Median : 56.3662   Median : 62.17851  
    ##  Mean   : 79.3716   Mean   : 79.9602   Mean   : 74.98394  
    ##  3rd Qu.:115.0148   3rd Qu.:117.7014   3rd Qu.:113.84933  
    ##  Max.   :370.6045   Max.   :342.9955   Max.   :266.09721  
    ##  NA's   :1          NA's   :1          NA's   :1          
    ##       rf_Nov.V1           rf_Dec.V1          rf_min         
    ##  Min.   :  1.03614   Min.   :  1.36195   Min.   :  0.01708  
    ##  1st Qu.: 19.84766   1st Qu.: 18.20516   1st Qu.:  5.60273  
    ##  Median : 51.08243   Median : 46.40158   Median : 20.41488  
    ##  Mean   : 68.40918   Mean   : 68.67415   Mean   : 28.98079  
    ##  3rd Qu.:109.80601   3rd Qu.:111.98526   3rd Qu.: 38.54648  
    ##  Max.   :250.01563   Max.   :244.65937   Max.   :156.81687  
    ##  NA's   :1           NA's   :1                              
    ##      rf_max           rf_mean            lst_Jan.V1     
    ##  Min.   :  1.867   Min.   :  1.867   Min.   :-26.88057  
    ##  1st Qu.: 64.972   1st Qu.: 33.937   1st Qu.:  3.51145  
    ##  Median :112.826   Median : 58.197   Median : 22.79524  
    ##  Mean   :129.052   Mean   : 72.396   Mean   : 15.67719  
    ##  3rd Qu.:191.133   3rd Qu.:106.297   3rd Qu.: 27.09480  
    ##  Max.   :414.221   Max.   :224.089   Max.   : 39.11428  
    ##                                      NA's   :1          
    ##      lst_Feb.V1          lst_Mar.V1          lst_Apr.V1    
    ##  Min.   :-23.27797   Min.   :-11.84861   Min.   :-1.37943  
    ##  1st Qu.:  6.83354   1st Qu.: 15.66428   1st Qu.:21.46083  
    ##  Median : 25.11674   Median : 26.66366   Median :25.82604  
    ##  Mean   : 17.61102   Mean   : 21.73921   Mean   :25.40631  
    ##  3rd Qu.: 28.17777   3rd Qu.: 29.68423   3rd Qu.:30.53653  
    ##  Max.   : 37.16087   Max.   : 39.25930   Max.   :45.38999  
    ##  NA's   :1           NA's   :1           NA's   :1         
    ##      lst_May.V1         lst_Jun.V1         lst_Jul.V1    
    ##  Min.   : 6.27075   Min.   : 1.89611   Min.   : 1.13235  
    ##  1st Qu.:23.24088   1st Qu.:24.52414   1st Qu.:24.59162  
    ##  Median :26.80737   Median :27.33964   Median :27.32254  
    ##  Mean   :27.60368   Mean   :28.61668   Mean   :29.07742  
    ##  3rd Qu.:31.12978   3rd Qu.:32.71567   3rd Qu.:34.42063  
    ##  Max.   :46.64552   Max.   :49.01211   Max.   :50.30423  
    ##  NA's   :1          NA's   :1          NA's   :2         
    ##      lst_Aug.V1         lst_Sep.V1         lst_Oct.V1    
    ##  Min.   : 3.47325   Min.   : 8.43120   Min.   :-4.01378  
    ##  1st Qu.:25.56972   1st Qu.:24.25955   1st Qu.:21.02343  
    ##  Median :27.91266   Median :27.82496   Median :26.66772  
    ##  Mean   :29.41525   Mean   :28.28848   Mean   :25.29385  
    ##  3rd Qu.:34.14683   3rd Qu.:31.94050   3rd Qu.:29.60187  
    ##  Max.   :51.10323   Max.   :48.28849   Max.   :43.18612  
    ##  NA's   :1          NA's   :1          NA's   :1         
    ##      lst_Nov.V1          lst_Dec.V1         lst_min           lst_max     
    ##  Min.   :-17.88531   Min.   :-25.21473   Min.   :-26.881   Min.   : -Inf  
    ##  1st Qu.: 13.35497   1st Qu.:  5.12436   1st Qu.:  2.831   1st Qu.:28.68  
    ##  Median : 24.55517   Median : 23.06760   Median : 17.915   Median :31.91  
    ##  Mean   : 20.66755   Mean   : 16.68506   Mean   :    Inf   Mean   : -Inf  
    ##  3rd Qu.: 28.98519   3rd Qu.: 27.73395   3rd Qu.: 24.615   3rd Qu.:36.52  
    ##  Max.   : 40.77923   Max.   : 39.30150   Max.   :    Inf   Max.   :51.10  
    ##  NA's   :1           NA's   :1                                            
    ##     lst_mean         sh_Jan            sh_Feb            sh_Mar      
    ##  Min.   :-2.51   Min.   : 0.6861   Min.   : 0.9473   Min.   : 1.660  
    ##  1st Qu.:19.74   1st Qu.: 3.2638   1st Qu.: 3.4635   1st Qu.: 4.017  
    ##  Median :26.03   Median : 6.1282   Median : 6.4110   Median : 6.957  
    ##  Mean   :23.84   Mean   : 8.0371   Mean   : 8.2238   Mean   : 8.650  
    ##  3rd Qu.:28.42   3rd Qu.:13.1616   3rd Qu.:13.5275   3rd Qu.:13.599  
    ##  Max.   :40.36   Max.   :18.1005   Max.   :18.0408   Max.   :18.427  
    ##  NA's   :1       NA's   :9         NA's   :9         NA's   :9       
    ##      sh_Apr           sh_May           sh_Jun           sh_Jul      
    ##  Min.   : 2.478   Min.   : 3.391   Min.   : 4.114   Min.   : 3.908  
    ##  1st Qu.: 4.989   1st Qu.: 5.847   1st Qu.: 6.882   1st Qu.: 7.965  
    ##  Median : 7.797   Median : 9.041   Median : 9.928   Median :10.402  
    ##  Mean   : 9.192   Mean   : 9.802   Mean   :10.580   Mean   :11.029  
    ##  3rd Qu.:13.675   3rd Qu.:13.282   3rd Qu.:13.924   3rd Qu.:14.262  
    ##  Max.   :18.184   Max.   :18.212   Max.   :20.417   Max.   :20.663  
    ##  NA's   :9        NA's   :9        NA's   :9        NA's   :9       
    ##      sh_Aug           sh_Sep           sh_Oct           sh_Nov      
    ##  Min.   : 3.987   Min.   : 3.874   Min.   : 2.480   Min.   : 1.418  
    ##  1st Qu.: 7.939   1st Qu.: 6.803   1st Qu.: 5.547   1st Qu.: 4.513  
    ##  Median : 9.998   Median :10.268   Median : 9.894   Median : 7.916  
    ##  Mean   :10.992   Mean   :10.354   Mean   : 9.667   Mean   : 8.811  
    ##  3rd Qu.:14.728   3rd Qu.:13.763   3rd Qu.:13.027   3rd Qu.:13.592  
    ##  Max.   :20.625   Max.   :19.588   Max.   :17.936   Max.   :17.617  
    ##  NA's   :9        NA's   :9        NA's   :9        NA's   :9       
    ##      sh_Dec            sh_min            sh_max          sh_mean      
    ##  Min.   : 0.8159   Min.   : 0.6861   Min.   : 5.697   Min.   : 3.266  
    ##  1st Qu.: 3.5329   1st Qu.: 3.3381   1st Qu.: 8.858   1st Qu.: 5.821  
    ##  Median : 6.4133   Median : 5.9238   Median :13.495   Median : 9.523  
    ##  Mean   : 8.2424   Mean   :    Inf   Mean   :   Inf   Mean   :   Inf  
    ##  3rd Qu.:13.7204   3rd Qu.:10.5277   3rd Qu.:16.425   3rd Qu.:13.528  
    ##  Max.   :18.0589   Max.   :    Inf   Max.   :   Inf   Max.   :   Inf  
    ##  NA's   :9                                                            
    ##     tair_Jan           tair_Feb          tair_Mar          tair_Apr     
    ##  Min.   :-21.2263   Min.   :-18.739   Min.   :-10.920   Min.   :-2.467  
    ##  1st Qu.: -0.5049   1st Qu.:  1.055   1st Qu.:  5.279   1st Qu.:10.100  
    ##  Median : 14.6227   Median : 16.205   Median : 17.523   Median :19.034  
    ##  Mean   : 10.6739   Mean   : 11.736   Mean   : 14.128   Mean   :16.724  
    ##  3rd Qu.: 22.6198   3rd Qu.: 22.954   3rd Qu.: 23.255   3rd Qu.:23.213  
    ##  Max.   : 27.9785   Max.   : 27.983   Max.   : 28.276   Max.   :31.225  
    ##  NA's   :9          NA's   :9         NA's   :9         NA's   :9       
    ##     tair_May        tair_Jun         tair_Jul          tair_Aug     
    ##  Min.   : 4.11   Min.   : 1.763   Min.   : 0.9713   Min.   : 1.712  
    ##  1st Qu.:14.68   1st Qu.:17.778   1st Qu.:18.5729   1st Qu.:18.422  
    ##  Median :18.95   Median :20.191   Median :22.3445   Median :21.996  
    ##  Mean   :18.73   Mean   :20.459   Mean   :21.4224   Mean   :21.499  
    ##  3rd Qu.:23.62   3rd Qu.:24.010   3rd Qu.:24.4549   3rd Qu.:24.615  
    ##  Max.   :34.75   Max.   :36.377   Max.   :38.1293   Max.   :37.876  
    ##  NA's   :9       NA's   :9        NA's   :9         NA's   :9       
    ##     tair_Sep         tair_Oct         tair_Nov          tair_Dec      
    ##  Min.   : 3.589   Min.   :-2.137   Min.   :-11.920   Min.   :-18.502  
    ##  1st Qu.:16.802   1st Qu.:12.416   1st Qu.:  7.131   1st Qu.:  1.587  
    ##  Median :19.978   Median :18.781   Median : 16.553   Median : 15.343  
    ##  Mean   :19.841   Mean   :17.279   Mean   : 13.959   Mean   : 11.470  
    ##  3rd Qu.:24.168   3rd Qu.:23.429   3rd Qu.: 22.442   3rd Qu.: 22.652  
    ##  Max.   :34.289   Max.   :29.352   Max.   : 27.281   Max.   : 27.932  
    ##  NA's   :9        NA's   :9        NA's   :9         NA's   :9        
    ##     tair_min          tair_max        tair_mean          sm_Jan     
    ##  Min.   : 0.6861   Min.   : 5.697   Min.   : 3.266   Min.   : 7.81  
    ##  1st Qu.: 3.3381   1st Qu.: 8.858   1st Qu.: 5.821   1st Qu.:21.13  
    ##  Median : 5.9238   Median :13.495   Median : 9.523   Median :26.69  
    ##  Mean   :    Inf   Mean   :   Inf   Mean   :   Inf   Mean   :26.04  
    ##  3rd Qu.:10.5277   3rd Qu.:16.425   3rd Qu.:13.528   3rd Qu.:31.40  
    ##  Max.   :    Inf   Max.   :   Inf   Max.   :   Inf   Max.   :40.79  
    ##                                                      NA's   :9      
    ##      sm_Feb           sm_Mar           sm_Apr           sm_May      
    ##  Min.   : 7.583   Min.   : 7.053   Min.   : 6.721   Min.   : 6.893  
    ##  1st Qu.:20.603   1st Qu.:20.316   1st Qu.:20.037   1st Qu.:20.907  
    ##  Median :25.942   Median :25.924   Median :26.541   Median :25.339  
    ##  Mean   :25.930   Mean   :25.557   Mean   :25.190   Mean   :24.715  
    ##  3rd Qu.:31.294   3rd Qu.:31.306   3rd Qu.:30.008   3rd Qu.:29.811  
    ##  Max.   :42.122   Max.   :41.258   Max.   :40.836   Max.   :37.962  
    ##  NA's   :9        NA's   :9        NA's   :9        NA's   :9       
    ##      sm_Jun           sm_Jul           sm_Aug           sm_Sep     
    ##  Min.   : 7.646   Min.   : 6.079   Min.   : 5.637   Min.   : 5.83  
    ##  1st Qu.:19.538   1st Qu.:17.348   1st Qu.:16.320   1st Qu.:16.95  
    ##  Median :24.823   Median :23.841   Median :22.968   Median :23.57  
    ##  Mean   :24.501   Mean   :23.901   Mean   :23.350   Mean   :23.62  
    ##  3rd Qu.:31.038   3rd Qu.:31.764   3rd Qu.:31.372   3rd Qu.:30.92  
    ##  Max.   :37.577   Max.   :37.223   Max.   :36.848   Max.   :37.86  
    ##  NA's   :9        NA's   :9        NA's   :9        NA's   :9      
    ##      sm_Oct           sm_Nov           sm_Dec          sm_min      
    ##  Min.   : 7.551   Min.   : 7.406   Min.   : 7.66   Min.   : 5.637  
    ##  1st Qu.:19.028   1st Qu.:19.941   1st Qu.:20.80   1st Qu.:15.026  
    ##  Median :24.996   Median :26.831   Median :27.34   Median :19.662  
    ##  Mean   :24.430   Mean   :25.321   Mean   :25.85   Mean   :   Inf  
    ##  3rd Qu.:31.141   3rd Qu.:30.655   3rd Qu.:30.61   3rd Qu.:25.272  
    ##  Max.   :37.380   Max.   :37.884   Max.   :40.26   Max.   :   Inf  
    ##  NA's   :9        NA's   :9        NA's   :9                       
    ##      sm_max         sm_mean     
    ##  Min.   :11.97   Min.   :11.97  
    ##  1st Qu.:25.89   1st Qu.:25.89  
    ##  Median :31.71   Median :31.71  
    ##  Mean   :  Inf   Mean   :  Inf  
    ##  3rd Qu.:35.46   3rd Qu.:35.46  
    ##  Max.   :  Inf   Max.   :  Inf  
    ##                                 
    ##  transmission.mode.to.humans.simplified.numeric
    ##  Min.   :0.000                                 
    ##  1st Qu.:1.000                                 
    ##  Median :1.000                                 
    ##  Mean   :1.291                                 
    ##  3rd Qu.:2.000                                 
    ##  Max.   :2.000                                 
    ## 

``` r
#direct: 0
#environmental: 1
#vector: 2
```

\#\#\#\#train and test split– rodents

``` r
# Make split index
train_index <- sample(1:nrow(dat), nrow(dat)*0.7)
# Full data set
label_col = which(names(dat)== "transmission.mode.to.humans.simplified.numeric")
data_variables <- as.matrix(dat[,-label_col])
data_label <- dat[,"transmission.mode.to.humans.simplified.numeric"]
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

\#\#\#\#K-folds Cross-validation to Estimate Error– rodents set the
objective to multi:softprob and the eval\_metric to mlogloss. These two
parameters tell the XGBoost algorithm that we want to to probabilistic
classification and use a multiclass logloss as our evaluation metric.
Use of the multi:softprob objective also requires that we tell is the
number of classes we have with
num\_class

``` r
numberOfClasses <- length(unique(dat$transmission.mode.to.humans.simplified.numeric))
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

\#\#\#\#Assess Out-of-Fold Prediction Error– rodents

``` r
OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = train_label + 1)
head(OOF_prediction)
```

    ##            X1        X2          X3 max_prob label
    ## 1 0.607808471 0.3251713 0.067020193        1     3
    ## 2 0.003221139 0.9929199 0.003859009        2     2
    ## 3 0.008040609 0.4222774 0.569681942        3     2
    ## 4 0.004968290 0.9713011 0.023730567        2     2
    ## 5 0.032687664 0.1280962 0.839216113        3     3
    ## 6 0.155002907 0.3529012 0.492095888        3     2

#### confusion matrix– rodents

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
    ##          1  2  1  2
    ##          2  5 48 17
    ##          3  1 16 23
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.6348          
    ##                  95% CI : (0.5399, 0.7226)
    ##     No Information Rate : 0.5652          
    ##     P-Value [Acc > NIR] : 0.07839         
    ##                                           
    ##                   Kappa : 0.3055          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.38698         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity           0.25000   0.7385   0.5476
    ## Specificity           0.97196   0.5600   0.7671
    ## Pos Pred Value        0.40000   0.6857   0.5750
    ## Neg Pred Value        0.94545   0.6222   0.7467
    ## Precision             0.40000   0.6857   0.5750
    ## Recall                0.25000   0.7385   0.5476
    ## F1                    0.30769   0.7111   0.5610
    ## Prevalence            0.06957   0.5652   0.3652
    ## Detection Rate        0.01739   0.4174   0.2000
    ## Detection Prevalence  0.04348   0.6087   0.3478
    ## Balanced Accuracy     0.61098   0.6492   0.6574

\#\#\#Train Full Model and Assess Test Set Error– rodents

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
    ##          1  0  2  0
    ##          2  3 18  4
    ##          3  2  6 15
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.66            
    ##                  95% CI : (0.5123, 0.7879)
    ##     No Information Rate : 0.52            
    ##     P-Value [Acc > NIR] : 0.0320          
    ##                                           
    ##                   Kappa : 0.3942          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.4575          
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3
    ## Sensitivity            0.0000   0.6923   0.7895
    ## Specificity            0.9556   0.7083   0.7419
    ## Pos Pred Value         0.0000   0.7200   0.6522
    ## Neg Pred Value         0.8958   0.6800   0.8519
    ## Precision              0.0000   0.7200   0.6522
    ## Recall                 0.0000   0.6923   0.7895
    ## F1                        NaN   0.7059   0.7143
    ## Prevalence             0.1000   0.5200   0.3800
    ## Detection Rate         0.0000   0.3600   0.3000
    ## Detection Prevalence   0.0400   0.5000   0.4600
    ## Balanced Accuracy      0.4778   0.7003   0.7657

\#\#\#Variable Importance– rodents

``` r
# get the feature real names
label_col = which(names(dat)== "transmission.mode.to.humans.simplified.numeric")
names <-  colnames(dat[,-label_col])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)
```

    ##                      Feature       Gain      Cover  Frequency
    ## 1:                  tair_Aug 0.07191795 0.05579411 0.03395062
    ## 2:                  lst_mean 0.07073743 0.03180607 0.01080247
    ## 3:                    rf_Nov 0.05570408 0.03223266 0.02160494
    ## 4:                    sm_May 0.05287974 0.03803149 0.02469136
    ## 5: X18.1_BasalMetRate_mLO2hr 0.04921744 0.02499279 0.02777778
    ## 6:                  tair_Jul 0.04421333 0.03838124 0.03240741

``` r
# plot
gp = xgb.ggplot.importance(importance_matrix, top_n = 20)
print(gp) 
```

![](multiclass_classification_files/figure-gfm/var_imp-1.png)<!-- -->

\#\#\#randomForest

``` r
#dat = subset(dat, transmission.mode.to.humans.simplified.numeric!=0)

#remove rows which have empty values
row.has.na <- apply(dat, 1, function(x){any(is.na(x))})
predictors_no_NA <- dat[!row.has.na, ]
dat = predictors_no_NA
dat$transmission.mode.to.humans.simplified.numeric=factor(dat$transmission.mode.to.humans.simplified.numeric)
# get the feature real names
label_col = which(names(dat)== "transmission.mode.to.humans.simplified.numeric")

names <-  colnames(dat[,-label_col])
y_col = label_col


model<-as.formula(paste(colnames(dat)[y_col], "~",
                        paste(names,collapse = "+"),
                        sep = ""))


#get train and test
DP =createDataPartition(y = dat$transmission.mode.to.humans.simplified.numeric, 
                        p = 0.8,
                        list = FALSE)
Train = dat[DP,]
Test = dat[-DP,]

model.rf = randomForest(model, data=Train, ntree=5000, mtry=15, importance=TRUE)
print(model.rf)
```

    ## 
    ## Call:
    ##  randomForest(formula = model, data = Train, ntree = 5000, mtry = 15,      importance = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 5000
    ## No. of variables tried at each split: 15
    ## 
    ##         OOB estimate of  error rate: 33.07%
    ## Confusion matrix:
    ##   0  1  2 class.error
    ## 0 0  7  4   1.0000000
    ## 1 1 58 13   0.1944444
    ## 2 1 16 27   0.3863636

``` r
#get predicted
Test$pred = predict(model.rf, Test, type="response")

table(Test$transmission.mode.to.humans.simplified.numeric, 
      Test$pred)
```

    ##    
    ##      0  1  2
    ##   0  1  1  0
    ##   1  0 11  6
    ##   2  0  6  4

``` r
varImpPlot(model.rf,type=2, n.var = 20)
```

![](multiclass_classification_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->
