### Weight-lifting performance prediction with wearable accelerometer data
========================================================

#### Executive summary
This report aims to develop a performance prediction model on how well subjects perform dumbbell lifting exercises. The training dataset is based Velloso et al.(2013), which contains body sensing data of six health human male participants (age between 20-28 years) when they were asked to perform this exercise in five different ways. Only one of them is correct (class A) while the other four are common mistakes (classes B to E). We build a machine learning algorithm to predict activity quality from activity monitors with random forest method and data cross-validation approach: constructing prediction models (with 52 features) with a training set, construct and compare out-of-sample predictions with the test dataset, and apply the final model to predict the twenty test cases.   

#### Section 1. Data preprocessing
Data preprocessing consists of two main steps. Firstly, we clean the raw data by deleting variables with missing observations after loading raw data. We then further dividing the training data into a training set and a test set for cross validation purpose. Secondly, we preprocess the data by standardizing all variables.
To begin with, load packages and the entire dataset. 

```r
# Global settings: echo codes, present results and store existing results
# for all analysis
opts_chunk$set(echo = TRUE, results = "show", cache = TRUE)
# Load packages for data analysis
library(kernlab)
library(lattice)
library(ggplot2)
library(AppliedPredictiveModeling)
library(caret)
# Load training dataset
dataset <- read.table("pml-training.csv", header = TRUE, sep = ",", na.strings = "?", 
    stringsAsFactors = TRUE)
# dataset <- dataset[complete.cases(dataset$classe),] Load prediction
# dataset
predictset <- read.table("pml-testing.csv", header = TRUE, sep = ",", na.strings = "?", 
    stringsAsFactors = TRUE)
# predictset <- predictset[complete.cases(predictset$classe),]
```

Step 1. Delete variables with missing data
The entire dataset consists of 160 potential predictors (also called features) and 19220 observations. Nevertheless, many of these variables are nearly 100% missing obversations, and they are completely absent in the 20-case test set. Hence, we only develop training and test sets based on a subset of the predictors with complete data (exluding the first seven classifers since they are unrelated to body senor data). Then we use the majority of them are assigned as training data and the remaining as testing data.

```r
# non-NA column selections
subdata <- subset(dataset, select = c("roll_belt", "pitch_belt", "yaw_belt", 
    "total_accel_belt", "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", 
    "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", 
    "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
    "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", 
    "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
    "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", 
    "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", 
    "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", 
    "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", 
    "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", 
    "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "classe"))
# split data into training and testing sets:
inTrain = createDataPartition(y = subdata$classe, p = 0.7, list = FALSE)
training = subdata[inTrain, ]
testing = subdata[-inTrain, ]
# dim(training);dim(testing) str(training)
```

Step 2. Basic preprocessing - data standarization. The cleaned data subset contains 52 candidate features. We then standardized the data.

```r
preObj <- preProcess(training[, -53], method = c("center", "scale"))
preObj1 <- preProcess(testing[, -53], method = c("center", "scale"))
# posttraiing contains all standardized features without y variable
posttraining <- predict(preObj, training[, -53])
posttesting <- predict(preObj1, testing[, -53])
# strain contains standardized features with y variable
strain <- training
stest <- testing
strain[, -53] <- posttraining
stest[, -53] <- posttesting
# dim(training) dim(posttraining) dim(strain) summary(posttesting)
```



#### Section 2. Exploratory analysis on predictors
Step 1. Removing zero covariates. The result (not reported due to space) suggests that all candidate predictors have sufficient variations.

```r
nsv <- nearZeroVar(training, saveMetrics = TRUE)
nsv
```

Step 2. We apply the Principal Components Analysis (PCA) to include variables that captures most of the information amongst similar variables. The results suggest that a good number of variables are correlated, and hence it might be a good idea to weight them.

```r
M <- abs(cor(training[, -53]))
diag(M) <- 0
which(M > 0.7, arr.ind = T)
# names(training)[c(26,27)] plot(training[,26],training[,27])
```

We could also study the correlations of the variables. In some model specifications, we will drop some of the highly correlated variables (we hide the results  due to limited space).

```r
cor(posttraining)
```


#### Section 3. Random forests model selection
The model selection consists of two main steps. First, we use the training subsamples to fit several versions of the random forest models and obtain in-sample error rate. Second, we use the model to predict out-of-sample error rates of the testing subsamples in the training set for accuracy comparisons.

```r
# increase processing speed:
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
cl <- makeCluster(detectCores())
registerDoParallel(cl)

library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# Model 1. With all (52) predictors and the standardized dataset
strain <- na.roughfix(strain)
modelFit1 <- randomForest(classe ~ ., method = "rf", data = strain, importance = TRUE, 
    keep.forest = TRUE)
print(modelFit1)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = strain, method = "rf",      importance = TRUE, keep.forest = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.51%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    3    0    1    1    0.001280
## B   14 2640    4    0    0    0.006772
## C    0   15 2378    3    0    0.007513
## D    0    0   20 2231    1    0.009325
## E    0    0    2    6 2517    0.003168
```

```r
# Show 'imporance' of variables: higher value mean more important:
# round(importance(modelFit1),2)
test1 <- predict(modelFit1, posttesting, predict.all = FALSE)
# print(test1)
confusionMatrix(stest$classe, test1)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1666    3    0    4    1
##          B    3 1125    1    0   10
##          C    0   26  975    4   21
##          D    0    0   14  949    1
##          E    0    1    0    4 1077
## 
## Overall Statistics
##                                         
##                Accuracy : 0.984         
##                  95% CI : (0.981, 0.987)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.98          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.974    0.985    0.988    0.970
## Specificity             0.998    0.997    0.990    0.997    0.999
## Pos Pred Value          0.995    0.988    0.950    0.984    0.995
## Neg Pred Value          0.999    0.994    0.997    0.998    0.993
## Prevalence              0.284    0.196    0.168    0.163    0.189
## Detection Rate          0.283    0.191    0.166    0.161    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.986    0.987    0.992    0.985
```

```r


# Model 2. With all (52) predictors and the pre-standardized dataset
strain <- na.roughfix(training)
modelFit2 <- randomForest(classe ~ ., method = "rf", data = training, importance = TRUE, 
    keep.forest = TRUE)
print(modelFit2)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, method = "rf",      importance = TRUE, keep.forest = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.46%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3904    2    0    0    0    0.000512
## B   14 2641    3    0    0    0.006396
## C    0   13 2380    3    0    0.006678
## D    0    0   19 2230    3    0.009769
## E    0    0    2    4 2519    0.002376
```

```r
# Show 'imporance' of variables: higher value mean more important:
# round(importance(modelFit2),2)
test2 <- predict(modelFit2, testing, predict.all = FALSE)
head(test2)
```

```
##  1  8 12 24 25 26 
##  A  A  A  A  A  A 
## Levels: A B C D E
```

```r
# print(test2)
confusionMatrix(stest$classe, test2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671    3    0    0    0
##          B    2 1135    2    0    0
##          C    0    6 1017    3    0
##          D    0    0    8  955    1
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                         
##                Accuracy : 0.995         
##                  95% CI : (0.993, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.994         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.990    0.994    0.999
## Specificity             0.999    0.999    0.998    0.998    0.999
## Pos Pred Value          0.998    0.996    0.991    0.991    0.997
## Neg Pred Value          1.000    0.998    0.998    0.999    1.000
## Prevalence              0.284    0.194    0.175    0.163    0.184
## Detection Rate          0.284    0.193    0.173    0.162    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.996    0.994    0.996    0.999
```

```r

# Model 3. combination of Model 1 and 2
# qplot(test1,test2,colour=classe,data=stest)
predDF <- data.frame(test1, test2, dv = stest$classe)
head(predDF)
```

```
##    test1 test2 dv
## 1      A     A  A
## 8      A     A  A
## 12     A     A  A
## 24     A     A  A
## 25     A     A  A
## 26     A     A  A
```

```r
combModFit <- randomForest(dv ~ ., data = predDF, importance = TRUE, keep.forest = TRUE)
print(combModFit)
```

```
## 
## Call:
##  randomForest(formula = dv ~ ., data = predDF, importance = TRUE,      keep.forest = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 1
## 
##         OOB estimate of  error rate: 0.51%
## Confusion matrix:
##      A    B    C   D    E class.error
## A 1671    3    0   0    0    0.001792
## B    3 1134    2   0    0    0.004390
## C    0    6 1017   3    0    0.008772
## D    0    0    8 955    1    0.009336
## E    0    0    1   3 1078    0.003697
```

```r
combPred <- predict(combModFit, predDF)
# print(combPred)


# Model 4. With a subset of predictors (key elements)
modelFit4 <- randomForest(classe ~ roll_belt + pitch_belt + yaw_belt + accel_belt_z + 
    magnet_belt_x + roll_arm + gyros_arm_y + gyros_dumbbell_x + gyros_dumbbell_z + 
    accel_dumbbell_y + magnet_dumbbell_y + magnet_dumbbell_z + roll_forearm + 
    pitch_forearm + magnet_forearm_z, preProcess = "pca", data = strain, importance = TRUE, 
    keep.forest = TRUE)
print(modelFit4)
```

```
## 
## Call:
##  randomForest(formula = classe ~ roll_belt + pitch_belt + yaw_belt +      accel_belt_z + magnet_belt_x + roll_arm + gyros_arm_y + gyros_dumbbell_x +      gyros_dumbbell_z + accel_dumbbell_y + magnet_dumbbell_y +      magnet_dumbbell_z + roll_forearm + pitch_forearm + magnet_forearm_z,      data = strain, preProcess = "pca", importance = TRUE, keep.forest = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 0.68%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3894    9    1    2    0    0.003072
## B   13 2629   14    2    0    0.010910
## C    1   12 2371   12    0    0.010434
## D    0    0   19 2231    2    0.009325
## E    0    4    2    1 2518    0.002772
```

```r
test4 <- predict(modelFit4, posttesting, predict.all = FALSE)
# print(test1)
confusionMatrix(stest$classe, test4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1079    0    0  502   93
##          B  581    0    0  425  133
##          C  466    0    0  506   54
##          D  377    0    0  532   55
##          E  445    0    0  480  157
## 
## Overall Statistics
##                                         
##                Accuracy : 0.3           
##                  95% CI : (0.289, 0.312)
##     No Information Rate : 0.501         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.096         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.366       NA       NA   0.2176   0.3191
## Specificity             0.797    0.806    0.826   0.8744   0.8285
## Pos Pred Value          0.645       NA       NA   0.5519   0.1451
## Neg Pred Value          0.556       NA       NA   0.6113   0.9303
## Prevalence              0.501    0.000    0.000   0.4155   0.0836
## Detection Rate          0.183    0.000    0.000   0.0904   0.0267
## Detection Prevalence    0.284    0.194    0.174   0.1638   0.1839
## Balanced Accuracy       0.582       NA       NA   0.5460   0.5738
```

Trying with different models and different subsamples, we can see that model 2 has the highest in-sample error rate (less than 1%) and the highest out-of-sample prediction rate (over 99%). Reducing the number of predictors do not help in this case. The next step is to generate the real out-of-sample predictions for the 20 testing cases.

#### Section 3. Model prediction

```r
# obtain a subset of the useful features:
predictset <- subset(predictset, select = c("roll_belt", "pitch_belt", "yaw_belt", 
    "total_accel_belt", "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", 
    "accel_belt_y", "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", 
    "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
    "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", 
    "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
    "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", 
    "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", 
    "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", 
    "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", 
    "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", 
    "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z"))
# dim(predictset)

# Predictions using Model 2:
predictions2 <- predict(modelFit2, predictset, predict.all = FALSE)
print(predictions2)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r


# standardzied the prediction sample for the remaining models:
preObj1 <- preProcess(predictset, method = c("center", "scale"))
predictset <- predict(preObj1, predictset)
# str(predictset)


# Predictions using Model 1:
predictions1 <- predict(modelFit1, predictset, predict.all = FALSE)
print(predictions1)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  E  A  A  E  A  E  D  B  A  E  B  A  D  A  E  E  E  B  E  B 
## Levels: A B C D E
```

```r

# Predictions using Model 3 (Model 1&2's combinations:
pred1V <- predict(modelFit1, predictset)
pred2V <- predict(modelFit2, predictset)
predDF <- data.frame(test1 = pred1V, test2 = pred2V)
combPredV <- predict(combModFit, predDF)
print(combPredV)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  E  A  A  E  A  E  E  E  A  E  E  A  E  A  E  E  E  E  E  E 
## Levels: A B C D E
```

```r

# Predictions using Model 4:
predictions4 <- predict(modelFit4, predictset, predict.all = FALSE)
print(predictions4)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  A  A  A  D  A  D  D  D  A  A  A  A  D  A  D  D  A  D  A  D 
## Levels: A B C D E
```

It turns out that the answers generated by predictions2 have the highest prediction power. This model correctly predicts 20 out of 20 cases in the prediction sample.

#### Reference
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
