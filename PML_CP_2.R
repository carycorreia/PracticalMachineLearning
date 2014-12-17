### PLM: Course Project Working Notes
###
### Cary Correia
###
### December 14, 2014
###
###################################################################################################################################################
##
## Background
##
## Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data 
## about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – 
## a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in 
## their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular 
## activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers 
## on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 
## 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har 
## (see the section on the Weight Lifting Exercise Dataset). 
##
###################################################################################################################################################
## 
## Goals
##
## The goals of your project: are
## 1) to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 
##       You may use any of the other variables to predict with. 
##       You should create a report describing how you built your model, 
##       how you used cross validation, 
##       what you think the expected out of sample error is, 
##       and why you made the choices you did. 
## 2) You will also use your prediction model to predict 20 different test cases. 
##
###################################################################################################################################################
##
## Analytical Flow for the project
##
# DONE 1) download datasets; Training and Validation
# DONE 2) Clean and prep both datasets:
# DONE          - remove redundant columns + columns with all NA's in them
#               - impute any fields that may be missing small amounts of data
# Done          - split the data into a 'training' and 'testing' set
#       3) Figure out what data is not importanct
#               - exploratory plots
# Done          - preliminary random forests model to screen variables
#               - perform correlations to see if we can use PCA on some variables
#       4) Create another model to finalize the first one--> but only use the variables that pass section (3)
# Done          -  filter out non-important Vartables
#               -  split the 'training' data into two:  'training' and 'training.test' the latter will be for cross validation
#               -  complete further testing and finalize the models
#               -  calculate the 'in sample' error
#       5) Test the new model on the 'testing' dataset
# Done          - perform cross validation on this dataset
# Done          - calculate the 'out sample' error
# Done  6) Use the finalModel to predict the values in the sample test data (20 rows)
#
###################################################################################################################################################
#
#  Section 1) download datasets; Training and Testing

#Prep....setup libraries and set.seed
library(caret); library(rattle); library(ggplot2); library(randomForest); set.seed(12142014)    

#Set the train and test url's
data.url<-'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'        
test.url<-'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

#Get and save the data
data       <-read.csv(url(data.url), header=TRUE, sep=",", fill=TRUE, na.strings=c("NA", "", " "))               
validation <- read.csv(url(test.url), header=TRUE, sep=",", na.strings="NA")
###################################################################################################################################################
#
#  Section 2) Clean and prep both datasets

# Create a function to detect which column has lots of NA's - count #NA's / totalRows
na.test <- function(x,y) {
        w<-sapply(x, function(x)length(which(!is.na(x)))/length(x)*100)          # calculate %-age of rows that are NA in each col
        z<<-as.data.frame(sapply(w, function(x)if(x>=y) keep<-x else keep<-NA))  # creates a vector keep that has columns we will keep
        names(z)<<-c("value")
}
na.test(data, 99)                                                                # input-> dataset + threshold for non-NA's
keep<-as.data.frame(row.names(na.omit(z))); names(keep)<-'label'
keep<-as.character(keep$label[8:60])                                             # output-> vector of all rows to keep

data<-data[,names(data) %in% keep]                                               # downselect the variables we want 
nzero<-nearZeroVar(data, saveMetrics=TRUE)                                       # check all remaining variables to see if they are near zero
                                                                                 # no non zero's were found

inTrain    <-createDataPartition(data$classe, p=.2, list = FALSE)               # create inTrain string for data partitions   (60:40 split)
training   <-data[inTrain,]                                                      # split data into training, testing and validation
testing    <-data[-inTrain,]
###################################################################################################################################################
#
# Section 3) Figure out what data is not important
library(doMC)
registerDoMC(cores = 4)
# All subsequent models are then run in parallel

screen<-randomForest(classe ~., data=training, importance=TRUE, proximity=TRUE) # 1st random forest--> 4K obs with 53 variables....
vi<-importance(screen)                                          # Create varImportance matrix
plot(vi[,1:5])
vi_vals<-as.data.frame(vi[,1:5])
top_vi<-as.data.frame(apply(vi_vals, 2, rank))                                  # Rank each variable
top_vi$total<-rowSums(top_vi); top_sort<-top_vi[order(-top_vi$total),]          # Sort rank the variables by importance

# 4) Create another model to finalize the first one--> but only use the variables that pass section (3)

# cut off the bottom of the variables....include these for the actual full model run
keep2<-as.character(row.names(top_sort[top_sort$total>=88,]))
keep2[37]<-"classe"                                                             # Add back in the "classe" output

data2<-data[,names(data) %in% keep2]                                            # Build the dataset with just the top values

inTrain2    <-createDataPartition(data2$classe, p=.60, list = FALSE)            # create inTrain string for data partitions   (60:40 split)
training2   <-data2[inTrain2,]                                                  # split data into training, testing and validation
testing2    <-data2[-inTrain2,]

model<-train(classe ~., data=training2, method='rf', importance=TRUE)           # 1st random forest--> 12K obs with 36 variables

# Section 5) Test the new model on the 'testing' dataset

accuracy<-confusionMatrix(testing2$classe, predict(model, testing2))            # apply model to testing data
accuracy                                                                        # print out accuracy (99.06%)


# Section 6) Use the finalModel to predict the values in the sample test data (20 rows)

# prep validation data
keep2[37]<- "problem_id"
test_final<-validation[,names(validation) %in% keep2]

final.p<-as.character(predict(model, test_final))

## Results:  much faster run....RandomForest 1st pass got it done quick....Accuracy in Testing2-> 99.15%

                    
      