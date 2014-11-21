#Course Project
library(caret)
library(randomForest)

#Get the training and testing data files
homedir <- getwd()
trainURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
testURL <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
trainfile <- paste0(homedir, '/PracticalMachineLearning/CourseProject/pml-training.csv')
testfile <- paste0(homedir, '/PracticalMachineLearning/CourseProject/pml-testing.csv')
#download.file(trainURL, destfile = trainfile, method = 'curl')
#download.file(testURL, destfile = testfile, method = 'curl')

#Load the training and testing data
train <- read.csv(file = trainfile)
test <- read.csv(file = testfile)

#Remove predictors that are very unlikely to have an effect on a model because they
#lack variabilty in comparison to the number of observations
cleantrain <- train[,-nearZeroVar(train)]
dim(cleantrain)
#[1] 19622  100

#Remove variables that have a significant (at least 5000) NAs - in most cases, these are
#aggregate variables in the data anyway
cleantrain <- cleantrain[, which(as.numeric(colSums(is.na(cleantrain))) < 5000)]
dim(cleantrain)
#[1] 19622   59

#Then remove the X variable as it is just an index column.  Also, remove the username and timestamps
cleantrain <- cleantrain[,-c(1:6)]
dim(cleantrain)
#[1] 19622    53

#Set the seed for creating a data partition in the now cleaned training data
set.seed(12345)

#Partition the cleantrain data into a training set (mytrain) and a testing set (mytest)
inTrain <- createDataPartition(y = cleantrain$classe, p = 0.5, list = FALSE)
mytrain <- cleantrain[inTrain,]
mytest <- cleantrain[-inTrain,]

#Fit a model using randomForest
modFit <- randomForest(classe ~ ., data = mytrain)
#varImpPlot(modFit)

predictTrain <- predict(modFit, mytrain)
#confusionMatrix(predictTrain, mytrain$classe)

#Test the model
predictTest <- predict(modFit, mytest)
confusionMatrix(predictTest, mytest$classe)

#Run the model of the validation set
#answers <- as.character(predict(modFit, test))

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

setwd(paste0(homedir, '/PracticalMachineLearning/CourseProject/Answers'))
pml_write_files(answers)
setwd(homedir)