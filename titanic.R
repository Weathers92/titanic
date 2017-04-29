#GIT commands to upload changes:
#git status
#git add .
#git status
#git commit -m "edit r file"
#git status
#git push origin master

###########################
### Reading in the Data ###
###########################

#clear working directory
rm(list = ls())

setwd("~/Kaggle Competitions/Titanic/titanic")

library(data.table)
train <- fread("data/train.csv")
test <- fread("data/test.csv")

######################
### Missing Values ###
######################

#A function to find the percent of missing observations.
pMiss <- function(x){(sum(is.na(x)) / length(x)) * 100}

#Detect the percent of missing values within each column
apply(train, 2, pMiss) #19.86532% of ages are missing.
apply(test, 2, pMiss) #20.5741627% of ages are missing and one fare is missing.

#Remove the Survived column
tempTrain <- subset(train, select = -Survived)

#Combine the training and test datasets.
combined <- rbind(tempTrain, test)

#Remove variables: PassengerId, Name, and Ticket
combined <- subset(combined, select = -c(PassengerId, Name, Ticket))

#Change blanks to none.
for(i in 1:nrow(combined))
{
  if(combined$Cabin[i] == "")
  {
    combined$Cabin[i] = "none"
  }
}

#Coerce into factor variable.
combined$Pclass <- as.factor(combined$Pclass)
combined$Sex <- as.factor(combined$Sex)
combined$Cabin <- as.factor(combined$Cabin)
combined$Embarked <- as.factor(combined$Embarked)

#Coerce into numeric variable.
combined$SibSp <- as.numeric(combined$SibSp)
combined$Parch <- as.numeric(combined$Parch)

library(mice)

#Impute values using predictive mean matching.
imputeVals1 <- mice(combined, m = 50, method = 'pmm', seed = 0692)

completed1 <- complete(imputeVals1, 1)

compTest1 <- completed1[892:1309,]
compTrain1 <- completed1[1:891,]
compTrain1 <- cbind(compTrain1, train$Survived)
names(compTrain1)[9] <- "Survived"

imputeVals2 <- mice(combined, m = 50, method = 'rf', seed = 0692)

completed2 <- complete(imputeVals2, 1)

compTest2 <- completed2[892:1309,]
compTrain2 <- completed2[1:891,]
compTrain2 <- cbind(compTrain2, train$Survived)
names(compTrain2)[9] <- "Survived"

######################
### Random Forests ###
######################

library(randomForest)

rf1 <- randomForest(Survived ~ ., importance = TRUE, data = compTrain1, set.seed(0692))
varImpPlot(price.rf, scale = FALSE)

rf2 <- randomForest(Survived ~ ., importance = TRUE, data = compTrain2, set.seed(0692))


