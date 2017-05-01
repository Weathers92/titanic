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

train$Survived <- as.factor(train$Survived)

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

#Create a new column indicating how many cabins a person purchased.
combined$nCabin <- 0

for(i in 1:nrow(combined))
{
  combined$nCabin[i] <- length(strsplit(x = combined$Cabin[i], split = " ")[[1]])
}

#Change blanks to none.
for(i in 1:nrow(combined))
{
  if(combined$Cabin[i] == "")
  {
    combined$Cabin[i] = "none"
  }
  if(combined$Embarked[i] == "")
  {
    combined$Embarked[i] = NA
  }
}

#Replace each of the cabin numbers with just their levels.
alpha <- c("A", "B", "C", "D", "E", "F", "G")

for(i in 1:length(alpha))
{
  combined$Cabin[grepl(alpha[i], combined$Cabin, ignore.case = FALSE)] <- alpha[i]
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
names(compTrain1)[10] <- "Survived"

imputeVals2 <- mice(combined, m = 50, method = 'rf', seed = 0692)

completed2 <- complete(imputeVals2, 1)

compTest2 <- completed2[892:1309,]
compTrain2 <- completed2[1:891,]
compTrain2 <- cbind(compTrain2, train$Survived)
names(compTrain2)[10] <- "Survived"

######################
### Random Forests ###
######################

library(randomForest)

rf1 <- randomForest(Survived ~ ., importance = TRUE, data = compTrain1, set.seed(0692))
varImpPlot(rf1, scale = FALSE)
rfpred1 <- predict(rf1, compTest1)
write.csv(rfpred1, file = "titanic_survivalrf1.csv")
#Your submission scored 0.73206.
#Position 6269.

rf2 <- randomForest(Survived ~ ., importance = TRUE, data = compTrain2, set.seed(0692))
varImpPlot(rf2, scale = FALSE)
rfpred2 <- predict(rf2, compTest2)
write.csv(rfpred2, file = "titanic_survivalrf2.csv")
#Your submission scored 0.74163.
#Position 6155.

###############################
### Support Vector Machines ###
###############################

library(e1071)

svm1 <- svm(formula = Survived ~ ., data = compTrain1)
svm1.pred <- predict(svm1, compTest1)
write.csv(svm1.pred, file = "titanic_survivalsvm1.csv")
#Your submission scored 0.77512.
#Position 3988.

svm2 <- svm(formula = Survived ~ ., data = compTrain2)
svm2.pred <- predict(svm2, compTest2)
write.csv(svm2.pred, file = "titanic_survivalsvm2.csv")
#Your submission scored 0.77512.
#Position 3990.

#####################################
### TUNED SUPPORT VECTOR MACHINES ###
#####################################

#Perform a grid search
tuneResult1 <- tune(svm, Survived ~ ., data = compTrain1, 
                   ranges = list(epsilon = seq(0, 1, 0.1), cost = 2^(2:9)))
print(tuneResult1)
plot(tuneResult1)
tunedModel1 <- tuneResult1$best.model

tsvm1.pred <- predict(tunedModel1, compTest1)
write.csv(tsvm1.pred, file = "titanic_survivaltsvm1.csv")
#Your submission scored 0.76555.

tuneResult2 <- tune(svm, Survived ~ ., data = compTrain2, 
                    ranges = list(epsilon = seq(0, 1, 0.1), cost = 2^(2:9)))
print(tuneResult2)
plot(tuneResult2)
tunedModel2 <- tuneResult2$best.model

tsvm2.pred <- predict(tunedModel2, compTest2)
write.csv(tsvm2.pred, file = "titanic_survivaltsvm2.csv")
#Your submission scored 0.78947.
#Position 2211.

