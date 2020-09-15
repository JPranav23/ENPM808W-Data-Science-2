# Q1: 
# Reading the dataset

qb.train <- read.csv("~/Desktop/Data Science 808W/Assignment 3/all/qb.train.csv")
View(qb.train)

qb.test <- read.csv("~/Desktop/Data Science 808W/Assignment 3/all/qb.test.csv")
View(qb.test)

qb.guess.sample <- read.csv("~/Desktop/Data Science 808W/Assignment 3/all/qb.guess-sample.csv")
View(qb.guess.sample)

# Q2: 
# SVM using Functions
# Functions 

library(e1071)
library(cwhmisc)

# Creating functions

model.accuracy <- function(data, name, model, test) {
  svm.pred <- predict(model, test)
  svm.table <- table(pred = svm.pred, true = test$corr)
  data <- rbind(data, data.frame(model=c(name), score=c(classAgreement(svm.table)$diag)))
  return(data)
}

paren_match <- function(page, text) {
  start <- cpos(page, "(")
  end <- cpos(page, ")")
  if (!is.na(start) && !is.na(end)) {
    search <- substring(page, start + 1, end - 1)
    return(grepl(tolower(search), tolower(text), fixed=TRUE))
  } else {
    return(FALSE)
  }
}

# Creating some additional parameters

a <- qb.train
b <- qb.test
a$obs_len <- apply(a, 1, function(x) {nchar(x['text'])})
b$obs_len <- apply(b, 1, function(x) {nchar(x['text'])})

a$scale_len <- scale(a$obs_len)
b$scale_len <- scale(b$obs_len)

a$scale_score <- scale(a$body_score)
b$scale_score <- scale(b$body_score)

a$paren_match <- apply(a, 1, function(x) {paren_match(x['page'], x['text'])})
b$paren_match <- apply(b, 1, function(x) {paren_match(x['page'], x['text'])})

a$log_links <- scale(log(as.numeric(a$inlinks) + 1))
b$log_links <- scale(log(as.numeric(b$inlinks) + 1))

index <- 1:nrow(a)
testindex <- sample(index, trunc(length(index)/5))
testset <- a[testindex,]
trainset <- a[-testindex,]

# Get the most frequent baseline

min.error <- sum(a$corr == "False") / (sum(a$corr == "False") + sum(a$corr == "True"))
min.error
models <- NULL
models <- data.frame(model=c("basic"), score=c(min.error))

# Using the functions to make models

models <- model.accuracy(models, "body_score.svm", svm(corr ~ body_score, data=trainset), testset)
models <- model.accuracy(models, "scale_score.svm", svm(corr ~ scale_score, data=trainset), testset)
models <- model.accuracy(models, "obs_len.svm", svm(corr ~ obs_len, data=trainset), testset)
models <- model.accuracy(models, "score+len.svm", svm(corr ~ obs_len + body_score, data=trainset), testset)
models <- model.accuracy(models, "paren+len.svm", svm(corr ~ obs_len + paren_match, data=trainset), testset)
models <- model.accuracy(models, "paren_match.svm", svm(corr ~ paren_match, data=trainset), testset)
models <- model.accuracy(models, "score+paren_match", svm(corr ~ scale_score + paren_match, data=trainset), testset)
models <- model.accuracy(models, "score+len+paren_match.svm", svm(corr ~ scale_len + scale_score + paren_match, data=trainset), testset)
models <- model.accuracy(models, "links.svm", svm(corr ~ inlinks, data=trainset), testset)
models <- model.accuracy(models, "loglinks.svm", svm(corr ~ log_links, data=trainset), testset)
models <- model.accuracy(models, "score+len+links+paren_match.svm", svm(corr ~ scale_len + scale_score + log_links + paren_match, data=trainset), testset)
models <- model.accuracy(models, "score+links+paren_match.svm", svm(corr ~ scale_len + scale_score + paren_match, data=trainset), testset)
View(models)

# SVM model using different kernels

a <- qb.train
b <- qb.test
a$obs_len <- apply(a, 1, function(x) {nchar(x['text'])})
b$obs_len <- apply(b, 1, function(x) {nchar(x['text'])})

a$scale_len <- scale(a$obs_len)
b$scale_len <- scale(b$obs_len)

a$scale_score <- scale(a$body_score)
b$scale_score <- scale(b$body_score)

a$paren_match <- apply(a, 1, function(x) {paren_match(x['page'], x['text'])})
b$paren_match <- apply(b, 1, function(x) {paren_match(x['page'], x['text'])})

index <- 1:nrow(a)
testindex <- sample(index, trunc(length(index)/5))
testset <- a[testindex,]
trainset <- a[-testindex,]

# Linear Kernel

svm11 <- svm(corr ~ body_score, kernel = "linear", data=a)
k11 <- predict(svm11, b)
View(k11)
l11 <- data.frame(b$row, k11)
colnames(l11) <- c("row", "corr")
View(l11)
write.csv(l11, "modelpred1.csv")

svm12 <- svm(corr ~ body_score + obs_len, kernel = "linear", data=a)
k12 <- predict(svm12, b)
View(k12)
l12 <- data.frame(b$row, k12)
colnames(l12) <- c("row", "corr")
View(l12)
write.csv(l12, "modelpred2.csv")

# Sigmoidal Kernel

svm21 <- svm(corr ~ body_score, kernel = "sigmoid", data=a)
k21 <- predict(svm21, b)
View(k21)
l21 <- data.frame(b$row, k21)
colnames(l21) <- c("row", "corr")
View(l21)
write.csv(l21, "modelpred3.csv")

svm22 <- svm(corr ~ body_score + obs_len, kernel = "sigmoid", data=a)
k22 <- predict(svm22, b)
View(k22)
l22 <- data.frame(b$row, k22)
colnames(l22) <- c("row", "corr")
View(l22)
write.csv(l22, "modelpred4.csv")

# Polynomial Kernel

svm31 <- svm(corr ~ body_score, kernel = "polynomial", data=a)
k31 <- predict(svm31, b)
View(k31)
l31 <- data.frame(b$row, k31)
colnames(l31) <- c("row", "corr")
View(l31)
write.csv(l31, "modelpred5.csv")

svm32 <- svm(corr ~ body_score, kernel = "polynomial", data=a)
k32 <- predict(svm32, b)
View(k32)
l32 <- data.frame(b$row, k32)
colnames(l32) <- c("row", "corr")
View(l32)
write.csv(l32, "modelpred6.csv")

# SVM using default Radial Kernel

svm4 <- svm(corr ~ scale_len + scale_score + paren_match, data=a)
k4 <- predict(svm4, b)
View(k4)
l4 <- data.frame(b$row, k4)
colnames(l4) <- c("row", "corr")
View(l4)
write.csv(l4, "modelpred7.csv")

svm5 <- svm(corr ~ inlinks + scale_score + paren_match, data=a)
k5 <- predict(svm5, b)
View(k5)
l5 <- data.frame(b$row, k5)
colnames(l5) <- c("row", "corr")
View(l5)
write.csv(l5, "modelpred8.csv")

svm6 <- svm(corr ~ scale_score + paren_match, data=a)
k6 <- predict(svm6, b)
View(k6)
l6 <- data.frame(b$row, k6)
colnames(l6) <- c("row", "corr")
View(l6)
write.csv(l5, "modelpred9.csv")

# Logistic Regression

a <- qb.train
b <- qb.test

index <- 1:nrow(a)
testindex <- sample(index, trunc(length(index)/5))
testset <- a[testindex,]
trainset <- a[-testindex,]

# Creating a jitter plot

plot(a$body_score, jitter(a$corr, 0.15), main = "Body score vs Correlation plot", col = "red", las = 1, pch = 19, xlab = "Body Score", ylab = "Correlation")

# Making logistic regression models

lr_model1 <- glm(corr ~ body_score, data = trainset, family = "binomial")
summary(lr_model1)
lr_model2 <- glm(corr ~ answer_type, data = trainset, family = "binomial")
summary(lr_model2)

# Predicting using the logistic regression models made above

lr_model1_pr <- predict(lr_model1, testset, type = "response")
lr_model1_pr <- ifelse(lr_model1_pr > 0.5, "True", "False")
View(lr_model1_pr)
lr_model2_pr <- predict(lr_model2, testset, type = "response")
lr_model2_pr <- ifelse(lr_model2_pr > 0.5, "True", "False")
View(lr_model2_pr)

# Creating the accuracy table

tb1 <- table(testset$corr, lr_model1_pr)
accuracy.glm1 <- sum(diag(tb1))/sum(tb1)
accuracy.glm1

tb2 <- table(testset$corr, lr_model2_pr)
accuracy.glm2 <- sum(diag(tb2))/sum(tb2)
accuracy.glm2

# Adding the accuracy of logistic regression model in final table

glm.acc <- data.frame(model = "Logistic Regreesion", score = accuracy.glm1)
models <- rbind(models, glm.acc)
models

# Correlation and Correlation Plots

a$corr <- ifelse(a$corr == "True", 1, 0)
cor(a[, c(-1, -3, -4, -5, -6, -7, -8, -10)])
library(corrplot)
corrplot(cor(a[, c(-1, -3, -4, -5, -6, -7, -8, -10)]))

# Decision tree
# Tree Library

a <- qb.train
b <- qb.test
a$corr <- ifelse(a$corr == "True", 1, 0)
levels(b$body_score) <- union(levels(a$body_score), levels(b$body_score))
index <- 1:nrow(a)
testindex <- sample(index, trunc(length(index)/5))
testset <- a[testindex,]
trainset <- a[-testindex,]

# Using tree library

library(tree)
dt.model <- tree(corr ~ body_score, data = trainset)
summary(dt.model)
plot(dt.model, main = "Decision Tree Model")
text(dt.model)
text(dt.model, pretty = 5, cex = 1)
dt.model.pred <- predict(dt.model, newdata = testset)
hist(dt.model.pred, main = "Decision Tree Prediction Plot", xlab ="Prediction", ylab = "Index", xlim = c(0.2, 1), las = 1)
dt.model.pred <- ifelse(dt.model.pred > 0.5, "True", "False")
View(dt.model.pred)
head(dt.model.pred, n = 5)

dt.model.a <- tree(corr ~ body_score, data = a)
dt.model.pred.a <- predict(dt.model, newdata = b)
h1 <- data.frame(b$row, dt.model.pred.a)
colnames(h1) <- c("row", "corr")
h1$corr <- ifelse(h1$corr > 0.5, "True", "False")
View(h1)
write.csv(h1, "modelpred10.csv")

# Party Library

a <- qb.train
b <- qb.test
library(party)
party.tree1 <- ctree(corr ~ body_score, data = trainset, controls = ctree_control(mincriterion = 0.95, minsplit = 1000))
party.tree1
plot(party.tree1, main = "Tree Diagram")

party.tree2 <- ctree(corr ~ body_score + inlinks, data = trainset, controls = ctree_control(mincriterion = 0.95, minsplit = 2500))
party.tree2
plot(party.tree2, main = "Tree Diagram")

party.tree.pred11 <- predict(party.tree1, testset, type = "response")
View(party.tree.pred11)
party.tree.pred12 <- predict(party.tree2, testset, type = "response")
View(party.tree.pred12)

ty1 <- table(party.tree.pred11, testset$corr)
ty2 <- table(party.tree.pred12, testset$corr)

acc.rpart1 <- sum(diag(ty1))/sum(ty1)
acc.rpart1

acc.rpart2 <- sum(diag(ty2))/sum(ty2)
acc.rpart2

# Decision Tree using rpart library

a <- qb.train
b <- qb.test
a$corr <- ifelse(a$corr == "True", 1, 0)

index <- 1:nrow(a)
testindex <- sample(index, trunc(length(index)/5))
testset <- a[testindex,]
trainset <- a[-testindex,]

# Calling the libraries

library(rpart)
library(rpart.plot)

# Rpart decision tree models

rpart.tree1 <- rpart(corr ~ body_score, data = trainset)
rpart.tree1
rpart.plot(rpart.tree1)

rpart.tree2 <- rpart(corr ~ body_score + inlinks, data = trainset)
View(rpart.tree2)
rpart.plot(rpart.tree2)

# Prediction

rpart.tree.pred1 <- predict(rpart.tree1, testset)
rpart.tree.pred1 <- ifelse(rpart.tree.pred1 > 0.5, "True", "False")
View(rpart.tree.pred1)
rpart.tree.pred2 <- predict(rpart.tree2, testset)
rpart.tree.pred2 <- ifelse(rpart.tree.pred2 > 0.5, "True", "False")
View(rpart.tree.pred2)

# Creating the accuracy table

tab <- table(rpart.tree.pred1, testset$corr)
print(tab)
accuracy.dt <- sum(diag(tab))/sum(tab)
accuracy.dt
dt.acc <- data.frame(model = "Decision_tree", score = accuracy.dt)
dt.acc

# Accuracy Table

final.acc <- rbind(models, dt.acc)
final.acc

# Q3: 

a <- qb.train
b <- qb.test
levels(a$answer_type)
barchart(a$answer_type, main = "Distribution of answer category in training data")

# Making subset for work and people answer category

c <- a[a$answer_type == "work", ]
View(c)
d <- a[a$answer_type == "people", ]
View(d)
levels(b$answer_type)
barchart(b$answer_type, main = "Distribution of answer category in testing data")
e <- b[b$answer_type == "work", ]
View(e)
f <- b[b$answer_type == "people", ]
View(f)

index.work <- 1:nrow(c)
testindex.work <- sample(index.work, trunc(length(index.work)/5))
testset.work <- c[testindex.work,]
trainset.work <- c[-testindex.work,]

index.peop <- 1:nrow(d)
testindex.peop <- sample(index.peop, trunc(length(index.peop)/5))
testset.peop <- d[testindex.peop,]
trainset.peop <- d[-testindex.peop,]

trainset.work$obs_len <- apply(trainset.work, 1, function(x) {nchar(x['text'])})
trainset.peop$obs_len <- apply(trainset.peop, 1, function(x) {nchar(x['text'])})
testset.work$obs_len <- apply(testset.work, 1, function(x) {nchar(x['text'])})
testset.peop$obs_len <- apply(testset.peop, 1, function(x) {nchar(x['text'])})

models.work <- svm(corr ~ obs_len + body_score, data=trainset.work)
models.peop <- svm(corr ~ obs_len + body_score, data=trainset.peop)
w1 <- predict(models.work, testset.work)
p1 <- predict(models.peop, testset.peop)
t1 <- table(w1, testset.work$corr)
a1 <- sum(diag(t1))/sum(t1)
t2 <- table(p1, testset.peop$corr)
a2 <- sum(diag(t2))/sum(t2)
a1
a2

# Showing how accuracy is increased using additional feature

plot(a1, ylim = c(0.76, 0.81), main = "Accuracy Comparision Plot", pch = 1, col = 2, ylab = "Accuracy", las = 1)
points(a2, pch = 1, col = 3)
points(models$score[models$model == "score+len.svm"], pch = 1, col = 4)
legend(x = "topright",legend = c("Work_SVM", "People_SVM", "General_SVM"), pch = c(1, 1, 1), col = c(2, 3, 4))

# As we see that general svm model has greatest accuracy followed by work svm model and svm model for people.
# As we see a1 > a2.
# Also we can see that the accuracy for models.work is somewhat greater than mode.people.
# On the other hand, the accuracy for models.people is less than svm models for complete training set.
# The above results also imply that number of observations in an answer category increases, accuracy increases.

# Q4:

# Creating an object with row_ID, predicted corr and true corr

error_analysis <- data.frame(testset.work$row, w1, testset.work$corr)
View(error_analysis)

# Creating an object with misclassifications

errors <- error_analysis[w1 != testset.work$corr, ]
View(errors)

# Creating an object v2 which has body score and obs length for misclassified rows

v1 <- testset.work[, c("row","obs_len", "body_score")]
View(v1)

# Summarizing columns of v1

summary(v2$obs_len)
summary(v2$body_score)
