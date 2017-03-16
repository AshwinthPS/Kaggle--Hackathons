library(caret)
library(xgboost)
library(data.table)
library(psych)
library(randomForest)

main_data<-read.csv("C:\\Users\\HP\\Downloads\\data.csv (4)\\data.csv")
data<-main_data[!is.na(main_data$shot_made_flag),]
test_data<-main_data[is.na(main_data$shot_made_flag),]


colnames(data)

modeldata<-data[c(1,2,6,7,9,10:19,22:25)]
test_data<-test_data[c(1,2,6,7,9,10:19,22:25)]

colnames(modeldata)
var.test(modeldata$shot_made_flag,modeldata$minutes_remaining)
var.test(modeldata$shot_made_flag,modeldata$shot_distance)

a<-aov(shot_made_flag~minutes_remaining*shot_distance,data=modeldata)
a<-aov(shot_made_flag~minutes_remaining*seconds_remaining,data=modeldata)
a1<-aov(shot_made_flag~minutes_remaining,data=modeldata)
a2<-aov(shot_made_flag~opponent,data=modeldata)
summary(a)
summary(a1)
summary(a2)
##############################   Importance using randomForest-varImport    ##########################


colnames(modeldata)

rf_data<-model.matrix(~action_type+shot_made_flag,data=modeldata)
rf_data1<-model.matrix(~combined_shot_type+shot_made_flag,data=modeldata)

rf_data<-data.frame(rf_data)
rf_data1<-data.frame(rf_data1)

model_rf<-randomForest(shot_made_flag ~ ., data = rf_data)
importance(model_rf)

model_rf1<-randomForest(shot_made_flag ~ ., data = rf_data1)
importance(model_rf1)
varImpPlot(model_rf1)
model_rf1$importance
plot(model_rf1)

######################## Total Time Var #############################

modeldata$total_time<-(modeldata$minutes_remaining*60)+modeldata$seconds_remaining
modeldata<-modeldata[-c(3,4)]

test_data$total_time<-(test_data$minutes_remaining*60)+test_data$seconds_remaining
test_data<-test_data[-c(3,4)]

colnames(modeldata)
colnames(test_data)
nrow(modeldata)
nrow(test_data)

########################  Shot Distance. ##############################

sh<-ggplot(data,aes(shot_distance))+
  geom_bar(aes(fill=factor(shot_made_flag)))
sh

modeldata[modeldata$shot_distance>45,"shot_distance"]<-45

test_data[test_data$shot_distance>45,"shot_distance"]<-45

################################# MODELLING ########################
?glm
mm1=glm(shot_made_flag~.,family=binomial,data=train_data)
mm1
summary(mm1)

train_index<-createDataPartition(1:nrow(modeldata),p=0.3,list=F)

train_data<-modeldata[train_index,]

train_data<-modeldata

train_data<-as.data.frame(lapply(train_data,as.numeric))
str(train_data)

trainM<-data.matrix(train_data, rownames.force = NA);
colnames(trainM)

#trainM<-trainM[,-c(1,2)]

ff<-which(colnames(trainM)=="shot_made_flag")

train_D <- xgb.DMatrix(data=trainM[,-ff], label=trainM[,ff], missing = NaN);

watchlist <- list(trainM=train_D);

set.seed(1984);

param <- list(  objective= "binary:logistic",booster= "gbtree",
                eval_metric= "logloss",eta= 0.01,max_depth= 4,
                subsample= 0.8,colsample_bytree= 0.8)


cv.mod <- xgb.cv(  params= param, data= train_D, nrounds= 1500,verbose= 1,
                   watchlist= watchlist,maximize= FALSE,nfold= 6,early.stop.round= 10,
                   print.every.n= 1);


bestRound <- which.min( as.matrix(cv.mod)[,3] );


clf <- xgb.train(   params= param,data= train_D,nrounds= bestRound, 
                    verbose= 1,watchlist= watchlist,maximize= FALSE)

colnames(trainM)

featureList <- names(modeldata[,-9])

featureVector <- c() 
for (i in 1:length(featureList)) { 
  featureVector[i] <- paste(i-1, featureList[i], "q", sep="\t") 
}

write.table(featureVector, "fmap.txt", row.names=FALSE, quote = FALSE, col.names = FALSE)
xgb.dump(model = clf, fname = 'xgb.dump', fmap = "fmap.txt", with.stats = TRUE)


getwd()

mod<-xgb.dump(clf,with.stats = T)

mod

head(mod)

names <- dimnames(train_D)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = clf)

# importance plot.
xgb.plot.importance(importance_matrix[1:10,])

colnames(trainM)
colnames(test_data)

ff<-which(colnames(test_data)=="shot_made_flag")

test_data<-test_data[,-ff]

test_data<-as.data.frame(lapply(test_data,as.numeric))

testM <-data.matrix(test_data, rownames.force = NA);

preds <- predict(clf, testM);

shotid<-main_data[is.na(main_data$shot_made_flag),"shot_id"]

submission <- data.frame(shot_id=shotid, shot_made_flag=preds);

write.csv(submission, "XGBoostSubmission.csv", row.names = F);
