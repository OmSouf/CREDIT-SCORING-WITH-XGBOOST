library(tidyverse)
library(magrittr)
library(caret)
library(xgboost)
library(knitr)
set.seed(0)
library(mlr)
library(data.table)

#############################################################################
bureau_balance <- read_csv("C:/Users/Soufiane/Documents/input/bureau_balance.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

avg_bureau_balance <- bureau_balance %>% 
  group_by(SK_ID_BUREAU) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(buro_count = bureau_balance %>%  
           group_by(SK_ID_BUREAU) %>% 
           count() %$% n)

#############################################################################
bureau <- read_csv("C:/Users/soufiane/Documents/input/bureau.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

bureau_final <- left_join(bureau, avg_bureau_balance, by = "SK_ID_BUREAU")

avg_bureau_final <- bureau_final %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(buro_count = bureau_final %>%  
           group_by(SK_ID_CURR) %>% 
           count() %$% n)

rm(avg_bureau_balance,bureau,bureau_balance,bureau_final)
gc()

#############################################################################
cred_card_bal <-  read_csv("C:/Users/soufiane/Documents/input/credit_card_balance.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

avg_cred_card_bal <- cred_card_bal %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(funs(mean(., na.rm = TRUE))) %>%
  mutate(card_count = cred_card_bal %>%
           group_by(SK_ID_CURR) %>%
           count() %$% n)

rm(cred_card_bal)
gc()

#############################################################################
pos_cash_bal <- read_csv("C:/Users/soufiane/Documents/input/POS_CASH_balance.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

avg_pos_cash_bal <- pos_cash_bal %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(funs(mean(., na.rm = TRUE))) %>%
  mutate(pos_count = pos_cash_bal %>%
           group_by(SK_ID_PREV, SK_ID_CURR) %>%
           group_by(SK_ID_CURR) %>%
           count() %$% n)

rm(pos_cash_bal)
gc()

#############################################################################
prev <- read_csv("C:/Users/soufiane/Documents/input/previous_application.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

avg_prev <- prev %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(funs(mean(., na.rm = TRUE))) %>%
  mutate(nb_app = prev %>%
           group_by(SK_ID_CURR) %>%
           count() %$% n)

rm(prev)
gc()

#############################################################################
installments_payments <- read_csv("C:/Users/soufiane/Documents/input/installments_payments.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

avg_installments_payments <- installments_payments %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(funs(mean(., na.rm = TRUE))) %>%
  mutate(pos_count = installments_payments %>%
           group_by(SK_ID_PREV, SK_ID_CURR) %>%
           group_by(SK_ID_CURR) %>%
           count() %$% n)

rm(installments_payments)
gc()

#############################################################################
tr <- read_csv("C:/Users/soufiane/Documents/input/application_train.csv")%>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

te <- read_csv("C:/Users/soufiane/Documents/input/application_test.csv")%>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

#############################################################################

tri <- 1:nrow(tr)
y <- tr$TARGET

tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  left_join(avg_bureau_final, by = "SK_ID_CURR") %>% 
  left_join(avg_cred_card_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_pos_cash_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_prev, by = "SK_ID_CURR") %>% 
  left_join(avg_installments_payments, by = "SK_ID_CURR") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer()))
######################################################
rm(avg_bureau_final,avg_cred_card_bal,avg_pos_cash_bal,avg_prev,avg_installments_payments,tr,te)
gc()
#######################################################

set.seed(123)
dtest <- tr_te[-tri, ]
tr_te <- tr_te[tri, ]
dtest[is.na(dtest)] <- -999
tr_te[is.na(tr_te)] <- -999
datatrain=tr_te
tr_te<-tr_te[,-207]
tr_te=tr_te[1:30000,]
y=y[1:30000]
tr_te=as.matrix(tr_te)

tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = tr_te[tri, ], label = y[tri], missing = -999)
dval <- xgb.DMatrix(data = tr_te[-tri, ], label = y[-tri], missing = -999)
cols <- colnames(tr_te)

###########################################################
#default parameters 
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1
)

#determine number of rounds
xgbcv <- xgb.cv(params = params
                ,data = dtrain
                ,nrounds = 11
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print_every_n = 1
                ,early_stop_round = 20
                ,maximize = F
)
###########################################################
data.to.train<-as.data.frame(tr_te)
data.to.train$target<-y
train<-data.to.train[tri,]
test<-data.to.train[-tri,]

###########################################################



#create tasks
traintask <- makeClassifTask(data = train,target = "target")
testtask <- makeClassifTask(data = test,target = "target")


#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals <- list(
  objective="binary:logistic",
  eval_metric="error",
  nrounds=100L,
  eta=0.1
)

#set parameter space
params <- makeParamSet(
  makeDiscreteParam("booster",values = "gbtree"),
  makeIntegerParam("max_depth",lower = 1L,upper = 10L),
  makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
  makeNumericParam("subsample",lower = 0.5,upper = 1),
  makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
  makeNumericParam("gamma",lower = 0.5,upper = 1),
  makeNumericParam("eta",lower = 0.000001,upper = 0.3),
  makeNumericParam("max_delta_step",lower = 1L,upper = 10L),
  makeIntegerParam("nthread",lower = 1L,upper = 10L),
  makeNumericParam("alpha",lower = 0.000001,upper = 1),
  makeNumericParam("lambda",lower = 0.000001,upper = 1)
)

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 2L)

#set parallel backend
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())



#parameter tuning
mytune <- tuneParams(learner = lrn
                     ,task = traintask
                     ,resampling = rdesc
                     ,measures = auc
                     ,par.set = params
                     ,control = ctrl
                     ,show.info = T)

mytune$y #0.873069

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)

#predict model
xgpred <- predict(xgmodel,testtask)

confusionMatrix(xgpred$data$response,xgpred$data$truth)
#Accuracy : 0.8747

#stop parallelizationmytune
parallelStop()
#################################################
####################################################
# training
cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 10,
          eta = 0.05,
          max_depth = 8,
          max_delta_step = 6,
          min_child_weight = 16,
          gamma = 0,
          subsample = 0.75,
          colsample_bytree = 0.75,
          alpha = 0.025,
          lambda = 0.025,
          nrounds = 3000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 200)

