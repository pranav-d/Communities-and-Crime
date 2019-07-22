# STAT-684 Project
#Violence Crime in US states

# Clear the workspace. I do this every time
rm(list = ls())

setwd("/Users/pranavd/Documents/tamu/STAT_684/data/")
library(gdata)
crime_data = read.xls ("communities_with_header.xlsx", sheet = 1, header = TRUE)
# crime_data   = read.csv(file='communities.csv')

str(crime_data)
# class(crime_data$state)

state_fips = read.csv(file='state_fips.csv') 
str(state_fips)

# The raw dataset has FIPS nmber for the state, we need actual state name
# Use the state_fips lookup file to extract actual state names
df<-merge(x=crime_data,y=state_fips[,c('STATE', 'STUSAB', 'STATE_NAME')],
          by.x="state", by.y = 'STATE',all.x=TRUE)
summary(df)


library(Hmisc)


#there are factor variables which have high number of missing values (shown as ? in raw data).
#we need to identify the variables with high missing vcalues and drop them from analysis.
library(dplyr)
nums <- unlist(lapply(df, is.factor))  
summary(df[,nums])

# Following variables will be dropped from the analysis because of high missing count
drop <- c('community', 'county', 'LemasGangUnitDeploy', 'LemasPctPolicOnPatr',
           'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop', 'LemasSwFTPerPop',
           'LemasSwornFT', 'LemasTotalReq', 'LemasTotReqPerPop', 'NumKindsDrugsSeiz',
           'OfficAssgnDrugUnits', 'PctPolicAsian', 'PctPolicBlack', 'PctPolicHisp',
           'PctPolicMinor', 'PctPolicWhite', 'PolicAveOTWorked', 'PolicBudgPerPop',
           'PolicCars', 'PolicOperBudg', 'PolicPerPop', 'PolicReqPerOffic', 'population',
           'RacialMatchCommPol', 'state')
df_crime_good <- df[ , !(names(df) %in% drop)]

# describe(df_crime_good)
# summary(df_crime_good)



# count by group
row_count_by_state <- data.frame(table(df_crime_good$STATE_NAME))
# Keep only those states which have row count >= 10
state_select <- row_count_by_state[row_count_by_state$Freq >= 10,'Var1']
crime_good <- df_crime_good[df_crime_good$STATE_NAME %in% state_select, ]

crime_good$STATE_NAME <- factor(crime_good$STATE_NAME)

# Check if there are any numeric columns with factor as type
nums1 <- unlist(lapply(crime_good, is.factor))  
summary(crime_good[,nums1])

# OtherPerCap is the only variable which is factor and should be converted to numeric
crime_good$OtherPerCap <- as.numeric(levels(crime_good$OtherPerCap)[crime_good$OtherPerCap])

# Crime good is the final dataset prrior to the analysis
summary(crime_good$OtherPerCap)
nrow(crime_good)
ncol(crime_good)
sapply(crime_good, function(x) sum( is.na(x) ))
names(crime_good)

names(crime_good[,!(names(crime_good) %in% c("STUSAB", "STATE_NAME", "fold","communityname"))])
# Full Model - Linear Model
full <- lm(ViolentCrimesPerPop ~ ., 
           data = crime_good[,!(names(crime_good) %in% 
            c("STUSAB", "STATE_NAME", "fold", "communityname"))])

# Intercept only for Linear Model - used for variable selection
fitstart <- lm(ViolentCrimesPerPop ~ 1, 
               data = crime_good[,!(names(crime_good) %in% 
                                           c("STUSAB", "STATE_NAME", "fold", "communityname"))])


#############################################################################
library(MASS)
# variable selection using stepAIC
# forward selection
fwd_stepAIC <- stepAIC(fitstart, scope = formula(full), direction = "forward", trace = TRUE, k = log(nrow(crime_good)))
fwd_stepAIC$anova
fwd_stepAIC$terms
summary(fwd_stepAIC)
#stepAIC forward variables selected - 13
# PctKids2Par, racePctWhite,  HousVacant, pctUrban, PctWorkMom, NumStreet, 
# MalePctDivorce,  PctIlleg, numbUrban, PctPersDenseHous, racepctblack, agePct12t29,
# MedOwnCostPctIncNoMtg 
#-------------------------------------------------------------------------------#
# backward selection
bkwd_stepAIC <- stepAIC(full, direction = "backward", trace = TRUE, k = log(nrow(crime_good)))
bkwd_stepAIC$anova
bkwd_stepAIC$terms
summary(bkwd_stepAIC)

#stepAIC Backward variables selected - 16
# racepctblack, pctUrban, pctWWage, pctWRetire, PctPopUnderPov, MalePctDivorce,
# PctKids2Par, PctWorkMom, PctIlleg, NumImmig, PctPersDenseHous, HousVacant, 
#RentLowQ,  MedRent, MedOwnCostPctIncNoMtg, NumStreet
#-------------------------------------------------------------------------------#
# Stepwise Selection
stepws_stepAIC <- stepAIC(fitstart, scope = formula(full), direction = "both", trace = TRUE, k = log(nrow(crime_good)))

stepws_stepAIC$anova
stepws_stepAIC$terms
summary(stepws_stepAIC)

#stepAIC Stepwise variables selected - 12
# PctKids2Par, HousVacant, pctUrban, PctWorkMom, NumStreet, MalePctDivorce,
# PctIlleg, numbUrban, PctPersDenseHous, racepctblack, PctPopUnderPov, MedOwnCostPctIncNoMtg
#############################################################################
# variable selection using step function
# forward selection
fwd_step <- step(fitstart, scope = formula(full), direction = "forward", trace = TRUE, k = log(nrow(crime_good)))
fwd_step$anova
fwd_step$terms
summary(fwd_step)
#step function forward selection - 13
# PctKids2Par, racePctWhite, HousVacant, pctUrban, PctWorkMom, NumStreet,
# MalePctDivorce, PctIlleg, numbUrban, PctPersDenseHous, racepctblack, agePct12t29, 
# MedOwnCostPctIncNoMtg
#-------------------------------------------------------------------------------#
# backward selection
bkwd_step <- step(full, direction = "backward", trace = TRUE, k = log(nrow(crime_good)))
bkwd_step$anova
bkwd_step$terms
summary(bkwd_step)

#step Backward variables selected - 16
# racepctblack, pctUrban, pctWWage, pctWRetire, PctPopUnderPov, MalePctDivorce,
# PctKids2Par, PctWorkMom, PctIlleg, NumImmig, PctPersDenseHous, HousVacant,
# RentLowQ, MedRent, MedOwnCostPctIncNoMtg, NumStreet
#-------------------------------------------------------------------------------#

#stepwise function
stpwise_step <- step(fitstart, scope = formula(full), direction = "both", trace = TRUE, k = log(nrow(crime_good)))
stpwise_step$anova
stpwise_step$terms
summary(stpwise_step)

#step Stepwise variables selected - 12
# PctKids2Par, HousVacant, pctUrban, PctWorkMom, NumStreet, MalePctDivorce,
# PctIlleg, numbUrban, PctPersDenseHous, racepctblack, PctPopUnderPov, MedOwnCostPctIncNoMtg
#############################################################################
# Training / Test Split
set.seed(1983)
train = sample(nrow(crime_good), ceil(nrow(crime_good)*0.75))
train_data = crime_good[train, !(names(crime_good) %in% c("STUSAB", "fold", "communityname"))]
test_data = crime_good[-train, !(names(crime_good) %in% c("STUSAB", "fold", "communityname"))]

#############################################################################


# Linear Model interation with States
# Run the model on the train data
lm_int <- lm(ViolentCrimesPerPop ~ PctKids2Par + HousVacant + pctUrban + PctWorkMom + NumStreet+ MalePctDivorce + PctIlleg + numbUrban + PctPersDenseHous + racepctblack + PctPopUnderPov + MedOwnCostPctIncNoMtg + STATE_NAME + PctKids2Par * STATE_NAME + HousVacant * STATE_NAME + pctUrban * STATE_NAME + PctWorkMom * STATE_NAME + NumStreet * STATE_NAME + MalePctDivorce * STATE_NAME + PctIlleg * STATE_NAME + numbUrban * STATE_NAME + PctPersDenseHous * STATE_NAME + racepctblack * STATE_NAME + PctPopUnderPov * STATE_NAME + MedOwnCostPctIncNoMtg * STATE_NAME, 
data = train_data)

summary(lm_int)

# The effects that are significant at alpha = 0.05
summary(lm_int)$coefficients[summary(lm_int)$coefficients[,4]<0.05,]
# Root Mean Square for linear model interation with states
# RMSE - Training
RMSE_lm <- sqrt(mean(lm_int$residuals^2))
RMSE_lm
#RMSE - Validation
# sqrt(mean((test_data$ViolentCrimesPerPop - predict(lm_int, test_data))^2))

val_matrix <- cbind(act = test_data$ViolentCrimesPerPop, pred=predict(lm_int, test_data))
val_matrix_final <- as.data.frame(val_matrix[val_matrix[,2] < 3 & val_matrix[,2] > -5, ])

#RMSE - Validation
sqrt(mean((val_matrix_final$act - val_matrix_final$pred)^2))

#############################################################################

#LASSO regression

# formula for selected variables and interaction
f <- as.formula(ViolentCrimesPerPop ~ PctKids2Par + HousVacant + pctUrban + PctWorkMom + NumStreet+ MalePctDivorce + PctIlleg + numbUrban + PctPersDenseHous + racepctblack + PctPopUnderPov + MedOwnCostPctIncNoMtg + STATE_NAME + PctKids2Par * STATE_NAME + HousVacant * STATE_NAME + pctUrban * STATE_NAME + PctWorkMom * STATE_NAME + NumStreet * STATE_NAME + MalePctDivorce * STATE_NAME + PctIlleg * STATE_NAME + numbUrban * STATE_NAME + PctPersDenseHous * STATE_NAME + racepctblack * STATE_NAME + PctPopUnderPov * STATE_NAME + MedOwnCostPctIncNoMtg * STATE_NAME)

# convert formula to matrix
# For Training Data
x <- model.matrix(f, train_data)
# For Test Data
x_test <- model.matrix(f, test_data)
# Training Data Response variable
y <- as.matrix(train_data$ViolentCrimesPerPop, ncol=1)
# Test Data Response variable
y_test <- as.matrix(test_data$ViolentCrimesPerPop, ncol=1)

library(glmnet)
# Select the range of values for the best lambda
# here we have chosen to implement the function over a grid of values ranging
# from λ = 10^10 to λ = 10^−2, essentially covering the full range of scenarios
# from the null model containing only the intercept, to the least squares fit.
grid =10^seq(10,-2, length =100)
# Fit Lasso Model on Training Data 
lasso.mod =glmnet (x, y, alpha =1, lambda = grid)
# shows the order in which variable enter the model
plot(lasso.mod)


set.seed(1)
# cross validation for best lamda
cv.out =cv.glmnet (x, y, alpha =1, type.measure = "mse", family = "gaussian")
plot(cv.out)
# Best Lambda
# bestlam =cv.out$lambda.min
# lambda.1se is the value of lambda stored in cv.out that resulted in the simplest 
# model i.e. model with fewest non zero parameters and was with in 1 standard error
# of the lamda that had smallest sum
bestlam = cv.out$lambda.1se
# prediction for testdata using best lambda
# lasso.pred=predict(lasso.mod, s=bestlam, newx=x_test)
lasso.pred=predict(cv.out, s=bestlam, newx=x_test)

# Root Mean square error for LASSO regression
mean((lasso.pred - y_test)^2)



lasso.coef=predict(cv.out, type ="coefficients", s=bestlam)
lasso.coef

# Non Zero coefficients of Lasso regression
lasso.coef[lasso.coef[,1] != '.' & lasso.coef[,1] != 0, ]

plot(lasso.mod, xvar = "dev", label = TRUE)


##########################################################################################
# Elastic Net regression

list_fits <- list()
for (i in 0:10) {
fit_name <- paste0("alpha", i/10) 
list_fits[[fit_name]] <- cv.glmnet(x, y, type.measure="mse", alpha=i/10,  family="gaussian")
}

results <- data.frame()
for (i in 0:10) {
    fit_name <- paste0("alpha", i/10)
    
    ## Use each model to predict 'y' given the Testing dataset
    predicted <- predict(list_fits[[fit_name]], s=list_fits[[fit_name]]$lambda.1se, newx=x_test)
    
    ## Calculate the Mean Squared Error...
    mse <- mean((y_test - predicted)^2)
    
    ## Store the results
    temp <- data.frame(alpha=i/10, mse=mse, fit_name=fit_name)
    results <- rbind(results, temp)
}

## View the results
results

# The least Mean Square Error (MSE) = 0.01769807 with alpha = 0.3 
# Train the elastic net model
elasticnet_alpha0.3 <- cv.glmnet(x, y, type.measure="mse", alpha=0.3, family="gaussian")

# predictions on the test data
pred_elasticnet_alpha0.3 <- predict(elasticnet_alpha0.3, s=elasticnet_alpha0.3$lambda.1se,
                                    newx=x_test)
# MSE Elastic Net Model
mean((y_test - pred_elasticnet_alpha0.3)^2)

# coefficients of final elastic net model with alpha = 0.3 and lamda = 0.03802176
elasticnet.coef=predict(elasticnet_alpha0.3, type ="coefficients", s=elasticnet_alpha0.3$lambda.1se)
elasticnet.coef

# Non Zero coefficients of Elastic net regression
elasticnet.coef[elasticnet.coef[,1] != '.' & elasticnet.coef[,1] != 0, ]

# predicted <- predict(list_fits[["alpha0.3"]], s=list_fits[["alpha0.3"]]$lambda.1se, newx=x_test)
# mse <- mean((y_test - predicted)^2)
# mse

##########################################################################################
# Beta regression
library(betareg)

# train_data_beta <- crime_good
train_data_beta <- train_data
test_data_beta <- test_data

train_data_beta$ViolentCrimesPerPop[train_data_beta$ViolentCrimesPerPop == 1] <- 0.999999

train_data_beta$ViolentCrimesPerPop[train_data_beta$ViolentCrimesPerPop == 0] <- 0.000001

pmod_betareg <- betareg(formula = ViolentCrimesPerPop ~ PctKids2Par + HousVacant + pctUrban + PctWorkMom + NumStreet+ MalePctDivorce + PctIlleg + numbUrban + PctPersDenseHous + racepctblack + PctPopUnderPov + MedOwnCostPctIncNoMtg + STATE_NAME , data = train_data_beta)
summary(pmod_betareg)



pred_betareg <- predict(pmod_betareg, test_data_beta)

# MSE for Beta Regression
mean((test_data_beta$ViolentCrimesPerPop - pred_betareg)^2)

##########################################################################################


# Root Mean square error for Beta Regression
sqrt(mean((test_data_beta$ViolentCrimesPerPop - pred_betareg)^2))

# Root Mean square error Elastic Net Model
sqrt(mean((y_test - pred_elasticnet_alpha0.3)^2))

# Root Mean square error for LASSO regression
sqrt(mean((lasso.pred - y_test)^2))

# Root Mean square for Linear Regression Model
sqrt(mean(lm_int$residuals^2))


##########################################################################################

# Model selected - Beta Regression
summary(pmod_betareg)

# Non Zero coefficients of Lasso regression - use to interprete interactions
lasso.coef[lasso.coef[,1] != '.' & lasso.coef[,1] != 0, ]

# We see a positive relationship between mean response i.e. proportion of violent crimes and percentage of people living in areas classified as urban

# we can see that proportion of violent crimes increases as number of vacant households increase in the communities

# In general, we see as percentage of males who are divorced go up, the proportion of violent crimes goes up as well. The effect is espcially significant for the state of South Carolina

# proportion of Violent crimes goes down as percentage of kids in family housing with two parents go up

# proportion of Violent crimes goes down as percentage of moms of kids under 18 in labor force go up
 
# In general, we see as percentage of kids born to never married parents goes up, the proportion of violent crimes goes up as well. The effect is espcially significant for the state of california

# In general, we as number of homeless people counted in the street increases, the proportion of violent crimes go up.

##########################################################################################




