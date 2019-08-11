# Communities-and-Crime

### Summary
The crime anf communities data set contains approximately 1994 variables and more than 100 variables. The objective is to predict violent crime per 100 thousand population and explain the factors that influence the proportion of violent crimes across different states in US.
 
### Data Cleaning
1. Various states like Hawaii, Illinois, Michigan, Montana, Nebraska are missing from the dataset. 
2. Cleaned dataset includes only those states where row-count is greater than or equal to 10. This further dropped 11 more states (Alaska, Delaware, District of Columbia, Idaho, Kansas, Minnesota, Nevada, North Dakota, South Dakota, Vermont & Wyoming) from analysis.
3. Various variables with completely missing data are also omitted from the cleaned data set. 
4. Cleaned data set replaces the FIPS state numbers with actual state names using lookup files.
5. A new column called 'Validation' is added to split the data set into 75% training and 25% validation.
6. Final cleaned dataset has 1941 observation and 104 variables

### Statistical Analysis
1. Model building is started by selecting variables using Forward selection method and Max validation R square as variable selection criteria. 
2. The variables that are selected are as follows:
Racepctblack, racePctWhite, agePct12t29, numbUrban, pctUrban, MalePctDivorce, PctKids2Par, PctWorkMom, PctIlleg, PctPersDenseHous,	HousVacant, MedOwnCostPctIncNoMtg, NumStreet.
3. Statistical models are fit using the 'violentcrimeperpop' as the response variable and the variables selected in earlier step as explanatory variables
4. The interaction terms for all the numeric predictors with state are included because I wish to explain the factors that influence the proportion of violent crimes across different states in US
5. Following regression models were fit for the data -
  - Least Squares Regression
  - Generalized regression, LASSO – Least absolute shrinkage and selection operator
  - Elastic Net
  - Elastic Net with Blank alpha – we let the JMP to estimate the alpha which we use in next model. 
  - Elastic Net with alpha = 0.89
  - Beta Regression with Lasso
  - Beta Regression Lasso with Huber technique for outliers
  - Beta Regression Lasso with Cauchy technique for outliers
6. Elastic Net and LASSO regression models are the ones with lowest RASE, but the predicted value for 'violentcrimeperpop' variable goes beyond 0 and 1 range which makes them invalid for interpretation of proportion of violent crimes. 
7. Hence, Beta regression – LASSO is selected for prediction of violent crimes.

### Model Interpretation
The most important variables that impacts the response variable i.e. total number of violent crimes per 100 population are as follows:
-	percentage of people living in areas classified as urban (pctUrban), 
-	percentage of population that is Caucasian (racePctWhite), 
-	percentage of males who are divorced (MalePctDivorce), 
-	percentage of kids in family housing with two parents (PctKids2Par) 
-	number of vacant households (HousVacant)

1. There is a negative relationship between proportion of violent crimes and percentage of population that is Caucasian and percentage of kids in family housing with two parents.
2. We can also see a positive relationship between propotion of violent crimes and percentage of people living in urban areas
3. In general violent crimes increase when percentage of divorced males go up and this is very evidently exhibited in the state of South Carolina.
4. Violent crimes go up when number of vacant household increases and this relationship is very strongly seen in the states like Arkansas and Missouri.
5. In general, we see a positive relationship between the violent crimes and State of Florida, but percentage of kids born to never married parents seems to be the most important factor.
