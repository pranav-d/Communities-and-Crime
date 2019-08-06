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
1. Model b
