## Employee Retention Prediction Model for a Pharmaceutical Company
## Introduction
In this report, we detail the development of a logistic regression-based Employee Retention Prediction Model for a pharmaceutical company. This model is designed to predict the likelihood of employees leaving the company, utilizing a dataset that encapsulates 32 variables across a total workforce of 1,424 individuals.

```
tibble [1,424 x 32] (S3: tbl_df/tbl/data.frame)
 $ Age                       : num [1:1424] 27 27 31 34 42 30 34 34 43 47 ...
 $ Leaving the company       : chr [1:1424] "No" "No" "No" "No" ...
 $ BusinessTravel            : chr [1:1424] "Travel_Frequently" "Travel_Rarely" "Travel_Rarely" "Travel_Rarely" ...
 $ Department                : chr [1:1424] "Sales" "Research & Development" "Research & Development" "Sales" ...
 $ Distance From Home        : num [1:1424] 8 14 11 14 1 5 10 13 9 27 ...
 $ Education                 : num [1:1424] 1 3 4 3 1 3 3 4 5 2 ...
 $ Education Field           : chr [1:1424] "Marketing" "Life Sciences" "Life Sciences" "Technical Degree" ...
 $ Employee Count            : num [1:1424] 1 1 1 1 1 1 1 1 1 1 ...
 $ Employee Number           : num [1:1424] 1 2 3 4 5 6 7 8 9 10 ...
 $ Environment Satisfaction  : num [1:1424] 3 1 4 3 2 2 4 4 4 2 ...
 $ Gender                    : chr [1:1424] "Male" "Male" "Male" "Female" ...
 $ Job Involvement           : num [1:1424] 3 3 3 3 3 3 3 3 3 4 ...
 $ Job Level                 : num [1:1424] 2 1 1 1 2 3 2 3 2 2 ...
 $ Job Role                  : chr [1:1424] "Sales Executive" "Research Scientist" "Research Scientist" "Sales Representative" ...
 $ Job Satisfaction          : num [1:1424] 4 1 4 3 3 4 3 3 3 3 ...
 $ Marital Status            : chr [1:1424] "Married" "Married" "Married" "Divorced" ...
 $ Monthly Income            : num [1:1424] 4342 2235 2356 2579 4907 ...
 $ Number of Companies Worked: num [1:1424] 0 1 3 1 1 2 1 1 3 4 ...
 $ Over 18                   : chr [1:1424] "Y" "Y" "Y" "Y" ...
 $ OverTime                  : chr [1:1424] "No" "Yes" "Yes" "Yes" ...
 $ Percent Salary Increase   : num [1:1424] 19 14 19 18 25 12 14 18 13 12 ...
 $ Performance Rating        : num [1:1424] 3 3 3 3 4 3 3 3 3 3 ...
 $ Relationship Satisfaction : num [1:1424] 2 4 2 4 3 3 3 3 2 4 ...
 $ Standard Hours            : num [1:1424] 40 40 40 40 40 40 40 40 40 40 ...
 $ Stock Option Level        : num [1:1424] 1 2 1 2 0 1 1 1 1 0 ...
 $ Total Working Years       : num [1:1424] 5 9 8 8 20 12 15 9 10 8 ...
 $ Training Times Last Year  : num [1:1424] 3 3 2 3 3 2 3 2 3 2 ...
 $ Work Life Balance         : num [1:1424] 3 2 3 3 3 3 3 2 3 3 ...
 $ Years At Company          : num [1:1424] 4 9 6 8 20 10 15 8 8 5 ...
 $ Years In Current Role     : num [1:1424] 2 7 4 2 16 9 14 7 7 4 ...
 $ Years Since Last Promotion: num [1:1424] 1 6 0 0 11 7 0 1 4 1 ...
 $ Years With Current Manager: num [1:1424] 1 8 2 6 6 4 7 1 7 3 ... 
```

## Data Preprocessing
Initial model construction involved running logistic regression against the full set of variables, with the exclusion of less relevant factors such as Educational Field, Employee Count, Job Role, individuals over 18, and Standard Hours. Prior to model construction, the dataset was split into training and testing sets in a 70/30 ratio, ensuring robust model evaluation. This split allowed the model to be trained on 70% of the data, encompassing a wide array of employee profiles, while the remaining 30% was used to assess the model's predictive performance on unseen data. The target variable for prediction, "Leaving the Company," originally comprised categorical responses (Yes/No), which were subsequently converted to a binary format (1/0) for computational efficacy — '1' indicating an employee's departure and '0' denoting retention. Concomitantly, other categorical variables were numerically transformed to facilitate the modeling process.

```
#Convert "Leaving The Company" to binary
data$`Leaving the company` <- ifelse(data$`Leaving the company` == "Yes", 1, 0)
```
```
# Set the seed for reproducibility
set.seed(123)
# Split the data into training and testing sets
split <- sample.split(data$`Leaving the company`, SplitRatio = 0.7) # Splitting 70% for training
train <- subset(data, split == TRUE) # Training data
test <- subset(data, split == FALSE) # Testing data
```

## Model Refinement
The initial model was subject to a variable importance analysis using the varImp function. This led to the distillation of the model down to 16 critical variables based on selecting the higher overall values >= 2, thereby enhancing both efficiency and effectiveness in predicting employee turnover.
```
Coefficients:
                             Estimate Std. Error z value Pr(>|z|)    
(Intercept)                   7.25411    1.00772   7.199 6.08e-13 ***
OverTime                      2.09103    0.22827   9.160  < 2e-16 ***
Age                          -0.03698    0.01571  -2.354 0.018579 *  
Department                   -0.29321    0.11972  -2.449 0.014321 *  
`Job Level`                  -0.48670    0.19079  -2.551 0.010740 *  
`Marital Status`             -0.77524    0.15218  -5.094 3.50e-07 ***
`Work Life Balance`          -0.41839    0.14277  -2.931 0.003384 ** 
`Years In Current Role`      -0.13558    0.04424  -3.065 0.002180 ** 
`Training Times Last Year`   -0.16179    0.08325  -1.943 0.051976 .  
`Environment Satisfaction`   -0.32354    0.09694  -3.338 0.000845 ***
`Number of Companies Worked`  0.16755    0.04310   3.888 0.000101 ***
`Relationship Satisfaction`  -0.21432    0.09546  -2.245 0.024767 *  
`Distance From Home`          0.04940    0.01252   3.944 8.00e-05 ***
`Total Working Years`        -0.07811    0.03190  -2.449 0.014330 *  
`Job Involvement`            -0.58660    0.14224  -4.124 3.73e-05 ***
`Job Satisfaction`           -0.43151    0.09511  -4.537 5.70e-06 ***
`Years Since Last Promotion`  0.20997    0.04498   4.668 3.04e-06 ***
```

## Model Predictions
Evaluative metrics of the refined model revealed that 81.3% of employees were accurately predicted to stay with the company, while approximately 2.3% were mistakenly thought to be on the verge of leaving. Conversely, 9.8% of employees were falsely predicted to remain, and 6.5% were correctly identified as potential leavers.
```
#Confusion Matrix
    FALSE  TRUE
  0 0.813 0.023
  1 0.098 0.065
```

## Quantitative Analysis
In absolute terms, of those predicted to stay or depart, 348 employees were correctly classified as stayers, albeit with 10 misclassifications. In contrast, 42 employees were erroneously forecasted to stay, yet the model successfully pinpointed 28 employees who were actually leaving. Overall, the model boasts an impressive accuracy rate of 87.85%, with a relatively modest error rate of 12.15%, underscoring the model's reliability.
```
     0   1
  0 348  10
  1  42  28
> #Accuracy rate
> mean(ifelse(glm.probs > 0.5, "1", "0") == test$`Leaving the company`)
   0.8785047
> #Error rate
> mean(ifelse(glm.probs > 0.5, "1", "0") != test$`Leaving the company`)
   0.1214953
```

## Visualization: ROC and AUC
The model's diagnostic prowess is encapsulated in a ROC curve visualization, complemented by an AUC metric. The ROC curve illustrates the model's competence in distinguishing between employees likely to leave or stay. Although our model doesn't exhibit the ideal 'right angle' ascent, the rapid initial climb and subsequent sweep to the right suggest commendable predictive capability, corroborated by an AUC of approximately 0.79.

<img src="https://github.com/andinetM/Employee-Retention/blob/main/leogit_model_ROC_Rplot.png" align="center" height="350" width="500"/>

```
# Logistic regression AUC
> prediction(glm.probs, test$`Leaving the company`) %>% +   performance(measure = "auc") %>% +   .@y.values
= 0.7943735
```

## Conclusion
The Employee Retention Prediction Model exhibits high accuracy, which could serve as a crucial instrument for informed decision-making in employee retention strategies. Despite inherent imperfections — as is the case with any predictive model — the low error rate instills confidence in its prognostications. This model is instrumental in identifying employees at risk of departure, enabling proactive retention interventions.

