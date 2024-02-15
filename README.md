## Employee Retention Prediction Model for a Pharmaceutical Company
##Introduction
In this report, we detail the development of a logistic regression-based Employee Retention Prediction Model for a pharmaceutical company. This model is designed to predict the likelihood of employees leaving the company, utilizing a dataset that encapsulates 32 variables across a total workforce of 1,424 individuals.


## Data Preprocessing
Initial model construction involved running logistic regression against the full set of variables, with the exclusion of less relevant factors such as Educational Field, Employee Count, Job Role, individuals over 18, and Standard Hours. The target variable for prediction, "Leaving the Company," originally comprised categorical responses (Yes/No), which were subsequently converted to a binary format (1/0) for computational efficacy — '1' indicating an employee's departure and '0' denoting retention. Concomitantly, other categorical variables were numerically transformed to facilitate the modeling process.

## Model Refinement
The initial model was subject to a variable importance analysis using the varImp function. This led to the distillation of the model down to 16 critical variables, thereby enhancing both efficiency and effectiveness in predicting employee turnover.

## Model Predictions
Evaluative metrics of the refined model revealed that 81.3% of employees were accurately predicted to stay with the company, while approximately 2.3% were mistakenly thought to be on the verge of leaving. Conversely, 9.8% of employees were falsely predicted to remain, and 6.5% were correctly identified as potential leavers.

## Quantitative Analysis
In absolute terms, of those predicted to stay or depart, 348 employees were correctly classified as stayers, albeit with 10 misclassifications. In contrast, 42 employees were erroneously forecasted to stay, yet the model successfully pinpointed 28 employees who were actually leaving. Overall, the model boasts an impressive accuracy rate of 87.85%, with a relatively modest error rate of 12.15%, underscoring the model's reliability.

## Visualization: ROC and AUC
The model's diagnostic prowess is encapsulated in a ROC curve visualization, complemented by an AUC metric. The ROC curve illustrates the model's competence in distinguishing between employees likely to leave or stay. Although our model doesn't exhibit the ideal 'right angle' ascent, the rapid initial climb and subsequent sweep to the right suggest commendable predictive capability, corroborated by an AUC of approximately 0.79.

## Conclusion
The Employee Retention Prediction Model exhibits high accuracy, which could serve as a crucial instrument for informed decision-making in employee retention strategies. Despite inherent imperfections — as is the case with any predictive model — the low error rate instills confidence in its prognostications. This model is instrumental in identifying employees at risk of departure, enabling proactive retention interventions.

