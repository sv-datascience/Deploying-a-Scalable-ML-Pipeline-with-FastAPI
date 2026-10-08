# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Sara Vancura as part of the WGU D501 Machine Learning DevOps course (Udacity project "Deploying a Machine Learning Model with FastAPI"). It is a binary classification model built with scikit-learn's `GradientBoostingClassifier` (version 1.5.1) using the default hyperparameters (100 estimators, learning rate of 0.1, max depth of 3) and `random_state=42` so the results can be reproduced. Categorical features are one-hot encoded with scikit-learn's `OneHotEncoder`, and the salary label is converted to 0/1 with a `LabelBinarizer`. I chose gradient boosting over a random forest because it scored a slightly higher F1 score and the saved model file is very small (about 0.14 MB), which makes it easy to store in GitHub and load quickly in the API.

## Intended Use
The model predicts whether a person's annual income is greater than $50K or less than or equal to $50K based on demographic and employment information from the U.S. Census. It is intended for educational purposes, to demonstrate how to build, test, and deploy a machine learning pipeline with a RESTful API using FastAPI. It should not be used to make real decisions about individuals, such as hiring, lending, housing, or insurance eligibility.

## Training Data
The model was trained on the publicly available Census Income dataset (also known as the "Adult" dataset) from the UCI Machine Learning Repository, extracted from the 1994 Census database. The dataset contains 32,561 rows and 15 columns. Fourteen columns are features, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country, and the label is the salary column. About 76% of the records are labeled <=50K and about 24% are labeled >50K, so the classes are imbalanced. The data was split into 80% training data (26,048 rows) and 20% test data (6,513 rows) using a stratified split on the salary column with `random_state=42`. The eight categorical features were one-hot encoded and the six numeric features were left as they were.

## Evaluation Data
The evaluation data is the 20% test set (6,513 rows) held out from the same census dataset. It was processed with the same encoder and label binarizer that were fit on the training data, so the model never saw this data during training. Because the split was stratified, the test set has the same class balance as the full dataset.

## Metrics
The model was evaluated with three metrics: precision, recall, and F1 score. Precision measures how many of the people the model predicted to earn >50K actually earn >50K. Recall measures how many of the people who actually earn >50K the model correctly found. The F1 score is the harmonic mean of precision and recall, which gives a single balanced score.

On the test set, the model achieved a **precision of 0.7881**, a **recall of 0.6237**, and an **F1 score of 0.6963**.

The model was also evaluated on slices of the data for every unique value of each categorical feature, and the full results are saved in `slice_output.txt`. A few examples:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex: Female | 2,158 | 0.8503 | 0.5796 | 0.6893 |
| sex: Male | 4,355 | 0.7784 | 0.6319 | 0.6975 |
| race: White | 5,533 | 0.7906 | 0.6291 | 0.7006 |
| race: Black | 662 | 0.8644 | 0.5484 | 0.6711 |
| race: Amer-Indian-Eskimo | 73 | 0.6000 | 0.3333 | 0.4286 |
| education: HS-grad | 2,120 | 0.8583 | 0.3038 | 0.4488 |
| education: Bachelors | 1,096 | 0.7585 | 0.8192 | 0.7877 |
| education: Masters | 318 | 0.8602 | 0.8791 | 0.8696 |

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, marital status, and native country, and the model uses them as features. This means the model can learn and repeat historical income inequalities that existed in 1994. The slice results show that performance is not equal across groups. For example, recall is lower for women (0.58) than for men (0.63), and it is much lower for the American Indian/Eskimo group (0.33), so the model misses more high earners in those groups. Some slices have very small sample sizes, such as "race: Other" with 45 records, and their perfect or very low scores are not reliable. Because of these issues, the model should not be used for any decision that affects real people.

## Caveats and Recommendations
The census data is from 1994, so income levels, job categories, and the $50K threshold do not reflect today's economy. The classes are imbalanced, which is one reason recall (0.62) is lower than precision (0.79). The model predicts the majority class more often and misses some people who earn more than $50K. Future improvements could include tuning hyperparameters with cross-validation, adjusting the decision threshold or using class weights to improve recall, removing or testing the impact of sensitive features, and retraining the model on more recent census data. Any slice with a small count should be interpreted with caution.
