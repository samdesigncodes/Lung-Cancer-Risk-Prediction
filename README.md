# Lung-Cancer-Risk-Prediction 📝
## Data Source 📊 [Kaggle](https://www.kaggle.com/competitions/iste-transcend24-hackathon/data).

## Introduction 🟠
In this project, the Linear Regression Model and Gradient Boosting Regressor are used to analyze selected patient features such as Age, Smoking History, Family History of Lung Cancer, Income Level, etc., to predict their Lung Cancer Risk Factor. The cancer risk factor scores range between 0 and 100, indicating the risk score of a patient developing lung cancer. Both models were trained and evaluated using the Root Mean Squared Error (RMSE). The results show that the Linear Regression Model and Gradient Boosting Regressor yielded an RMSE of 6.81 and 0.88, respectively. The Gradient Boosting Regressor significantly outperforms the Linear Regression model, which suggests that Gradient Boosting better captures the complexity and non-linear relationships within the data.
The findings underscore the potential of machine learning approaches, particularly Gradient Boosting in early predicting Lung Cancer in patients, facilitating timely diagnosis and treatment decisions in clinical practices. 

## Linear Regression Model
The linear regression model is a supervised learning method that models a relationship between independent and dependent variables by fitting a linear equation into the observed data. The main goal is to find the best-fitting line through the data points. This is first used to train the dataset for its simplicity and interpretability. 
Y and X are the Dependent (Risk) and independent variables (Other Features such as Smoking History and Family History), respectively.
After training the dataset with Linear Regression, an RMSE of 6.81 was obtained.

## Gradient Boosting 🟠
A more complex model is used in the dataset (Gradient Boosting using LightGBM). Gradient Boosting is a machine-learning technique that can be used for both regression and classification problems. 
It produces a prediction model as an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion and generalizes them by allowing optimization of an arbitrary differentiable loss function. Gradient Boosting can handle mixed data types and often provides superior predictive performance. 
After training the dataset with Gradient Boosting, it yielded an RMSE of 0.88.
Gradient Boosting Regressor significantly outperforms the Linear Regression model, suggesting complexity and non-linear relationships within the data are better captured by Gradient Boosting.
Therefore, the Gradient Boosting Regressor is used to predict the test dataset.

## Hyperparameter Tuning 🟠
Before testing with Gradient Boosting, hyperparameter tuning is performed to help optimize the model parameters. Critical parameters for tuning in Gradient Boosting include the number of trees (n_estimators), learning rate, max depth of trees, min sample split, min samples leaf, and max features. 
Grid Search estimates the model's parameters due to its exhaustiveness. Grid search exhaustively searches through a manually specified subset of hyperparameter space. It controls the learning process, which may impact the model's performance. It achieves this by systematically working through multiple combinations of hyperparameter values by defining a grid over the hyperparameter space and evaluating the model's performance for each point on the grid.

After performing the Grid Search using "GridSearchCV" over a defined parameter grid of:

param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': [None, 'sqrt', 'log2'], 'subsample': [0.6, 0.8, 1.0] }

## Results and Discussion 💡
In this Project, we aimed to predict lung cancer risk factors using two machine learning models. The Analysis was based on a dataset with features such as Age, Smoking History, Genetic predisposition Score, Exposure to Carcinogens, Air Pollution Index, Dietary Habits Score, Physical Activity Level, BMI, Family History of Lung Cancer, and Income level.
The performance of the models was evaluated using the Root Mean Squared Error (RMSE). Lower values of RMSE indicate more accurate predictions. The initial model, a simple linear regression, provided a baseline RMSE 6.81. Subsequently, we employed a Gradient Boosting Regressor, which significantly improved the RMSE to 0.88.

### Feature Importance and Model Insights
The gradient-boosting model highlighted several key predictors of lung cancer risk. Smoking History and Genetic Predisposition Score were identified as the most influential factors. [3] This is consistent with existing medical literature that recognizes smoking as a primary risk factor for lung cancer. The influence of genetic factors also aligns with current understanding, suggesting that individuals with a family history of lung cancer or specific genetic markets are at increased risk

Further model enhancement was achieved through hyperparameter tuning, with the best model configuration using parameters such as 'n_estimators=100', 'max_depth=3', and 'learning_rate=0.1'. This configuration yielded a slightly improved RMSE, demonstrating the effectiveness of model tuning to enhance predictive accuracy

### Results

| ID  | Predicted Risk |
| ------------- | ------------- |
| 1401  | 3.73  |
| 1402  | 96.59  |
| 1403  | 3.21  |
| 1404  | 24.44  |
| 1405  | 4.45  |
