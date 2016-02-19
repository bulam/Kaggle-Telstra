# Kaggle-Telstra
First iteration of classification model for the Telstra Kaggle competition.

The separate datasets are joined into one dataframe using numpy and pandas on the 'id' column. All attributes, as well as the location attribute have been converted into categorical variables, with one row per 'id'.

The dataset is then split into train and test datasets. The Gradient Boosting Classifier, Random Forest, and a simple linear logistic model were tested and measured for accuracy against a sample size of 0.3, with the Gradient Boosting Classifier showing the best performance with an accuracy of 0.76. The script here only includes the implementation for Gradient Boosting Classifier. 

# ToDo
- Try using XGBoost Random Forest and PyBrain Neural Network models for potential performance gains. 
- Continue to engineer and test new features
- Test dropping certain features
- Tune model paramaters
