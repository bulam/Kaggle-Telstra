# Kaggle-Telstra
First iteration of classification model for the Telstra Kaggle competition.

The separate datasets are joined into one dataframe using numpy and pandas on the 'id' column. All attributes, as well as the location attribute have been converted into categorical variables, with one row per 'id'.

The dataset is then trained using the GradientBoostingClassifier in scikit, split into train and test datasets and measured for accuracy. 
# ToDo
Try using XGBoost Random Forest and PyBrain Neural Network models for potential performance gains 
