
# coding: utf-8

# In[161]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics, preprocessing


# read files into pandas dataframes
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
event_type = pd.read_csv('event_type.csv')
log_feature = pd.read_csv('log_feature.csv')
resource_type = pd.read_csv('resource_type.csv')
severity_type = pd.read_csv('severity_type.csv')

# create categorical variables out of locations
train_crosstab = pd.merge(train, pd.crosstab(train.id, train.location), how='left', left_on='id', right_index=True)
train_crosstab = train_crosstab.drop(['location'], axis=1).drop_duplicates()

# create categorical variables out of resource types
resource_type_crosstab = pd.merge(resource_type, pd.crosstab(resource_type.id, resource_type.resource_type), how='left', left_on='id', right_index=True)
resource_type_crosstab = resource_type_crosstab.drop(['resource_type'], axis=1).drop_duplicates()

# create categorical variables out of severity types
severity_type_crosstab = pd.merge(severity_type, pd.crosstab(severity_type.id, severity_type.severity_type), how='left', left_on='id', right_index=True)
severity_type_crosstab = severity_type_crosstab.drop(['severity_type'], axis=1).drop_duplicates()

# create categorical variables out of event types
event_type_crosstab = pd.merge(event_type, pd.crosstab(event_type.id, event_type.event_type), how='left', left_on='id', right_index=True)
event_type_crosstab = event_type_crosstab.drop(['event_type'], axis=1).drop_duplicates()

# pivot the log feature data to create a variable for each log feature; log feature volume is the scalar value 
log_feature_pivot = pd.merge(log_feature, log_feature.pivot(index='id', columns='log_feature', values='volume'), how='left', left_on='id', right_index=True)
log_feature_pivot = log_feature_pivot.drop(['log_feature','volume'], axis=1).drop_duplicates()


# merge dataframes into one
merge1 = pd.merge(train_crosstab,log_feature_pivot,on='id',how='left')
merge2 = pd.merge(merge1, event_type_crosstab,on='id',how='left')
merge3 = pd.merge(merge2, resource_type_crosstab,on='id',how='left')
merge4= pd.merge(merge3, severity_type_crosstab,on='id',how='left')



#prepare the test file and put in the same format where the locations are categorical variables
test_crosstab = pd.merge(test, pd.crosstab(test.id, test.location), how='left', left_on='id', right_index=True)
test_crosstab = test_crosstab.drop(['location'], axis=1).drop_duplicates()


# merge the test dataframes into one
merge5 = pd.merge(test_crosstab,log_feature_pivot,on='id',how='left')
merge6 = pd.merge(merge5, event_type_crosstab,on='id',how='left')
merge7 = pd.merge(merge6, resource_type_crosstab,on='id',how='left')
merge8= pd.merge(merge7, severity_type_crosstab,on='id',how='left')


# concatenate the merged train and test dataframes to standardize columns across datasets
concatenated_data = pd.concat([merge4,merge8],axis=0)

# separate the train and test data back
train_data = concatenated_data[pd.notnull(concatenated_data['fault_severity'])]
test_data = concatenated_data[pd.isnull(concatenated_data['fault_severity'])]

# replace NaNs with zeros
train_data = train_data.fillna(value=0)
test_data = test_data.fillna(value=0)

# split the train dataset by columns into training columns and the target column

cols = [col for col in train_data.columns if col not in ['id', 'fault_severity']]
train_columns = train_data[cols]
target_columns = train_data['fault_severity']

train_columns.to_csv('train_columns.csv')

# create Gradient Boosting Classifier object
gbc = GradientBoostingClassifier(n_estimators=200, max_depth=7)

# further split train dataset into train and test for cross-validation
X_train, X_test, Y_train, Y_test = train_test_split(train_columns, target_columns, test_size=0.2,random_state=3)

# train the model
gbc.fit(X_train, Y_train)

# get predictions from the model
Y_pred = gbc.predict(X_test)

# show model accuracy
print(metrics.accuracy_score(Y_test,Y_pred))



# In[162]:

# take only the attribute columns from the test dataset
cols = [col for col in test_data.columns if col not in ['id', 'fault_severity']]
test_columns = test_data[cols]

# get predictions on the test dataset 
Y_pred_test = gbc.predict(test_columns)

# add the predictions to the original file and select only id and Predictions columns
predictions = pd.DataFrame(Y_pred_test, columns=['Predictions'])
test_w_predictions = test_data.join(predictions)[['id', 'Predictions']]


# format predictions according to the requested submission format
test_w_predictions = test_w_predictions.pivot(index='id', columns='Predictions', values='Predictions')
test_w_predictions = test_w_predictions.notnull().astype(int)
test_w_predictions = test_w_predictions.rename(columns={0:'predict_0',1:'predict_1',2:'predict_2'})
test_w_predictions.to_csv('predictions.csv')


# In[151]:



