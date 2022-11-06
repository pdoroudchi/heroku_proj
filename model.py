import pandas as pd
import numpy as np
import sklearn
import pickle

df_data = pd.read_csv("adult.csv")

df_data = df_data.drop(['fnlwgt', 'educational-num'], axis = 1) 

col_names = df_data.columns
for c in col_names: 
    df_data = df_data.replace("?", np.NaN) 
df_data = df_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

category_col =['workclass', 'education', 'marital-status', 'occupation', 'relationship', 
               'race', 'gender', 'native-country', 'income']  
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    df_data[col] = labelEncoder.fit_transform(df_data[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping
    
X = df_data.drop('income', axis = 1) 
Y = df_data['income']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
ypred = rf.predict(X_test)

pickle_out = open("model.pkl", "wb")
pickle.dump(rf, pickle_out)
pickle_out.close()