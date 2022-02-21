import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('insurance.csv')
df = df[['age','sex','smoker', 'charges']]
df.columns = ['age', 'sex', 'smoker', 'Insurance_Amount']

# took only 3 attributed to create the model
X = df[['age', 'sex', 'smoker']]
y = df['Insurance_Amount']

# sex, smoker are factor variables, so dummified both of these
X = pd.get_dummies(X)

# loaded the linear regression module
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


regressor.fit(X,y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))

## calling the model to predict 
data =[]

item = [20, 'Male', 'No']

# As the The training data was dummified one, so we have to pass the 
# test data in the same format ('age','sex_female',	'sex_male',
# 'smoker_no','smoker_yes

data.append(item[0])
if item[1] == 'Male':
    data.append(0)
    data.append(1)
else:
    data.append(1)
    data.append(0)

if item[2] == 'No':
    data.append(1)
    data.append(0)
else:
    data.append(0)
    data.append(1)

print(data)

# this is single sample
print(model.predict([data]))