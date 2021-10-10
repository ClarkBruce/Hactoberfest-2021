#starts 
with open(root_path+"dats1.txt", 'r') as f:
    myNames = [line.strip() for line in f]

myNames = [i for i in myNames if i]

all_x = []
all_y = []

for i in range(int(len(myNames)/2)):
  all_x.append(myNames[2*i].split())
  all_y.append(myNames[2*i+1])


all_y = [float(i) for i in all_y]

all_x = [[float(i) for i in e] for e in all_x]

type(all_x[0][0])

all_x = [i[2:3] for i in all_x]


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


import numpy as np
X = all_y.copy()

y_ = all_x.copy()

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
y = min_max_scaler.fit_transform(pd.DataFrame(y_).values)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(pd.DataFrame(X).values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn import linear_model
regressor = linear_model.BayesianRidge()  
regressor.fit(X_train, y_train)

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

input1 = Input(shape=(1,))
l1 = Dense(10, activation = 'relu')(input1)
l2 = Dense(50, activation = 'relu')(l1)
l3 = Dense(50, activation = 'relu')(l2)
out = Dense(3)(l3)

model = Model(inputs = input1, outputs = [out])
model.compile(optimizer='adam', loss = ['mean_squared_error'])

history = model.fit(X_train, [y_train], epochs = 100, batch_size = 64)

y_pred = model.predict(X_test)

y_preds = []
for iy in y_pred:
  temp = []
  temp.append(iy)
  y_preds.append(temp)

y_preds = np.array(y_preds)
y_preds = np.squeeze(y_preds, axis=(1,))
y_preds = [max([0.0],i) for i in y_preds]

df = pd.DataFrame({'Actual0': list(y_test[:,2]), 'Predicted0': list(y_preds[:,2])})

df1 = pd.concat((pd.DataFrame(df[col].tolist()) for col in df), axis=1) 
df1.plot(kind='bar',figsize=(25,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


