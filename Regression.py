import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("REGRESSION")
st.header("Housing Prices in HYDERABAD")
data1 = pd.read_csv('Datasets/hyderabad.csv')
data2 = pd.read_csv('Datasets/bangalore.csv')
data3 = pd.read_csv('Datasets/mumbai.csv')
data4 = pd.read_csv('Datasets/delhi.csv')
data5 = pd.read_csv('Datasets/chennai.csv')
data6 = pd.read_csv('Datasets/kolkata.csv')
data1.head()


#plt.subplot(1, 2, 1)
#plt.scatter(data1['Price'], data1['Area'], color = 'red')
#plt.subplot(1, 2, 2)
#plt.scatter(data2['Price'], data2['Area'], color = 'blue')

#plt.subplot(1, 2, 1)
#plt.scatter(data3['Price'], data3['Area'], color = 'green')
#plt.subplot(1, 2, 2)
#plt.scatter(data4['Price'], data4['Area'], color = 'yellow')

#plt.subplot(1, 2, 1)
#plt.scatter(data5['Price'], data5['Area'], color = 'orange')
#plt.subplot(1, 2, 2)
#plt.scatter(data6['Price'], data6['Area'], color = 'black')

y = np.array(data1['Price'])
#data = data1
#del data['Price']
#del data['Location']
X = np.array(data1['Area']).reshape(-1,1)

from sklearn.model_selection import train_test_split
def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test(X, y)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

from sklearn.metrics import r2_score
def prediction_acc(y_true, y_predict):
    return r2_score(y_true,y_predict)

X_test = X_test.reshape(-1,1)
y_predict = reg.predict(X_test)

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_predict, color = 'blue')
plt.xlabel('Area')
plt.ylabel('Price')
st.pyplot()

#Checking Accuracy
r2 = prediction_acc(y_test, y_predict)
st.write("R2 SCORE OF THE MODEL = ",r2)

Amount = st.number_input('Insert The Area of HOUSE')
Amount = float(Amount)
A = np.array(Amount).reshape(-1,1)
if Amount>1:
    ans = reg.predict(A)
    if ans > 0:
        st.write("THIS IS EXPECTED PRICE IN HYDERABAD",ans)
    else:
        st.write("ZERO")
else:
    st.write('insufficient Amount')


st.write('MADE BY : SHAURYA ADITYA SINGH')
st.write('ORIGINALX')
st.write('DATASET FROM = https://www.kaggle.com/ruchi798/housing-prices-in-metropolitan-areas-of-india')