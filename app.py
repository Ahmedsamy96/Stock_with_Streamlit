#Open (https://finance.yahoo.com/most-active/) Choose a stock to predict

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import keras

start = '2009-12-12'
end = '2021-12-12'

st.title("Build and Deploy Stock Market App Using Streamlit")
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = data.DataReader(user_input,'yahoo',start,end)

#Descriping data
st.header("A Basic Data Science Web Application")
st.subheader("Data From 2009 to 2021")
st.write(df.describe())


st.subheader("Closing Price vs. Time")
ma100 = df.Close.rolling(100, win_type ='triang').mean()
ma200 = df.Close.rolling(200, win_type ='triang').mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)


#Data Spliting
train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test = pd.DataFrame(df['Close'][int(len(df)*0.70):])


#Data Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(train)

important_note = """
To understand the methodology of how we predict next price of the stream of ( Close data ) we take in consideration for example last ten prices in Close and depending on them we predict next number in consequece , In the next iteration we forget first used number of the ten numbers and insert to them the last predicted number to make our new prediction.
First predict : 25 63 87 96 54 23 58 44 55 65 >> 44 -> second predict : 63 87 96 54 23 58 44 55 65 44 >> 33
"""

x_train = []
y_train = []

for i in range(100 , scaled_train.shape[0]):
    x_train.append(scaled_train[i-100 :i])
    y_train.append(scaled_train[i,0])
    
x_train , y_train = np.array(x_train) , np.array(y_train)



reconstructed_model = keras.models.load_model("my_h5_model.h5")

past_100_datatrain = train.tail(100)
final_df = past_100_datatrain.append(test, ignore_index = True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100 :i])
    y_test.append(input_data[i,0])
    
x_test , y_test = np.array(x_test) , np.array(y_test)

y_prediction = reconstructed_model.predict(x_test)

# very import now i will reverse my scale process in an amazing way

#get the scale ratio 
#print(scaler.scale_)
scaler_fac = scaler.scale_
#set scale factor
scale_factor = 1/scaler_fac[0] 

#reverse scalling
y_prediction = y_prediction * scale_factor
y_test = y_test * scale_factor

st.subheader("Real Price vs. Predicted Price")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label= 'Original value')
plt.plot(y_prediction,'r' , label= 'Predicted value')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)










