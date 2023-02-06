import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib # 모델 내보내기
  
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('ml_data/gangnam.csv', encoding='cp949')
df['CNTRCT_DE'] = pd.to_datetime(df['CNTRCT_DE'])
plt.subplot()
plt.plot(df['CNTRCT_DE'], df['RENT_GTN'])
plt.legend()
plt.tight_layout()

re_df = df.filter(['RENT_GTN'])
dataset = re_df.values
training = int(np.ceil(len(dataset) * .95))



scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(re_df)

train_data = scaled_data[0:int(training), :]
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64,
                            return_sequences=True,
                            input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.summary

model.compile(optimizer='adam',
            loss='mean_squared_error')
history = model.fit(x_train,
                    y_train,
                    epochs=20)

test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# evaluation metrics
mse = np.mean(((predictions - y_test) ** 2))
# print("MSE", mse)
# print("RMSE", np.sqrt(mse))

train = df[:training]
test = df[training:]
test['Predictions'] = predictions

# plt.figure(figsize=(10, 8))
# plt.plot(train['RENT_GTN'])
# plt.plot(test[['RENT_GTN', 'Predictions']])
# plt.title('gangnam')
# plt.xlabel('Date')
# plt.ylabel("RENT_GTN")
# plt.legend(['Train', 'Test', 'Predictions'])


model_file = open('/content/drive/MyDrive/Colab Notebooks/project/models/gangnam.pkl', 'wb')
joblib.dump(model, model_file)
model_file.close()