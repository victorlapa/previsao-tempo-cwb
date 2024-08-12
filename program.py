import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import logging
import psutil

obj_Disk = psutil.disk_usage('/')

print(obj_Disk.percent)

if(obj_Disk.percent >= 99):
    logging.critical("Espaço em disco igual ou acima 99%, encerrando o programa")
    exit()

if(obj_Disk.percent > 80):
    logging.warning("Cuidado! Espaço em disco acima de 80%")

logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logging.info("Iniciando execução...")

file_path = 'curitiba2023.csv'
df = pd.read_csv(file_path, delimiter=';')

df['datetime'] = pd.to_datetime(df['Data'] + ' ' + df['Hora UTC'], format='%Y/%m/%d %H%M UTC')

df.set_index('datetime', inplace=True)

df.drop(columns=['Data', 'Hora UTC'], inplace=True)

df = df.replace(',', '.', regex=True)

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(method='ffill') # type: ignore

features = ['TEMPERATURA DO AR - BULBO SECO, HORARIA (C)', 'UMIDADE RELATIVA DO AR, HORARIA (%)', 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)']
df = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns=features, index=df.index)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 24

X, y = create_sequences(scaled_df.values, SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

logging.info(f'Training shape: {X_train.shape}, Testing shape: {X_test.shape}')

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(features)))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

logging.info("\nTraining and Validation Loss")
for epoch in range(len(history.history['loss'])):
    logging.info(f"Epoch {epoch+1}: Training Loss = {history.history['loss'][epoch]:.4f}, Validation Loss = {history.history['val_loss'][epoch]:.4f}, Training MAE = {history.history['mae'][epoch]:.4f}, Validation MAE = {history.history['val_mae'][epoch]:.4f}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

loss, mae = model.evaluate(X_test, y_test)
logging.info(f'Test Loss: {loss}')
logging.info(f'Test MAE: {mae}')

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 0], label='Actual Temperature')
plt.plot(predictions[:, 0], label='Predicted Temperature')
plt.legend()
plt.show()
