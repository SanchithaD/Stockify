import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
tf.random.set_seed(1234)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from vecstack import stacking
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import pickle


df_cnbc = pd.read_csv('data/sentiment_cnbc_groupday.csv', parse_dates = ['dates'], index_col = 'dates')
df_fortune = pd.read_csv('data/sentiment_fortune_groupday.csv', parse_dates = ['dates'], index_col = 'dates')
df_reuters = pd.read_csv('data/sentiment_reuters_groupday.csv', parse_dates = ['dates'], index_col = 'dates')
df_wsj = pd.read_csv('data/sentiment_wsj_groupday.csv', parse_dates = ['dates'], index_col = 'dates')
df_sp500 = pd.read_csv('data/SP500.csv', parse_dates = ['dates'], index_col = 'dates')
df = pd.concat([df_cnbc['compound'], 
    df_fortune['compound'], df_reuters['compound'], df_wsj['compound'], df_sp500['High'], df_sp500['Low'], df_sp500['Adj Close']], axis=1)
df = df.dropna(how='any',axis=0) 
df = df.set_axis(['cnbc', 'fortune', 'reuters', 'wsj', 'high', 'low', 'closing'], axis=1, inplace=False)
print(df)

plt.figure(figsize = (10, 6))
plt.plot(df.index, df['closing'], color ='black')
plt.xlabel("Date", {'fontsize': 12}) 
plt.ylabel("Closing Price", {'fontsize': 12})

train_size = int(len(df)*0.7)
train_dataset, test_dataset = df.iloc[:train_size],df.iloc[train_size:]

plt.figure(figsize = (10, 6))
plt.plot(train_dataset['closing'])
plt.plot(test_dataset['closing'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend(['Train set', 'Test set'], loc='upper right')
print('Dimension of train data: ',train_dataset.shape)
print('Dimension of test data: ', test_dataset.shape)

X_train = train_dataset.drop('closing', axis = 1)
y_train = train_dataset.loc[:,['closing']]

X_test = test_dataset.drop('closing', axis = 1)
y_test = test_dataset.loc[:,['closing']]
print(len(y_test))

scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))

input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)

train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)

test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)

def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)
TIME_STEPS = 26
X_test, y_test = create_dataset(test_x_norm, test_y_norm,   
                                TIME_STEPS)
X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                  TIME_STEPS)
print(y_train.shape[0])

print('X_train.shape: ', X_test.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_train.shape)

def create_model(units, m):
    model = Sequential()
    model.add(m (units = units, return_sequences = True,
                input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2))
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.compile(loss='mse', optimizer='adam')
    return model

model_gru = create_model(100, GRU)
model_lstm = create_model(100, LSTM)

def fit_model(model):
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)
    history = model.fit(X_train, y_train, epochs = 100,  
                        validation_split = 0.2, batch_size = 32, 
                        shuffle = False, callbacks = [early_stop])
    return history

history_lstm = fit_model(model_lstm)
history_gru = fit_model(model_gru)
print(y_test.shape[0])
def plot_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
plot_loss (history_lstm)
plot_loss (history_gru)

y_test = y_test.reshape(-1,1)
y_test = scaler_y.inverse_transform(y_test)
y_train = scaler_y.inverse_transform(y_train)

# # Make prediction
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction

prediction_lstm = prediction(model_lstm)
prediction_gru = prediction(model_gru)
# Plot true future vs prediction
def plot_future(prediction, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='True Future')     
    plt.plot(np.arange(range_future),np.array(prediction),
            label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Day')
    plt.ylabel('Closing Price')
plot_future(prediction_lstm, y_test)
plot_future(prediction_gru, y_test)


test_pred_gru = pd.DataFrame(model_gru.predict(X_test))
test_pred_lstm = pd.DataFrame(model_lstm.predict(X_test))

models = dict()
models['gru'] = test_pred_gru
models['lstm'] = test_pred_lstm

level0 = list()
level0.append(('gru', test_pred_gru))
level0.append(('lstm', test_pred_lstm))

level1 = test_pred_gru
# define the stacking ensemble
model = StackingClassifier(estimators=level0, final_estimator=level1, cv=2)

# all_models = [model_gru, model_lstm]

# s_train, s_test = stacking(all_models, X_train, y_test, X_test, regression=True, n_folds=4)

# final_model = LinearRegression()
 
# # fitting the second level model with stack features
# final_model = final_model.fit(s_train, y_train)
 
# # predicting the final output using stacking
# pred_final = final_model.predict(X_test)
 
# # printing the root mean squared error between real value and predicted value
# print(mean_squared_error(y_test, pred_final))
ln = LogisticRegression()
stack = StackingCVRegressor(regressors=(model_lstm, model_gru),
                            meta_regressor=ln, cv=12,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

#stack.fit(X_train, y_train)

#X_test.columns = ['cnbc', 'fortune', 'reuters', 'wsj', 'high', 'low']
pred = stack.predict(model)
# score = r2_score(y_test, pred)
plot_future(pred, y_test)
