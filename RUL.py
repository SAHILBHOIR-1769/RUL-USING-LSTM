from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model_path = 'models/lstm_model_no_es.h5'
model = load_model(model_path)

train_df = pd.read_csv('D:\SEM 8\Major Project\FINAL\data\Train_Data_CSV.csv')
test_df = pd.read_csv('D:\SEM 8\Major Project\FINAL\data\Test_Data_CSV.csv')

def x_reshape(df, columns, sequence_length):
    data = df[columns].values
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements-sequence_length-10), range(sequence_length, num_elements-10)):
        yield(data[start:stop, :])

def get_x_slices(df, feature_columns):
    feature_list = [list(x_reshape(df[df['Data_No'] == i], feature_columns, 20)) for i in range(1, df['Data_No'].nunique() + 1) if len(df[df['Data_No']  == i]) > 20]
    feature_array = np.concatenate(list(feature_list), axis=0).astype(np.float64)
    length = len(feature_array) // 128
    return feature_array[:length*128]

def y_reshape(df, sequence_length, columns=None):
    if columns is None:
        columns = ['Differential_pressure']
    data = df[columns].values
    num_elements = data.shape[0]
    return data[sequence_length+10:num_elements, :]

def get_y_slices(df):
    label_list = [y_reshape(df[df['Data_No'] == i], 20) for i in range(1, df['Data_No'].nunique()+1)]
    label_array = np.concatenate(label_list).astype(np.float64)
    length = len(label_array) // 128
    return label_array[:length*128]

scaler = MinMaxScaler()
X_train = get_x_slices(train_df, ['Differential_pressure', 'Flow_rate'])
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = get_x_slices(test_df, ['Differential_pressure', 'Flow_rate'])
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
y_train = get_y_slices(train_df)
y_test = get_y_slices(test_df)

def get_individual_x_slices(differential_pressure, flow_rate, sequence_length):
    data = np.array([[differential_pressure, flow_rate]])
    slices = []
    for start in range(len(data) - sequence_length - 10):
        end = start + sequence_length
        slices.append(data[start:end, :])
    slices = np.array(slices)
    if len(slices) == 0:
        slices = np.zeros((1, sequence_length, 2))
    return slices

def get_individual_y_slices(differential_pressure, sequence_length):
    data = np.array([differential_pressure])
    return data[sequence_length+10:]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    differential_pressure = float(request.form['differential_pressure'])
    flow_rate = float(request.form['flow_rate'])

    # Get individual slices
    x_slices = get_individual_x_slices(differential_pressure, flow_rate, 20)
    y_slices = get_individual_y_slices(differential_pressure, 20)

    # Reshape and preprocess input data for model prediction
    x_slices = scaler.transform(x_slices.reshape(-1, x_slices.shape[-1])).reshape(x_slices.shape)

    # Make prediction
    prediction = model.predict(x_slices)

    # Remove negative sign if exists
    prediction = np.abs(prediction) if prediction < 0 else prediction

    # Print some debug info
    print("Input values:", differential_pressure, flow_rate)
    print("Prediction:", prediction)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
