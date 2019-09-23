from flask import Flask, render_template, flash, request, url_for, redirect
import pickle
# import re
import warnings
import pandas as pd
import tensorflow as tf
from keras import models, layers
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from keras import backend as K
from plot_graphs import build_plot,build_graph



warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates")


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/multiforecasting')
def analysis():
    model2 = pickle.load(open("/home/anthony/PycharmProjects/first_project/model_multi.dat", "rb"))

    #data = pd.read_csv("/home/anthony/PycharmProjects/first_project/Demand+Weather.csv")
    #data = data.set_index('Timestamp')

    act_df = pd.read_csv('/home/anthony/PycharmProjects/first_project/Multi_Actual.csv')
    pred_df = pd.read_csv('/home/anthony/PycharmProjects/first_project/multi_Predicted.csv')

    act_x = act_df['Timestamp']
    act_y = act_df['demand']

    pred_x = pred_df['Timestamp']
    pred_y = pred_df['predicted']


    graph3_url = build_graph(pred_x, pred_y)
    graph4_url = build_plot(pred_x,act_y,pred_y)


    K.clear_session()

    return render_template('analysis.html',
                           graph3=graph3_url,
                           graph4=graph4_url)


@app.route('/uniforecasting')
def forecasting():
    model1 = pickle.load(open("/home/anthony/PycharmProjects/first_project/model_uni_final_new.dat", "rb"))
    # preprocessing data

    data = pd.read_csv("/home/anthony/PycharmProjects/first_project/KSEB14-18.csv")
    # data.head()
    data = data.set_index('timestamp')

    # scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    x_test_df = pd.read_csv("/home/anthony/PycharmProjects/first_project/x_test_df.csv")
    # X_test = x_test_df.to_numpy()
    X_test = x_test_df.as_matrix()

    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    testPredict = model1.predict(X_test)
    # scaler.fit_transform
    testPredict = scaler.inverse_transform(testPredict)

    predict_df = pd.DataFrame(testPredict, columns=['demand_forecasting'])
    # y_test_df = pd.DataFrame(y_test)

    ######### plotss #############################
    actual_df = pd.read_csv('/home/anthony/PycharmProjects/first_project/Univar_Actual.csv')
    predicted_df = pd.read_csv('/home/anthony/PycharmProjects/first_project/Univar_Predicted.csv')

    #predicted_df['Timestamp'] = pd.to_datetime(predicted)
    x_pred = predicted_df['timestamp']
    y_pred = predicted_df['predicted']

    x_act = actual_df['timestamp']
    y_act = actual_df['demand']

    graph1_url = build_graph(x_pred, y_pred)
    graph2_url = build_plot(x_pred,y_act,y_pred)



    K.clear_session()

    return render_template("forecasting.html", tables=[predict_df.head(10).to_html(classes='demand_data')],
                           titles=predict_df.columns.values,
                           graph1=graph1_url,
                           graph2=graph2_url)


if __name__ == '__main__':
    app.run(debug=True)
