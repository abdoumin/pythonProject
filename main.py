from flask import Flask,request,jsonify
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.utils import resample

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    # cgpa = request.form.get('cgpa')
    # iq = request.form.get('iq')
    # profile_score = request.form.get('profile_score')

    nitrogen = request.form.get('nitrogen')
    phosporus = request.form.get('phosporus')
    potassium = request.form.get('potassium')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    ph = request.form.get('ph')
    rainfall = request.form.get('rainfall')

    input_query = np.array([nitrogen,phosporus,potassium,temperature,humidity,ph,rainfall])
    single_pred = np.array(input_query).reshape(1, -1)

    prediction = model.predict(single_pred)

    return jsonify({'prediction':f"{str(prediction.item().title())} are recommended by the A.I for your farm."})

if __name__ == '__main__':
    app.run(debug=True)