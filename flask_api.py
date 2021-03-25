# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 13:09:27 2021

@author: Raghu
"""

from flask import Flask, request
import pandas as pd
import  numpy as np
import pickle 

app = Flask(__name__)
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome all hello world"


@app.route('/predict')
def predictAuthentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=model.predict([[variance,skewness,curtosis,entropy]])
    return "Hello The answer is" + str(prediction)


@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction= model.predict(df_test)
    return "The predicted values of csv is" + str(list(prediction))



if __name__ == '__main__':
    app.run()