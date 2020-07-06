

import numpy as np
import pickle
import pandas as pd
from flasgger import Swagger
import streamlit as st 
from flask import Flask,request


app=Flask(__name__)
Swagger(app)

model=pickle.load(open('model.pkl','rb+'))

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["GET"])
def predict():
    
    """Let's Predict the flight price prediction.
    ---
    parameters:  
      - name: Name of Airlines
        in: query
        enum: [" ","Air India","Go-Air","Indigo","Jet Airways","Jet Airways Business","Multi Carriers","Multi Carriers Premium Economy","Spicejet","Trujet","Vistara","Vistara Premium Economy"]
        description: Select the airlines for which you want to predict the price
        required: true
        type: string
      - name: Source
        in: query
        enum: [" ","Bangalore","Chennai","Delhi","Kolkata","Mumbai"]
        description: Select the city where your journey starts
        required: true
        type: string
      - name: Destination
        in: query
        enum: [" ","Bangalore","Cochin","Hyderabad","Delhi","Kolkata","New Delhi"]
        description: Select he destination city you wanted to reach
        required: true
        type: string
      - name: Date of Journey
        in: query
        type: string
        description: Format is DD/MM/YYYY
        required: true
      - name: Arrival Time
        in: query
        type: string
        description: Format is HH:MM
        required: true
      - name: Departure Time
        in: query
        type: string
        description: Format is HH:MM
        required: true
      - name: Total Duration
        in: query
        type: number
        description: Format is in minutes
        required: true
      - name: Total No of Stops
        in: query
        enum: ["Non-Stop","1-Stop","2-Stop","3-Stop","4-Stop"]
        description: Select No of Stops
        required: true
        type: string
    responses:
        200:
            description: The Predicted Flight Price
        
    """
    airlines={"Air India":0,"Go-Air":1,"Indigo":2,"Jet Airways":3,"Jet Airways Business":4,"Multi Carriers":5,"Multi Carriers Premium Economy":6,"Spicejet":7,"Trujet":8,"Vistara":9,"Vistara Premium Economy":10}
    source={"Bangalore":0,"Chennai":1,"Delhi":2,"Kolkata":3,"Mumbai":4} 
    destination={"Bangalore":0,"Cochin":1,"Hyderabad":2,"Delhi":3,"Kolkata":4,"New Delhi":5}
    stops={"Non-Stop":0,"1-Stop":1,"2-Stop":2,"3-Stop":3,"4-Stop":4}
    features=np.zeros(30)
    airline=request.args.get('Name of Airlines')
    src=request.args.get('Source')
    dest=request.args.get('Destination')
    dtofjrny=request.args.get('Date of Journey')
    artime=request.args.get('Arrival Time')
    deptime=request.args.get('Departure Time')
    total=request.args.get('Total Duration')
    totalstops=request.args.get('Total No of Stops')
    features[0]=total
    features[1]=stops[totalstops]
    dtofjrny=dtofjrny.split('/')
    features[2]=int(dtofjrny[1])
    features[3]=int(dtofjrny[0])
    artime=artime.split(':')
    deptime=deptime.split(':')
    features[4]=int(deptime[0])
    features[5]=int(deptime[1])
    features[6]=int(artime[0])
    features[7]=int(artime[1])
    features[8+airlines[airline]]=1
    features[19+source[src]]=1
    features[24+destination[dest]]=1
    pred=model.predict([features])
    prediction=pred[0]
    return str(prediction)



app.run(debug=False)   
    