# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:42:47 2024

@author: koralsenturk
"""

# %%

#LOAD MODEL
from joblib import dump, load
filename = "myFirstSavedModel.joblib"

clf = load(filename)
#%%

from  sklearn import  datasets
dataSet=datasets.load_iris()
features = dataSet.data
labels = dataSet.target
labelsNames =list(dataSet.target_names)

#%%

from typing import Union

from fastapi import FastAPI, Request
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request, name="base.html")


@app.get("/predict/",response_class=HTMLResponse)
async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float):
   

   testData= np.array([L1,W1,L2,W2]).reshape(-1,4)
   probabilities = clf.predict_proba(testData)[0]
   predicted = np.argmax(probabilities)
   probability = probabilities[predicted]
   predicted = labelsNames[predicted]
   
   return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={"probabilities": probabilities,
                 "predicted": predicted,
                 "probability": probability
                 }
        )


