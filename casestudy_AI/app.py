import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

class BankNote(BaseModel):
    Variance:float
    skewness:float
    curtosis:float
    entropy:float

app=FastAPI()
pickle_in=open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {"message":"hello stranger"}

@app.get('/{name}')
def get_name(name:str):
    return {"message":"f'{name}"}
@app.post('/predict')
def BankNote_Prediction(data:BankNote):
     data=data.dict()
     print(data)
     print("hello")
     variance=data['variance']
     skewness=data['skewness']
     curtosis=data['curtosis']
     entropy=data['entropy']
     prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
     if prediction[0] > 0.5:
         prediction ="fake note"
     else:
         prediction ="Bank note"
     return{
        'prediction': prediction 
        }


if __name__=='__main__':
    uvicorn.run(app ,host='0.0.0.0',port=8080)
  



