import pandas as pd
import numpy as np
import io

from model import return_rate
from io import BytesIO
from fastapi import FastAPI, File, UploadFile


app = FastAPI()


@app.post("/breathing_rate/")
async def get_breathing_rate(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return {"error": "Please upload a CSV file"}
    else:
      contents = await file.read()
      df = pd.read_csv(BytesIO(contents), header=None)
      
      breathing_rate = return_rate(df)
      
      return {"breathing_rate": breathing_rate}

