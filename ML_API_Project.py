from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import pandas as pd
import joblib

model = joblib.load("Ashutosh's_Placement_Prediction")
scaler = joblib.load("scaler.pkl")

Model_version = "1.0.0 "

app = FastAPI()

class UserInput(BaseModel):
    CGPA : Annotated[float, Field(..., ge= 0.0, le= 10.0, description= "Enter your CGPA")]
    Internships : Annotated[int, Field(..., ge= 0, le= 2, description= "Enter how many Internships you have completed")]
    Projects : Annotated[int, Field(..., ge= 0, le= 3, description= "Enter how many Projects you have completed")]
    Workshops_Certifications : Annotated[int, Field(..., ge= 0, le=2, description= "Enter how may Workshop you attend or How many Certficate you get ")]
    AptitudeTestScore : Annotated[int, Field(..., ge= 0, le= 100, description= "Enter Aptitude Test Score")]
    SoftSkillsRating : Annotated[float, Field(..., ge= 0.0, le= 5.0, description= "Enter your Soft Skills Rating")]
    ExtracurricularActivities : Annotated[bool, Field(..., description= "Do you have done Extra Curricular Activities")]
    PlacementTraining : Annotated[bool, Field(..., description= "Do you have completed your Placement Tranning")]
    SSC_Marks : Annotated[int, Field(..., ge= 0, le= 100, description= "Enter your Class 10th marks")]
    HSC_Marks : Annotated[int, Field(..., ge= 0, le= 100, description= "Enter your Class 12th marks")]

@app.get("/")
def home():
    return {"Message" : "Placement Predictor API"}

@app.get("/health")
def health_check():
    return {"Status" : "Ok","Version" : Model_version, "Model_loaded" : model is not None}

@app.post("/Predict")
def predict_placement(data : UserInput):
    input_df = pd.DataFrame([{
        "CGPA" : float(data.CGPA),
        "Internships" : int(data.Internships),
        "Projects" : int(data.Projects),
        "Workshops/Certifications" : int(data.Workshops_Certifications),
        "AptitudeTestScore" : int(data.AptitudeTestScore),
        "SoftSkillsRating" : float(data.SoftSkillsRating),
        "ExtracurricularActivities" : int(data.ExtracurricularActivities),
        "PlacementTraining" : int(data.PlacementTraining),
        "SSC_Marks" : int(data.SSC_Marks),
        "HSC_Marks" : int(data.HSC_Marks)
        }])
    
    try:
        scaled_input = scaler.transform(input_df)
        prediction = int(model.predict(scaled_input)[0])
        result = "Placed" if prediction == 1 else "Not Placed"
        return JSONResponse(status_code= 200, content= {"Prediction" : result})
    except Exception as e :
        return JSONResponse (status_code= 500, content= str(e))