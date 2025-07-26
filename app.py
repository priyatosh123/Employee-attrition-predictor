import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("rf_model.pkl")

# Define prediction function
def predict_attrition(age, income, total_years):
    df = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": income,
        "TotalWorkingYears": total_years
    }])
    pred = model.predict(df)[0]
    return "Yes" if pred == 1 else "No"

# Define Gradio Interface
interface = gr.Interface(
    fn=predict_attrition,
    inputs=[
        gr.Number(label="Age", value=22),
        gr.Number(label="Monthly Income", value=1500),
        gr.Number(label="Total Working Years", value=1)
    ],
    outputs=gr.Textbox(label="Attrition Prediction"),
    title="Employee Attrition Predictor",
    description="Enter employee details to predict whether they may leave the company (Attrition)."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
