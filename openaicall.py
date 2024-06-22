import pandas as pd
import openai
from fetcherv6 import *

def preprocess_data(income_statement, balance_sheet):
    # Check if 'period_end' or 'index' needs to be set as index
    if 'period_end' in income_statement.columns:
        income_statement.set_index('period_end', inplace=True)
    else:
        income_statement.set_index('index', inplace=True)
    
    if 'index' in balance_sheet.columns:
        balance_sheet.set_index('index', inplace=True)
    
    # Transpose the dataframes
    income_statement = income_statement.T
    balance_sheet = balance_sheet.T
    
    # Standardize column names for consistency
    income_statement.columns = [col.lower().replace(" ", "_") for col in income_statement.columns]
    balance_sheet.columns = [col.lower().replace(" ", "_") for col in balance_sheet.columns]
    
    return income_statement, balance_sheet

def create_prompt(income_statement, balance_sheet):
    prompt = f"""
    Analyze the following financial statements to predict if the company's earnings will increase or decrease next year. Follow these steps:

    1. Identify notable changes in financial statement items.
    2. Compute key financial ratios (without limiting the set of ratios).
    3. Provide economic interpretations of the computed ratios.
    4. Predict whether earnings are likely to increase or decrease.
    5. Provide a rationale for the prediction.
    6. Estimate the magnitude of earnings change (large, moderate, small).
    7. Provide a confidence score (0 to 1).

    Income Statement:
    {income_statement.to_string()}

    Balance Sheet:
    {balance_sheet.to_string()}

    Structure your response as follows:
    1. Notable Changes: [Your analysis as consice as possible]
    2. Key Ratios: [Your calculations, simply]
    3. Economic Interpretations: [Your interpretations as consice as possible]
    4. Earnings Prediction: [Increase/Decrease]
    5. Rationale: [Your explanation briefly]
    6. Magnitude: [Large/Moderate/Small]
    7. Confidence Score: [0-1]

    Begin your response with the prediction in the format: 'next_year; earnings_direction; magnitude; confidence_score'
    """
    return prompt
def extract_prediction_from_rationale(rationale):
    first_line = rationale.split('\n')[0]
    if ";" in first_line:
        parts = first_line.split(';')
        if len(parts) >= 4:
            return f"{parts[1].strip()}; {parts[2].strip()}; {parts[3].strip()}"
    return "Prediction not found"

def get_predictions(company_code, output_type="B"):
    # Fetch the data
    income_statement = fetch_income_statement(company_code)
    balance_sheet = fetch_balance_sheet(company_code)
    
    # Preprocess the data
    income_statement, balance_sheet = preprocess_data(income_statement, balance_sheet)
    
    # Determine available years for prediction
    years = income_statement.index.astype(int)
    predictions = []
    
    for i in range(5, len(years)):
        historical_years = years[i-5:i]
        prediction_year = years[i]
        
        # Create the prompt
        historical_income = income_statement.loc[historical_years]
        historical_balance = balance_sheet.loc[historical_years]
        prompt = create_prompt(historical_income, historical_balance)
        
        # Use OpenAI API to analyze the financials
        response = openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            top_p=1,
            logprobs=True,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )
        
        rationale = response.choices[0].message.content.strip()
        
        if output_type == "A":
            predictions.append((prediction_year, rationale))
        elif output_type == "B":
            prediction = extract_prediction_from_rationale(rationale)
            predictions.append((prediction_year, prediction))
    
    return predictions



if __name__ == "__main__":
    company_code = input("enter cvm code") # Replace with actual company code\
    # ask for output type
    output_type = input("Enter output type (A/B): ") # Choose between detailed analysis (A) or prediction only (B)
    predictions = get_predictions(company_code, output_type)
    for prediction in predictions:
        print(f"Prediction for {prediction[0]}: {prediction[1]}")


