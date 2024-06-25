import pandas as pd
import openai
from fetcherv6 import *
import requests
import json
import pandas as pd


system_prompt=f"""As a seasoned Brazilian financial analyst, your expertise lies in interpreting financial reports to assess company health and predict future earnings. 
Analyze financial data, utilize key ratios and historical trends to forecast performance, and present your findings concisely."""

def create_prompt(income_statement, balance_sheet):
    prompt = f"""
    Analyze the balance sheet, income statement, and cash flow statement of this company to assess its financial health and performance. 
    Identify key financial ratios and trends to provide a comprehensive overview of its financial position.
    Analyze the following financial statements to predict if the company's earnings will increase or decrease next year.
    Identify patterns, anomalies, and potential risks or opportunities for future growtha comprehensive assessment of its financial health
    Follow these steps in your analysis.

    1. Read through the financial statement items and identify notable trends and changes.
    2. Compute financial ratios useful for the analysis.
    3. Make economic and analytical interpretations of the computed ratios.
    4. Put all analysis together to predict whether earnings are likely to increase or decrease.
    5. Summarize the rationale for the prediction.
    6. Estimate the magnitude of earnings change (large, moderate, small).
    7. Provide a confidence score (0 to 1).

    Income Statement:
    {income_statement.to_string()}

    Balance Sheet:
    {balance_sheet.to_string()}

    Structure your response output as JSON in the following format, maintaining under 500 tokens for the response:
    {{  
        "Year": the predicted year",
        "earnings direction": "increase or decrease",
        "magnitude": "large, moderate, small",
        "confidence score": "0 to 1",
        "summary of rationale": "Brief rationale for the prediction"
    }}
    """
    return prompt
def calculate_earnings_direction(cvm_code):
    data = retrieve_income_with_lenght(cvm_code)
    income_statement = data['income_statement'].T
    years = income_statement.columns[-5:]  # Get the last 5 years
    earnings_direction = {}

    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i - 1]
        earnings_direction[current_year] = income_statement[current_year] - income_statement[previous_year]

    earnings_direction = pd.DataFrame(earnings_direction)
    earnings_direction = earnings_direction.iloc[:, -1]  # Select the last column
    earnings_direction.index.name = 'Year'  # Set the index name

    return earnings_direction

def get_predictions(company_code):
    # Create a dataframe to store the predictions
    predictions = pd.DataFrame()

    # Fetch the data
    income_statement = retrieve_income_with_lenght(company_code)
    balance_sheet = retrieve_balance_with_lenght(company_code)

    bs_len = balance_sheet['len']
    bs_numbers = balance_sheet['balance_sheet']

    is_len = income_statement['len']
    is_numbers = income_statement['income_statement']
    

    is_numbers.index = is_numbers.index.astype(int)
    bs_numbers.index = bs_numbers.index.astype(int)

    if bs_len != is_len:
        print("Data is inconsistent")
        return {"error": "Data inconsistency between balance sheet and income statement lengths"}

    # Determine available years for prediction
    years = bs_numbers.index
    predictions_list = []
    
    for i in range(5, bs_len):
        historical_years = years[i-5:i]
        if not historical_years.isin(is_numbers.index).all():
            print(f"Missing data for years: {historical_years[~historical_years.isin(is_numbers.index)]}")
            continue

        
        # Create the prompt
        historical_income = is_numbers.loc[historical_years]
        historical_balance = bs_numbers.loc[historical_years]
        prompt = create_prompt(historical_income, historical_balance)
        
        # Use OpenAI API to analyze the financials
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            temperature=0,
            top_p=1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )
        
        resp = response.choices[0].message.content.strip()
        cleaned_json_string = resp.strip('```json\n')

        # Convert the cleaned string to a JSON object
        try:
            json_obj = json.loads(cleaned_json_string)
            predictions_list.append(json_obj)
            #print year processed
            print(f"Processed year: {json_obj['Year']}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
    
    # Convert the list of JSON objects to a DataFrame
    if predictions_list:
        predictions = pd.DataFrame(predictions_list)
    
    return predictions

def get_predictions_ppxt(company_code):
    # Fetch the data
    url = "https://api.perplexity.ai/chat/completions"

    income_statement = retrieve_income_with_lenght(company_code)
    balance_sheet = retrieve_balance_with_lenght(company_code)

    bs_len = balance_sheet['len']
    bs_numbers = balance_sheet['balance_sheet']

    is_len = income_statement['len']
    is_numbers = income_statement['income_statement']
    is_numbers.index = is_numbers.index.astype(int)
    bs_numbers.index = bs_numbers.index.astype(int)

    # Ensure data consistency
    if bs_len != is_len:
        print("Data is inconsistent")
        return {"error": "Data inconsistency between balance sheet and income statement lengths"}

    # Determine available years for prediction
    years = bs_numbers.index
    predictions = []

    for i in range(5, bs_len):
        historical_years = years[i-5:i]
        # Ensure all required years are present in the index
        if not historical_years.isin(is_numbers.index).all():
            print(f"Missing data for years: {historical_years[~historical_years.isin(is_numbers.index)]}")
            continue

        prediction_year = years[i]
        
        # Create the prompt
        historical_income = is_numbers.loc[historical_years]
        historical_balance = bs_numbers.loc[historical_years]
        prompt = create_prompt(historical_income, historical_balance)
        
        payload = {
            "model": "mixtral-8x7b-instruct",
            "messages": [
                {"role": "system", "content": "you are a financial analyst"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0,
            "top_p": 1,
            "return_citations": False,
            "return_images": False,
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Bearer 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"
        }

        response = requests.post(url, json=payload, headers=headers)
        predictions.append(response.json())

    return predictions

def process_response_ppxt(answer):
    def clean_json(content):
        content = content.replace('\n', '').replace('\\"', '"')
        return json.loads(content)

    extracted_data = []
    for item in answer:
        content = item['choices'][0]['message']['content']
        try:
            parsed_content = clean_json(content)
            extracted_data.append(parsed_content)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

    # Create DataFrame from extracted data
    df = pd.DataFrame(extracted_data)

    # Format the DataFrame as desired
    df['row_format'] = df.apply(lambda x: f"{x.get('earnings direction', '')}|{x.get('magnitude', '')}|{x.get('confidence score', '')}|{x.get('summary of rationale', '')}", axis=1)
    #drop row_format column
    df = df.drop(columns=['row_format'], axis=1)
    # add column with company name
    return df

if __name__ == "__main__":
    # Example usage
    # Predict earnings for a company with a given cvm_code
    # asks for cvm code as input 
    cvm_code = input("Enter the cvm_code:  ")
    #asks for output type as input explaining the output type
    #fetches the predictions

    #ppxt calls
    ppxt = get_predictions_ppxt(cvm_code)
    ppxt_pretty = process_response_ppxt(ppxt)



    predictions = get_predictions(cvm_code)
    #prints the predictions
    print(predictions)
    

## improve prompt and create simple prompt 
## make the predictions log better