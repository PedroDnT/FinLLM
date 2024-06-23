import pandas as pd
from openaicall import get_predictions
from fetcherv6 import fetch_income_statement, fetch_balance_sheet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# retrieve list of cvm codes, checks for column available_years >= 6. return list of valid cvm codes
import pandas as pd
import os
file = 'extended_company_data.csv'

def check_data_availability(file):
    df = pd.read_csv(file)
    valid_cvm_codes = df[df['available_years'] >= 6]['cvm_code'].tolist()
    return valid_cvm_codes
    
def get_predictions_for_valid_cvm_codes(file):
    valid_cvm_codes = check_data_availability(file)
    predictions = {}
    for cvm_code in valid_cvm_codes:
        logging.info(f"Fetching predictions for cvm_code: {cvm_code}")
        predictions[cvm_code] = get_predictions(cvm_code)
    return predictions

if __name__ == "__main__":
    predictions = get_predictions_for_valid_cvm_codes('extended_company_data.csv')
    # log and save as csv the cvm_code, trade_name, prediction_year, prediction
    # Save the predictions to a CSV file
    logging.info("Saving predictions to CSV file")

    predictions_df = pd.DataFrame(columns=["cvm_code", "trade_name", "prediction_year", "prediction"])
    for cvm_code, prediction_list in predictions.items():
        for prediction in prediction_list:
            predictions_df = predictions_df.append({
                "cvm_code": cvm_code,
                "trade_name": prediction[0],
                "prediction_year": prediction[1],
                "prediction": prediction[2]
            }, ignore_index=True)
    predictions_df.to_csv("predictions.csv", index=False)

    