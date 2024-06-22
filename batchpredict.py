import pandas as pd
from openaicall import get_predictions
from fetcherv6 import fetch_income_statement, fetch_balance_sheet
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_data_availability(cvm_code):
    income_statement = fetch_income_statement(cvm_code)
    balance_sheet = fetch_balance_sheet(cvm_code)
    
    if income_statement.empty or balance_sheet.empty:
        return False
    
    years = [col for col in income_statement.columns if col.isdigit()]
    if len(years) < 6:  # We need at least 6 years of data (T-5 to T)
        return False
    
    return True

def get_actual_earnings_change(income_statement, year):
    year = str(year)
    prev_year = str(int(year) - 1)
    
    if year in income_statement.columns and prev_year in income_statement.columns:
        earnings_t = income_statement[year]['net_income']
        earnings_t_minus_1 = income_statement[prev_year]['net_income']
        return 'increase' if earnings_t > earnings_t_minus_1 else 'decrease'
    return None

def calculate_accuracy(predictions, actuals):
    correct = sum(p == a for p, a in zip(predictions, actuals))
    return correct / len(predictions) if predictions else 0

def calculate_f1_score(predictions, actuals):
    tp = sum((p == 'increase' and a == 'increase') for p, a in zip(predictions, actuals))
    fp = sum((p == 'increase' and a == 'decrease') for p, a in zip(predictions, actuals))
    fn = sum((p == 'decrease' and a == 'increase') for p, a in zip(predictions, actuals))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def process_cvm_codes(cvm_codes_file, output_file):
    with open(cvm_codes_file, 'r') as f:
        cvm_codes = [line.strip() for line in f]

    valid_cvm_codes = [code for code in cvm_codes if check_data_availability(code)]
    logging.info(f"Found {len(valid_cvm_codes)} out of {len(cvm_codes)} CVM codes with sufficient data")

    all_predictions = []
    all_actuals = []

    for cvm_code in valid_cvm_codes:
        income_statement = fetch_income_statement(cvm_code)
        predictions = get_predictions(cvm_code, output_type="B")
        for year, prediction in predictions:
            actual = get_actual_earnings_change(income_statement, year+1)
            if actual:
                all_predictions.append({
                    'cvm_code': cvm_code,
                    'year': year,
                    'prediction': prediction.split(';')[0].strip(),
                    'actual': actual
                })
                all_actuals.append(actual)

    df_predictions = pd.DataFrame(all_predictions)
    df_predictions.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")

    accuracy = calculate_accuracy(df_predictions['prediction'], df_predictions['actual'])
    f1_score = calculate_f1_score(df_predictions['prediction'], df_predictions['actual'])

    logging.info(f"Total companies processed: {len(valid_cvm_codes)}")
    logging.info(f"Total predictions made: {len(df_predictions)}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1-score: {f1_score:.4f}")

if __name__ == "__main__":
    cvm_codes_file = "cvm_codes.txt"
    output_file = "predictions.csv"
    process_cvm_codes(cvm_codes_file, output_file)