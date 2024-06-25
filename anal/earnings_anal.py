import pandas as pd
from openaicall import get_predictions
from fetcherv6 import net_income_direction, get_company_name
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_earnings(cvm_code):
    start_time = time.time()

    # Get predictions
    pred_start_time = time.time()
    predictions = get_predictions(cvm_code)
    pred_end_time = time.time()
    logging.info(f"Time elapsed for get_predictions: {pred_end_time - pred_start_time:.2f} seconds")
    logging.info(f"Predictions shape: {predictions.shape}")
    logging.info(f"Predictions columns: {predictions.columns}")
    logging.info(f"Predictions head:\n{predictions.head()}")
    
    # Get actual earnings direction
    earnings_start_time = time.time()
    actual_earnings = net_income_direction(cvm_code)
    earnings_end_time = time.time()
    logging.info(f"Time elapsed for net_income_direction: {earnings_end_time - earnings_start_time:.2f} seconds")
    logging.info(f"Actual earnings shape: {actual_earnings.shape}")
    logging.info(f"Actual earnings head:\n{actual_earnings.head()}")
    
    # Get company name
    name_start_time = time.time()
    company_name = get_company_name(cvm_code)
    name_end_time = time.time()
    logging.info(f"Time elapsed for get_company_name: {name_end_time - name_start_time:.2f} seconds")
    logging.info(f"Company name: {company_name}")
    
    # Prepare data for CSV
    csv_data = []
    process_start_time = time.time()
    for index, row in predictions.iterrows():
        year = row['Year']
        logging.info(f"Processing year: {year}")
        
        try:
            year = int(year)
        except ValueError:
            logging.warning(f"Invalid year value: {year}")
            continue
        
        if year in actual_earnings.index:
            actual_direction = 'increase' if actual_earnings[year] > 0 else 'decrease'
            csv_data.append({
                'Company Name': company_name,
                'Year': year,
                'Actual Earnings Direction': actual_direction,
                'Predicted Earnings Direction': row['earnings direction'],
                'Magnitude': row['magnitude'],
                'Confidence Score': row['confidence score'],
                'Summary of Rationale': row['summary of rationale']
            })
        else:
            logging.warning(f"Year {year} not found in actual earnings data. Skipping.")
    
    process_end_time = time.time()
    logging.info(f"Time elapsed for data processing: {process_end_time - process_start_time:.2f} seconds")

    # Create DataFrame
    df = pd.DataFrame(csv_data)
    logging.info(f"Final DataFrame shape: {df.shape}")
    logging.info(f"Final DataFrame head:\n{df.head()}")
    
    if df.empty:
        logging.warning("No matching years found between predictions and actual earnings.")
    
    # Save to CSV
    csv_start_time = time.time()
    csv_filename = f"{company_name.replace(' ', '_')}_earnings_analysis.csv"
    df.to_csv(csv_filename, index=False)
    csv_end_time = time.time()
    logging.info(f"Time elapsed for saving CSV: {csv_end_time - csv_start_time:.2f} seconds")
    logging.info(f"Analysis saved to {csv_filename}")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time elapsed: {total_time:.2f} seconds")

    return df

if __name__ == "__main__":
    cvm_code = input("Enter the CVM code: ")
    result_df = analyze_earnings(cvm_code)
    print(result_df)