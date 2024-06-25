import pandas as pd
import concurrent.futures
from openaicall import get_predictions
from fetcherv6 import net_income_direction, get_company_name, fetch_datx_y
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_year(year_data, company_name, actual_earnings):
    year = year_data['Year']
    logging.info(f"Processing year: {year}")
    
    try:
        year = int(year)
    except ValueError:
        logging.warning(f"Invalid year value: {year}")
        return None
    
    if year in actual_earnings.index:
        actual_direction = 'increase' if actual_earnings[year] > 0 else 'decrease'
        return {
            'Company Name': company_name,
            'Year': year,
            'Actual Earnings Direction': actual_direction,
            'Predicted Earnings Direction': year_data['earnings direction'],
            'Magnitude': year_data['magnitude'],
            'Confidence Score': year_data['confidence score'],
            'Summary of Rationale': year_data['summary of rationale']
        }
    else:
        logging.warning(f"Year {year} not found in actual earnings data. Skipping.")
        return None

def calculate_metrics(df):
    df['Correct'] = df['Actual Earnings Direction'] == df['Predicted Earnings Direction']
    
    df['TP'] = ((df['Actual Earnings Direction'] == 'increase') & (df['Predicted Earnings Direction'] == 'increase')).astype(int)
    df['FP'] = ((df['Actual Earnings Direction'] == 'decrease') & (df['Predicted Earnings Direction'] == 'increase')).astype(int)
    df['TN'] = ((df['Actual Earnings Direction'] == 'decrease') & (df['Predicted Earnings Direction'] == 'decrease')).astype(int)
    df['FN'] = ((df['Actual Earnings Direction'] == 'increase') & (df['Predicted Earnings Direction'] == 'decrease')).astype(int)

    df['Cumulative Accuracy'] = df['Correct'].expanding().mean()
    
    df['Cumulative Precision'] = (df['TP'].expanding().sum() /
                                  (df['TP'].expanding().sum() + 
                                   df['FP'].expanding().sum()))
    
    df['Cumulative Recall'] = (df['TP'].expanding().sum() /
                               (df['TP'].expanding().sum() + 
                                df['FN'].expanding().sum()))
    
    df['Cumulative F1-Score'] = (2 * df['Cumulative Precision'] * df['Cumulative Recall'] / 
                                 (df['Cumulative Precision'] + df['Cumulative Recall']))

    return df

def analyze_earnings(cvm_code):
    start_time = time.time()

    # Fetch all data at once
    data_start_time = time.time()
    financials = fetch_datx_y(cvm_code)
    company_name = get_company_name(cvm_code)
    data_end_time = time.time()
    logging.info(f"Time elapsed for fetching all data: {data_end_time - data_start_time:.2f} seconds")

    # Get predictions (parallelized)
    pred_start_time = time.time()
    predictions = get_predictions(cvm_code)
    pred_end_time = time.time()
    logging.info(f"Time elapsed for get_predictions: {pred_end_time - pred_start_time:.2f} seconds")
    logging.info(f"Predictions shape: {predictions.shape}")
    
    # Calculate actual earnings direction
    earnings_start_time = time.time()
    actual_earnings = net_income_direction(cvm_code)
    earnings_end_time = time.time()
    logging.info(f"Time elapsed for net_income_direction: {earnings_end_time - earnings_start_time:.2f} seconds")
    logging.info(f"Actual earnings shape: {actual_earnings.shape}")

    # Process data in parallel
    process_start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_year, row, company_name, actual_earnings) 
                   for _, row in predictions.iterrows()]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    csv_data = [r for r in results if r is not None]
    process_end_time = time.time()
    logging.info(f"Time elapsed for data processing: {process_end_time - process_start_time:.2f} seconds")

    # Create DataFrame and calculate metrics
    df = pd.DataFrame(csv_data)
    if not df.empty:
        df = df.sort_values('Year')
        df = calculate_metrics(df)
    
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
