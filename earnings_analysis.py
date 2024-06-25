import pandas as pd
from openaicall import get_predictions
from fetcherv6 import net_income_direction, get_company_name
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_earnings(cvm_code):
    # Get predictions
    predictions = get_predictions(cvm_code)

    
    # Get actual earnings direction
    actual_earnings = net_income_direction(cvm_code)
  
    
    # Get company name
    company_name = get_company_name(cvm_code)
    logging.info(f"Company name: {company_name}")
    
    # Prepare data for CSV
    csv_data = []
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
    
    # Create DataFrame
    df = pd.DataFrame(csv_data)
    logging.info(f"Final DataFrame shape: {df.shape}")
    logging.info(f"Final DataFrame head:\n{df.head()}")
    
    if df.empty:
        logging.warning("No matching years found between predictions and actual earnings.")
    
    # Save to CSV
    csv_filename = f"{company_name.replace(' ', '_')}_earnings_analysis.csv"
    df.to_csv(csv_filename, index=False)
    logging.info(f"Analysis saved to {csv_filename}")

    return df

if __name__ == "__main__":
    cvm_code = input("Enter the CVM code: ")
    result_df = analyze_earnings(cvm_code)
    print(result_df)