import os
import pandas as pd
import concurrent.futures
from fetcherv6 import fetch_company_data, fetch_balance_sheet, fetch_income_statement
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_all_financial_data(cvm_codes):
    financial_data = {}
    
    def fetch_data(cvm_code):
        try:
            bs = fetch_balance_sheet(cvm_code)
            is_ = fetch_income_statement(cvm_code)
            return cvm_code, (bs, is_)
        except Exception as e:
            logging.error(f"Error fetching data for cvm_code {cvm_code}: {str(e)}")
            return cvm_code, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(fetch_data, cvm_codes)
    
    for cvm_code, data in results:
        if data is not None:
            financial_data[cvm_code] = data
    
    return financial_data

def extract_years(columns):
    years = {int(col) for col in columns if isinstance(col, int) or (isinstance(col, str) and col.isdigit())}
    return years


def calculate_available_years(financial_data):
    try:
        bs, is_ = financial_data
        if bs is None or is_ is None:
            return 0
        bs_years = extract_years(bs.columns)
        is_years = extract_years(is_.columns)
        available_years = len(bs_years & is_years)
    except Exception as e:
        logging.error(f"Error calculating available years: {str(e)}")
        available_years = 0
    
    return available_years

def enrich_company_data():
    companies = fetch_company_data()
    cvm_codes = companies['cvm_code'].unique()
    
    financial_data = fetch_all_financial_data(cvm_codes)
    companies['available_years'] = companies['cvm_code'].apply(
        lambda x: calculate_available_years(financial_data.get(x, (None, None)))
    )
    
    return companies

def save_enriched_data(companies, filepath):
    if os.path.exists(filepath):
        existing_data = pd.read_csv(filepath)
        if not companies.equals(existing_data):
            combined_data = pd.concat([existing_data, companies]).drop_duplicates().reset_index(drop=True)
            combined_data.to_csv(filepath, index=False)
            logging.info("CSV file updated with new data.")
        else:
            logging.info("No changes detected. CSV file not updated.")
    else:
        companies.to_csv(filepath, index=False)
        logging.info("CSV file created with new data.")

if __name__ == "__main__":
    enriched_companies = enrich_company_data()
    save_enriched_data(enriched_companies, 'data/extended_company_data.csv')
    print(enriched_companies.head())
