
import pandas as pd
from io import StringIO
import os
import logging
import concurrent.futures
# Mock fetch functions
def fetch_company_data():
    data = '''cvm_code,trade_name,b3_trade_name,b3_issuer_code,b3_segment,b3_sector
    1,CompanyA,TRADEA,CODEA,SegmentA,SectorA
    2,CompanyB,TRADEB,CODEB,SegmentB,SectorB
    3,CompanyC,TRADEC,CODEC,SegmentC,SectorC
    '''
    return pd.read_csv(StringIO(data))

def fetch_balance_sheet(cvm_code):
    data = {
        1: '''index,2019,2020,2021
            assets,100,110,120
            liabilities,50,55,60
            ''',
        2: '''index,2018,2019,2020
            assets,200,210,220
            liabilities,100,105,110
            ''',
        3: ''  # Missing data case
    }
    return pd.read_csv(StringIO(data[cvm_code])) if data[cvm_code] else pd.DataFrame()

def fetch_income_statement(cvm_code):
    data = {
        1: '''index,2020,2021,2022
            revenue,300,310,320
            expenses,150,155,160
            ''',
        2: '',  # Missing data case
        3: '''index,2018,2019,2020
            revenue,400,410,420
            expenses,200,205,210
            '''
    }
    return pd.read_csv(StringIO(data[cvm_code])) if data[cvm_code] else pd.DataFrame()

# Original functions from company_data_enricher.py
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
    years = {int(col) for col in columns if col.isdigit()}
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

# Test the enriched data
def test_enrich_company_data():
    filepath = 'test_data.csv'
    
    # Ensure the test environment is clean
    if os.path.exists(filepath):
        os.remove(filepath)
    
    # Run the enrichment process and save the data
    enriched_companies = enrich_company_data()
    save_enriched_data(enriched_companies, filepath)
    
    # Validate initial save
    saved_data = pd.read_csv(filepath)
    assert not saved_data.empty, "CSV file should not be empty after initial save."
    
    # Validate update logic by running enrichment again and saving
    enriched_companies_new = enrich_company_data()
    save_enriched_data(enriched_companies_new, filepath)
    
    updated_data = pd.read_csv(filepath)
    assert saved_data.equals(updated_data), "CSV file should be the same after re-running with identical data."
    
    # Modify data to test extension/overwrite logic
    modified_data = enriched_companies.copy()
    modified_data.loc[0, 'trade_name'] = "ModifiedCompanyA"
    save_enriched_data(modified_data, filepath)
    
    final_data = pd.read_csv(filepath)
    assert not final_data.equals(saved_data), "CSV file should be updated with modified data."
    assert "ModifiedCompanyA" in final_data['trade_name'].values, "CSV file should reflect the modifications."
    
    print("All test cases passed!")

# Run the tests
test_enrich_company_data()
