# Fetcherv6.py
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Supabase credentials
SUPABASE_USER = os.getenv("SUPABASE_USER")
SUPABASE_PASSWORD = os.getenv("SUPABASE_PASSWORD")
SUPABASE_HOST = os.getenv("SUPABASE_HOST")
SUPABASE_PORT = os.getenv("SUPABASE_PORT")
SUPABASE_DBNAME = os.getenv("SUPABASE_DBNAME")

# Check if all environment variables are set
if not all([SUPABASE_USER, SUPABASE_PASSWORD, SUPABASE_HOST, SUPABASE_PORT, SUPABASE_DBNAME]):
    logging.error("One or more environment variables are missing.")
    raise EnvironmentError("Missing environment variables")

# Construct database URL
database_url = f"postgresql://{SUPABASE_USER}:{SUPABASE_PASSWORD}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DBNAME}"
logging.info(f"Connecting to database with URL: {database_url}")

# Create engine with exception handling
try:
    engine = create_engine(database_url)
except SQLAlchemyError as e:
    logging.error(f"Error connecting to the database: {e}")
    raise

def fetch_data(query, params=None):
    # Convert numpy types to native Python types
    if params:
        params = {key: int(value) if isinstance(value, (np.integer, np.int64)) else value for key, value in params.items()}
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except SQLAlchemyError as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_yearly_data(df, date_column):
    if not df.empty:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        numeric_df = df.select_dtypes(include=['number'])
        yearly_df = numeric_df.resample('YE').sum()  # Use 'A' for annual resampling
        yearly_df.index = yearly_df.index.year
        yearly_df = yearly_df.transpose()
        return yearly_df.reset_index()
    return df

def clean_dataframe(df, drop_columns=None, drop_index=None, rename_columns=None):
    if drop_columns:
        missing_columns = [col for col in drop_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Columns {missing_columns} not found in DataFrame")
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
    if drop_index:
        df.drop(index=drop_index, inplace=True, errors='ignore')
    if rename_columns:
        df.rename(columns={old: new for old, new in rename_columns.items() if old in df.columns}, inplace=True)
    return df

def fetch_balance_sheet(cvm_code):
    query = """
        SELECT *
        FROM balance_sheet
        WHERE cvm_code = :cvm_code
        ORDER BY reference_date ASC
    """
    df = fetch_data(query, {'cvm_code': cvm_code})
    if not df.empty:
        df['reference_date'] = pd.to_datetime(df['reference_date'])
        year_end_df = df[df['reference_date'].dt.is_year_end]
        year_end_df = year_end_df.set_index('reference_date').transpose()
        year_end_df.columns = year_end_df.columns.year  # Change columns to year only
        
        year_end_df = clean_dataframe(year_end_df, drop_index=['cvm_code', 'statement_type'])
        
        # Add a row for the result of assets - equity - liabilities
        year_end_df.loc['check'] = year_end_df.loc['assets'] - year_end_df.loc['liabilities']
        year_end_df.reset_index(inplace=True)
        year_end_df.rename(columns={'index': 'acc_entry'}, inplace=True)
        year_end_df.set_index('acc_entry', inplace=True)
    return year_end_df

def fetch_income_statement(cvm_code):
    query = """
        SELECT *
        FROM income_statement
        WHERE cvm_code = :cvm_code
        ORDER BY period_end ASC
    """
    df = fetch_data(query, {'cvm_code': cvm_code})
    yearly_df = process_yearly_data(df, 'period_end')
    
    if not yearly_df.empty:
        yearly_df = clean_dataframe(yearly_df, drop_columns=['period_end'], drop_index='cvm_code')
    yearly_df.rename(columns={'index': 'acc_entry'}, inplace=True)
    yearly_df.set_index('acc_entry', inplace=True)
    return yearly_df

def fetch_cash_flow(cvm_code):
    query = """
        SELECT *
        FROM cash_flow
        WHERE cvm_code = :cvm_code
        ORDER BY period_end ASC
    """
    df = fetch_data(query, {'cvm_code': cvm_code})
    yearly_df = process_yearly_data(df, 'period_end')
    
    if not yearly_df.empty:
        yearly_df.rename(columns={'index': 'acc_entry'}, inplace=True)
        yearly_df.set_index('acc_entry', inplace=True)
        return yearly_df

def fetch_company_data():
    """Fetch all data from the company table."""
    query = """
        SELECT *
        FROM company
    """
    df = fetch_data(query)
    if df.empty:
        logging.warning("No data found in the company table.")
    else:
        logging.info(f"Successfully fetched {len(df)} rows from the company table.")
    df.set_index('cvm_code', inplace=True)
    return df

def get_company_name(cvm_code):
    f = fetch_company_data()
    selected_row = f.loc[int(cvm_code)]
    return selected_row.iloc[0]

def fetch_financials(cvm_code):
    """Fetch balance sheet, income statement, and cash flow data for a given cvm_code."""
    balance_sheet_df = fetch_balance_sheet(cvm_code)
    income_statement_df = fetch_income_statement(cvm_code)
    cash_flow_df = fetch_cash_flow(cvm_code)

    return balance_sheet_df, income_statement_df, cash_flow_df

def fetch_datx_y(cvm_code):
    balance_sheet = fetch_balance_sheet(cvm_code)
    income_statement = fetch_income_statement(cvm_code)

    balance_sheet = balance_sheet.T
    income_statement = income_statement.T
    
    income_statement.columns = [col.lower().replace(" ", "_") for col in income_statement.columns]
    balance_sheet.columns = [col.lower().replace(" ", "_") for col in balance_sheet.columns]
    y_is = len(income_statement.index.astype(int))
    y_bs = len(balance_sheet.index.astype(int))
    # return tuple of dataframes and years
    tpl = {'income_statement': (income_statement, y_is), 'balance_sheet': (balance_sheet, y_bs)}
    return tpl

def retrieve_income_with_lenght(cvm_code):
    income_statement = fetch_income_statement(cvm_code)
    income_statement = income_statement.T
    income_statement.columns = [col.lower().replace(" ", "_") for col in income_statement.columns]
    y_is = len(income_statement.index.astype(int))
    return {'len':(y_is),'income_statement':income_statement}

def retrieve_balance_with_lenght(cvm_code):
    balance_sheet = fetch_balance_sheet(cvm_code)
    balance_sheet = balance_sheet.T
    balance_sheet.columns = [col.lower().replace(" ", "_") for col in balance_sheet.columns]
    y_bs = len(balance_sheet.index.astype(int))
    return {'len':(y_bs),'balance_sheet':balance_sheet}


if __name__ == "__main__":
    # Example usage
    cvm_code = "example_cvm_code"
    balance_sheet_df, income_statement_df, cash_flow_df = fetch_financials(cvm_code)
    print(balance_sheet_df)
    print(income_statement_df)
    print(cash_flow_df)

    # Fetch and print company data
    company_df = fetch_company_data()
    print(company_df)