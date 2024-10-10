import os
import sys
import requests
import pandas as pd
import sqlite3
from tqdm import tqdm
import csv
import time
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env into the environment

# ================================ Configuration and Constants ================================

FMP_API_KEY = os.getenv("FMP_API_KEY")
BENCHMARK_INDICES = ['^VIX']  # Add more benchmark indices here if needed
ETF_VOLUME_THRESHOLD = 200000  # Minimum average daily volume
HISTORICAL_DATA_START_DATE = '1994-10-09'  # Adjust to get up to 30 years
HISTORICAL_DATA_END_DATE = '2024-10-09'  # Current date or as needed

HISTORICAL_MONITOR_CSV_FILE = 'historical_data_monitor.csv'  # CSV file for monitoring historical data downloads
ETF_MONITOR_CSV_FILE = 'etf_filter_monitor.csv'  # CSV file for monitoring ETF filtering

# ================================ Define Output Directory ================================

def define_output_directory():
    """Define and create the output directory relative to the script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")
    return output_dir

# ================================ Fetch Symbols from Financial Modeling Prep ================================

def fetch_stocks(api_key):
    """Fetch stock symbols with Market Cap > $500M using Financial Modeling Prep API."""
    stocks_url = f'https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=500000000&apikey={api_key}'
    try:
        response = requests.get(stocks_url)
        response.raise_for_status()  # Raise an error for bad status
        stocks_data = response.json()
        stock_symbols = [stock['symbol'] for stock in stocks_data]
        print(f"Fetched {len(stock_symbols)} stocks with Market Cap > $500M")
        return stock_symbols
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return []

def fetch_etfs(api_key):
    """Fetch ETF symbols using Financial Modeling Prep API."""
    etfs_url = f'https://financialmodelingprep.com/api/v3/etf/list?apikey={api_key}'
    try:
        response = requests.get(etfs_url)
        response.raise_for_status()
        etfs_data = response.json()
        etf_symbols = [etf['symbol'] for etf in etfs_data]
        print(f"Fetched {len(etf_symbols)} ETFs")
        return etf_symbols
    except Exception as e:
        print(f"Error fetching ETFs: {e}")
        return []

# ================================ Filter ETFs by Volume using Financial Modeling Prep ================================

def filter_etfs_by_volume(etf_symbols, volume_threshold, output_dir, api_key):
    """Filter ETFs based on trading volume and log the process to a CSV file."""
    filtered_etfs = []
    failed_symbols = []
    monitor_csv_path = os.path.join(output_dir, ETF_MONITOR_CSV_FILE)

    # Initialize the CSV file with headers
    with open(monitor_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Volume', 'Status'])

    for symbol in tqdm(etf_symbols, desc="Filtering ETFs by Volume"):
        try:
            # Fetch ETF details to get average volume or latest volume
            # Using Financial Modeling Prep's quote endpoint
            quote_url = f'https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}'
            quote_response = requests.get(quote_url)
            quote_response.raise_for_status()
            quote_data = quote_response.json()

            if not quote_data:
                print(f"No data found for ETF: {symbol}")
                writer.writerow([symbol, 0, 'No Data'])
                continue

            # Assuming 'volume' is the latest trading volume
            volume = quote_data[0].get('volume', 0)
            status = 'Pass' if volume >= volume_threshold else 'Fail'
            writer.writerow([symbol, volume, status])

            if status == 'Pass':
                filtered_etfs.append(symbol)

        except Exception as e:
            print(f"Error fetching volume for {symbol}: {e}")
            writer.writerow([symbol, 0, 'Error'])
            failed_symbols.append(symbol)

    print(f"Filtered ETFs with Volume >= {volume_threshold}: {len(filtered_etfs)}")
    if failed_symbols:
        print(f"Failed to fetch volume for {len(failed_symbols)} ETFs")
    return filtered_etfs


# ================================ Add Benchmark Indices ================================

def add_benchmark_indices(benchmark_indices):
    """Add benchmark indices to the list."""
    print(f"Adding benchmark indices: {', '.join(benchmark_indices)}")
    return benchmark_indices

# ================================ Save Symbols to Excel ================================

def save_symbols_to_excel(stock_symbols, etf_symbols, benchmark_indices, output_dir):
    """Save stock, ETF, and benchmark index symbols to separate sheets in an Excel file."""
    symbols_excel_path = os.path.join(output_dir, 'symbols.xlsx')
    df_stocks = pd.DataFrame(stock_symbols, columns=['Stock Symbol'])
    df_etfs = pd.DataFrame(etf_symbols, columns=['ETF Symbol'])
    df_benchmarks = pd.DataFrame(benchmark_indices, columns=['Benchmark Symbol'])

    try:
        with pd.ExcelWriter(symbols_excel_path, engine='xlsxwriter') as writer:
            df_stocks.to_excel(writer, sheet_name='Stocks', index=False)
            df_etfs.to_excel(writer, sheet_name='ETFs', index=False)
            df_benchmarks.to_excel(writer, sheet_name='Benchmark Indices', index=False)
        print(f"Saved symbols to {symbols_excel_path}")
    except Exception as e:
        print(f"Error saving symbols to Excel: {e}")


# ================================ Load Symbols from Excel ================================

def load_symbols_from_excel(output_dir):
    """Load symbols from the existing Excel file."""
    symbols_excel_path = os.path.join(output_dir, 'symbols.xlsx')
    if not os.path.exists(symbols_excel_path):
        print(f"Error: {symbols_excel_path} does not exist. Please fetch symbols first.")
        sys.exit(1)

    try:
        xls = pd.ExcelFile(symbols_excel_path)
        stock_symbols = xls.parse('Stocks')['Stock Symbol'].dropna().tolist()
        etf_symbols = xls.parse('ETFs')['ETF Symbol'].dropna().tolist()
        benchmark_symbols = xls.parse('Benchmark Indices')['Benchmark Symbol'].dropna().tolist()
        print(f"Loaded {len(stock_symbols)} stocks, {len(etf_symbols)} ETFs, and {len(benchmark_symbols)} benchmark indices from {symbols_excel_path}")
        return stock_symbols, etf_symbols, benchmark_symbols
    except Exception as e:
        print(f"Error loading symbols from Excel: {e}")
        sys.exit(1)

# ================================ Initialize SQLite Database ================================

def initialize_database(db_path):
    """Initialize the SQLite database and create the historical_data table if it doesn't exist."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if not exists
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
        '''
        cursor.execute(create_table_query)
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON historical_data(symbol)')
        conn.commit()
        print(f"Initialized SQLite database and ensured table 'historical_data' exists.")
        return conn, cursor
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)


# ================================ Download Historical Data using Financial Modeling Prep ================================

def download_historical_data_fmp(symbols, start_date, end_date, conn, output_dir, overwrite=False, api_key=FMP_API_KEY):
    """Download historical data for all symbols using Financial Modeling Prep and save them into the SQLite database."""
    cursor = conn.cursor()
    monitor_csv_path = os.path.join(output_dir, HISTORICAL_MONITOR_CSV_FILE)

    # Initialize the CSV file with headers and keep it open during writing
    with open(monitor_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Status', 'Records Downloaded'])

        for idx, symbol in enumerate(tqdm(symbols, desc="Downloading Historical Data")):
            try:
                print(f"Downloading data for {symbol}")
                # Fetch historical data from FMP
                historical_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}'
                response = requests.get(historical_url)
                response.raise_for_status()
                data = response.json()

                if 'historical' not in data:
                    if "Error Message" in data:
                        status = f"Error: {data['Error Message']}"
                    elif "Note" in data:
                        status = f"Error: {data['Note']}"
                    else:
                        status = "Error: Unexpected response format"
                    print(f"{symbol}: {status}")
                    records = 0
                    writer.writerow([symbol, status, records])
                    continue

                historical_data = data['historical']
                if not historical_data:
                    print(f"No historical data found for {symbol} in the specified date range.")
                    status = 'No Data'
                    records = 0
                    writer.writerow([symbol, status, records])
                    continue

                # Convert historical data to DataFrame
                df = pd.DataFrame(historical_data)
                df['symbol'] = symbol
                df = df[['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']]

                # Handle missing or malformed data
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

                records = len(df)

                if overwrite:
                    # Remove existing records for the symbol
                    cursor.execute('DELETE FROM historical_data WHERE symbol = ?', (symbol,))
                    conn.commit()

                # Insert data into the database
                try:
                    df.to_sql('historical_data', conn, if_exists='append', index=False, method='multi', chunksize=1000)
                    print(f"Saved data for {symbol}: {records} records")
                    status = 'Success'
                except sqlite3.IntegrityError:
                    # Handle unique constraint failed error
                    print(f"Duplicate entry found for {symbol}. Skipping insertion.")
                    status = 'Duplicate Entry'

                writer.writerow([symbol, status, records])

            except Exception as e:
                print(f"Error downloading data for {symbol}: {e}")
                writer.writerow([symbol, f'Error: {e}', 0])

            # Rate limiting code removed as per subscription

    print(f"All historical data downloaded and saved to the database.")


# ================================ Main Execution Flow ================================


def main():
    """Main function to orchestrate the fetching, filtering, and saving of symbols and data."""
    # Step 1: Define Output Directory
    output_dir = define_output_directory()

    # ================================ Fetch Symbols Section ================================

    # Uncomment the following lines if you want to fetch symbols and generate the Excel file
    # Comment them out if you want to skip this section (e.g., during debugging)

    # Step 2: Fetch Stocks
    # stock_symbols = fetch_stocks(FMP_API_KEY)

    # Step 3: Fetch ETFs
    # etf_symbols = fetch_etfs(FMP_API_KEY)

    # Step 4: Filter ETFs by Volume with Monitoring
    # filtered_etf_symbols = filter_etfs_by_volume(etf_symbols, ETF_VOLUME_THRESHOLD, output_dir, FMP_API_KEY)

    # Step 5: Add Benchmark Indices
    # final_benchmark_indices = add_benchmark_indices(BENCHMARK_INDICES)

    # Step 6: Save Symbols to Excel with Separate Sheets
    # save_symbols_to_excel(stock_symbols, filtered_etf_symbols, final_benchmark_indices, output_dir)

    # ================================ Load Symbols from Excel Section ================================

    # Uncomment the following line if you want to load existing symbols from the Excel file
    # This is useful when you have already fetched symbols and just want to download historical data

    # Uncomment the following line to load symbols instead of fetching them again
    stock_symbols, etf_symbols, benchmark_symbols = load_symbols_from_excel(output_dir)

    # ================================ Download Historical Data Section ================================

    # Combine all symbols for historical data download
    all_symbols = stock_symbols + etf_symbols + benchmark_symbols

    # Step 7: Initialize SQLite Database in Output Directory
    db_path = os.path.join(output_dir, 'historical_data.db')
    conn, cursor = initialize_database(db_path)

    # Step 8: Download Historical Data with Progress Monitoring
    download_historical_data_fmp(
        all_symbols,
        HISTORICAL_DATA_START_DATE,
        HISTORICAL_DATA_END_DATE,
        conn,
        output_dir,
        overwrite=False,  # Set to True if you want to overwrite existing data
        api_key=FMP_API_KEY
    )

    # Close the database connection
    cursor.close()
    conn.close()
    print("Closed the SQLite database connection.")

if __name__ == "__main__":
    main()
