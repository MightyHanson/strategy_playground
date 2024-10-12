# backtest.py

import os
import pandas as pd
from data_loader import DataLoader
from strategy import BuyTheDipStrategy
from performance_metrics import PerformanceMetrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm  # For progress bars
from visualization import plot_cumulative_returns, compile_charts_to_pdf
from benchmark import Benchmark  # Added import for Benchmark class
from concurrent.futures import ProcessPoolExecutor, as_completed

def backtest_strategy(symbol, benchmark_symbol, data_loader, start_date, end_date,
                      initial_capital=50000000, buy_percentage=0.1, sell_percentage=0.3,
                      method='soared_then_decay', buy_threshold=None, decay_days=None,
                      window=None, num_std=None):
    """
    Perform backtest for a single symbol using a specific benchmark strategy.

    Parameters:
    - symbol (str): The stock or ETF symbol.
    - benchmark_symbol (str): Benchmark symbol.
    - data_loader (DataLoader): Instance of DataLoader.
    - start_date (str): Start date for backtest (YYYY-MM-DD).
    - end_date (str): End date for backtest (YYYY-MM-DD).
    - initial_capital (float): Initial capital for the account.
    - buy_percentage (float): Percentage of cash to use for buying.
    - sell_percentage (float): Percentage of position to sell.
    - method (str): Strategy method name.
    - buy_threshold (float): Buy threshold percentage (for 'soared_then_decay').
    - decay_days (int): Number of decay days (for 'soared_then_decay').
    - window (int): Window size for moving averages or Bollinger Bands.
    - num_std (float): Number of standard deviations for Bollinger Bands.

    Returns:
    - metrics (dict): Performance metrics for the symbol and benchmark.
    """
    try:
        # Initialize Benchmark with updated parameters
        benchmark = Benchmark(
            symbol=benchmark_symbol,
            data_loader=data_loader,
            start_date=start_date,
            end_date=end_date,
            method=method,
            buy_threshold=buy_threshold,
            decay_days=decay_days,
            window=window,
            num_std=num_std
        )

        # Load historical price data for the symbol
        price_data = data_loader.load_historical_data(symbol, start_date, end_date)
        if price_data is None or price_data.empty:
            print(f"No historical data for symbol: {symbol}")
            return None

        # Align benchmark data with price data dates
        benchmark_data = benchmark.data
        common_dates = price_data.index.intersection(benchmark_data.index)
        if common_dates.empty:
            print(f"No overlapping dates between price data and benchmark data for {symbol}.")
            return None

        price_data = price_data.loc[common_dates]
        benchmark_data = benchmark_data.loc[common_dates]
        signals = benchmark.signals.loc[common_dates]

        # Initialize strategy
        strategy = BuyTheDipStrategy(benchmark, buy_threshold, decay_days, window, num_std)
        # Signals are already generated in benchmark.signals

        # Merge price data with signals
        merged_data = price_data.join(signals[['Buy', 'Sell']], how='left')
        merged_data['Buy'] = merged_data['Buy'].astype(bool).fillna(False)
        merged_data['Sell'] = merged_data['Sell'].astype(bool).fillna(False)

        # Initialize portfolio variables
        cash = initial_capital
        position = 0  # Number of shares held
        portfolio = []
        trade_log = []

        for date, row in merged_data.iterrows():
            close_price = row['close']

            # Buy Signal
            if row['Buy']:
                if cash > 0:
                    amount_to_invest = cash * buy_percentage
                    num_shares_to_buy = amount_to_invest // close_price
                    if num_shares_to_buy > 0:
                        cost = num_shares_to_buy * close_price
                        cash -= cost
                        position += num_shares_to_buy
                        trade_log.append({
                            'date': date,
                            'action': 'buy',
                            'price': close_price,
                            'shares': num_shares_to_buy,
                            'cash': cash,
                            'position': position
                        })
                        print(
                            f"Bought {num_shares_to_buy} shares of {symbol} on {date} at {close_price}, total cost {cost}.")
                    else:
                        print(f"Insufficient funds to buy shares of {symbol} on {date}.")
                else:
                    print(f"No cash available to buy shares of {symbol} on {date}.")
            # Sell Signal
            elif row['Sell']:
                # Debug statement
                # print(f"Sell signal on {date} for symbol {symbol}. Position before sell: {position} shares.")
                if position > 0:
                    num_shares_to_sell = int(position * sell_percentage)
                    if num_shares_to_sell <= 0:
                        num_shares_to_sell = position  # Sell all remaining shares if calculated shares to sell is less than or equal to zero
                    proceeds = num_shares_to_sell * close_price
                    cash += proceeds
                    position -= num_shares_to_sell
                    trade_log.append({
                        'date': date,
                        'action': 'sell',
                        'price': close_price,
                        'shares': num_shares_to_sell,
                        'cash': cash,
                        'position': position
                    })
                    print(
                        f"Sold {num_shares_to_sell} shares of {symbol} on {date} at {close_price}, proceeds {proceeds}.")
                else:
                    print(f"No position to sell for {symbol} on {date}.")

            # Update position value and total portfolio value
            position_value = position * close_price
            total_portfolio_value = cash + position_value
            portfolio.append({
                'date': date,
                'total_value': total_portfolio_value,
                'cash': cash,
                'position_value': position_value,
                'close_price': close_price  # Include the stock price here
            })

        portfolio_df = pd.DataFrame(portfolio)
        portfolio_df.set_index('date', inplace=True)

        # Check for missing close_price values
        if portfolio_df['close_price'].isnull().any():
            print(f"Warning: Missing stock prices in portfolio_df for symbol {symbol}.")
            portfolio_df['close_price'].fillna(method='ffill', inplace=True)

        # Create trade log DataFrame
        trades_df = pd.DataFrame(trade_log)
        if not trades_df.empty:
            trades_df.set_index('date', inplace=True)
        else:
            trades_df = pd.DataFrame(columns=['action', 'price', 'shares', 'cash', 'position'])

        # Merge portfolio with trades
        portfolio_df = portfolio_df.merge(trades_df[['action']], how='left', left_index=True, right_index=True)

        # Calculate performance metrics
        perf_metrics = PerformanceMetrics(portfolio_df)
        metrics = perf_metrics.calculate_metrics()

        # Add additional info
        metrics['Symbol'] = symbol
        metrics['Benchmark'] = benchmark.symbol
        metrics['Initial Capital'] = initial_capital
        metrics['Buy Percentage'] = buy_percentage
        metrics['Sell Percentage'] = sell_percentage
        if buy_threshold is not None:
            metrics['Buy Threshold (%)'] = buy_threshold
        if decay_days is not None:
            metrics['Decay Days'] = decay_days
        if window is not None:
            metrics['Window'] = window
        if num_std is not None:
            metrics['Num Std'] = num_std

        # Generate and save the plot
        charts_dir = os.path.join('backtest_performance', 'charts')
        plot_cumulative_returns(portfolio_df, symbol, benchmark.method, charts_dir)

        return metrics

    except Exception as e:
        print(f"An error occurred while backtesting symbol {symbol} with benchmark {benchmark_symbol}: {e}")
        return None

def worker(symbol, start_date, end_date, initial_capital, buy_percentage, sell_percentage,
           buy_threshold, decay_days):
    """
    Worker function for multiprocessing.
    """
    # Initialize DataLoader inside the worker
    data_loader = DataLoader(dataset_dir='dataset')

    metrics = backtest_strategy(
        symbol=symbol,
        benchmark_symbol='^VIX',  # Adjust as needed
        data_loader=data_loader,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        buy_percentage=buy_percentage,
        sell_percentage=sell_percentage,
        method='soared_then_decay',
        buy_threshold=buy_threshold,
        decay_days=decay_days
    )
    if metrics is not None:
        metrics['Symbol'] = symbol
        metrics['Buy Threshold (%)'] = buy_threshold
        metrics['Decay Days'] = decay_days
    return metrics

def optimize_parameters(symbols, start_date, end_date, optimization_params, initial_capital,
                        buy_percentage, sell_percentage, output_dir):
    """
    Optimize strategy parameters using grid search with multiprocessing.
    """
    results = []
    all_individual_results = []
    param_combinations = list(product(optimization_params['buy_thresholds'], optimization_params['decay_days_list']))
    total_combinations = len(param_combinations)
    print(f"Starting parameter optimization with {total_combinations} combinations.")

    for idx, (buy_threshold, decay_days) in enumerate(param_combinations, 1):
        print(f"Optimizing Combination {idx}/{total_combinations}: Buy Threshold={buy_threshold}%, Decay Days={decay_days}")

        temp_results = []

        worker_args = [
            (symbol, start_date, end_date, initial_capital, buy_percentage, sell_percentage,
             buy_threshold, decay_days) for symbol in symbols
        ]

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(worker, *args): args[0] for args in worker_args}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Param Combo {idx}/{total_combinations}"):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        temp_results.append(data)
                except Exception as exc:
                    print(f"{symbol} generated an exception during optimization: {exc}")

        if temp_results:
            temp_df = pd.DataFrame(temp_results)
            all_individual_results.append(temp_df)

            agg_metrics = {
                'Buy Threshold (%)': buy_threshold,
                'Decay Days': decay_days,
                'Average Cumulative Return': temp_df['Cumulative Return'].mean(),
                'Average Annualized Return': temp_df['Annualized Return'].mean(),
                'Average Sharpe Ratio': temp_df['Sharpe Ratio'].mean(),
                'Average Max Drawdown': temp_df['Max Drawdown'].mean(),
                'Average Win Rate': temp_df['Win Rate'].mean()
            }
            results.append(agg_metrics)
            print(f"Completed optimization for Buy Threshold={buy_threshold}%, Decay Days={decay_days}")
        else:
            print(f"No results for Buy Threshold={buy_threshold}%, Decay Days={decay_days}")

    optimization_results = pd.DataFrame(results)

    # Save individual results
    if all_individual_results:
        all_individual_results_df = pd.concat(all_individual_results, ignore_index=True)
        individual_results_file = os.path.join(output_dir, 'optimization_individual_results.xlsx')
        all_individual_results_df.to_excel(individual_results_file, index=False)
        print(f"Individual optimization results saved to {individual_results_file}")
    else:
        all_individual_results_df = pd.DataFrame()
        print("No individual optimization results to save.")

    return optimization_results, all_individual_results_df


def main():
    """
    Orchestrate the entire backtesting and optimization process.
    """
    # Define backtest period
    start_date = '2018-01-01'
    end_date = '2024-10-11'  # Adjusted to the latest available date

    # Initialize DataLoader
    # global data_loader  # Declare as global for use in other functions
    data_loader = DataLoader(dataset_dir='dataset')

    # Load unique symbols
    unique_symbols = data_loader.load_symbols()
    print(f"Total unique symbols to backtest: {len(unique_symbols)}")

    # Load benchmark configurations
    benchmark_configs = data_loader.load_benchmark_symbols_and_methods()
    print(f"Total benchmarks loaded: {len(benchmark_configs)}")

    if not benchmark_configs:
        print("No valid benchmarks loaded. Exiting backtest.")
        return

    # Ensure output directory exists
    output_dir = 'backtest_performance'
    os.makedirs(output_dir, exist_ok=True)
    individual_output_file = os.path.join(output_dir, 'single_performance.xlsx')
    optimization_output_file = os.path.join(output_dir, 'optimization_results.xlsx')
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)

    # Define adjustable parameters
    # global initial_capital, buy_percentage, sell_percentage  # Declare as global for use in other functions
    initial_capital = 50000000  # or any value you want
    buy_percentage = 0.2        # 20% of available cash
    sell_percentage = 0.4       # 40% of the position

    # Initialize results list for individual backtests
    results = []

    # Individual Backtesting
    print("Starting individual backtests...")

    # Define a worker function for individual backtests
    def individual_worker(symbol):
        symbol_results = []
        for benchmark_config in benchmark_configs:
            benchmark_symbol = benchmark_config['Benchmark Symbol']
            method = benchmark_config.get('Method', 'soared_then_decay')
            # Extract strategy parameters based on method
            method_normalized = method.lower().replace(' ', '_')
            buy_threshold = decay_days = window = num_std = None
            if method_normalized == 'soared_then_decay':
                buy_threshold = benchmark_config.get('Buy Threshold (%)', 20)
                decay_days = benchmark_config.get('Decay Days', 10)
            elif method_normalized in ['ma_20_50', 'ma_50_100']:
                # No additional parameters for moving average strategies
                pass
            elif method_normalized == 'bollinger_bands':
                window = benchmark_config.get('Window', 20)
                num_std = benchmark_config.get('Num Std', 2)
            else:
                print(f"Warning: Method '{method}' for benchmark '{benchmark_symbol}' is not recognized. Skipping.")
                continue

            metrics = backtest_strategy(
                symbol=symbol,
                benchmark_symbol=benchmark_symbol,
                data_loader=data_loader,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                buy_percentage=buy_percentage,
                sell_percentage=sell_percentage,
                method=method,
                buy_threshold=buy_threshold,
                decay_days=decay_days,
                window=window,
                num_std=num_std
            )
            if metrics is not None:
                symbol_results.append(metrics)
        return symbol_results if symbol_results else None

    # Use ThreadPoolExecutor for parallel processing of individual backtests
    with ThreadPoolExecutor(max_workers=100) as executor:
    # with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(individual_worker, symbol): symbol for symbol in unique_symbols}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Individual Backtests"):
            symbol = futures[future]
            try:
                symbol_metrics = future.result()
                if symbol_metrics:
                    results.extend(symbol_metrics)
            except Exception as exc:
                print(f"{symbol} generated an exception during individual backtest: {exc}")

    # Create results DataFrame for individual backtests
    if results:
        results_df = pd.DataFrame(results)
        # Save to Excel
        results_df.to_excel(individual_output_file, index=False)
        print(f"Individual backtests completed. Results saved to {individual_output_file}")
    else:
        print("No backtest results to save for individual backtests.")

    # Parameter Optimization
    optimization_params = {
        'buy_thresholds': [15, 20, 25],  # Example thresholds in percentage
        'decay_days_list': [4,5,7]  # Example decay days
    }

    print("Starting parameter optimization...")

    optimization_results, all_individual_results_df = optimize_parameters(
        symbols=unique_symbols,
        start_date=start_date,
        end_date=end_date,
        optimization_params=optimization_params,
        initial_capital=initial_capital,
        buy_percentage=buy_percentage,
        sell_percentage=sell_percentage,
        output_dir=output_dir
    )

    # Save optimization results
    if not optimization_results.empty:
        optimization_results.to_excel(optimization_output_file, index=False)
        print(f"Parameter optimization completed. Results saved to {optimization_output_file}")
    else:
        print("No optimization results to save.")

    # Generate PDF report compiling all charts
    output_pdf_path = os.path.join(output_dir, 'backtest_report.pdf')
    print(f"Generating PDF report at {output_pdf_path}")
    compile_charts_to_pdf(charts_dir, output_pdf_path, min_file_size_kb=20)
    print("PDF report generation completed.")


if __name__ == "__main__":
    main()
