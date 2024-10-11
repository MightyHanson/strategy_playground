# Backtest Strategy Project
# *Part 1 Data Manager*
## Overview

The **Data Manager** is the foundational step in building a robust backtest strategy project. It is designed to collect and organize financial data from **Financial Modeling Prep (FMP)** and **Yahoo Finance**, creating a comprehensive dataset tailored for strategic analysis and backtesting.

## Features

- **Stock Selection:** Gathers stock symbols with a market capitalization exceeding **$500M** to ensure liquidity and market presence.
- **ETF Filtering:** Extracts ETF symbols that maintain a minimum daily trading volume of **200,000** to guarantee trading activity.
- **Historical Data Collection:** Compiles historical market data spanning the last **30 years**, providing a rich dataset for in-depth analysis.
- **Benchmark Integration:** Incorporates benchmark indices like the **VIX** to serve as indicators for strategy signals.
- **Data Storage:** Organizes the collected data into structured formats, including Excel files and a SQLite database, facilitating easy access and manipulation.

## Data Included

- **Stocks:** Active stock symbols with a market cap greater than $500M.
- **ETFs:** ETF symbols filtered based on a minimum daily trading volume of 200,000.
- **Benchmark Indices:** Includes indices such as `^VIX` to provide market volatility signals.
- **Historical Market Data:** Comprehensive historical price data (Open, High, Low, Close, Volume) from October 9, 1994, to October 9, 2024. [To be updated]

# *Part 2 Backtest Platform - Buy the Dip Strategy Development*
## Overview

The **Backtest Platform** builds on the dataset created by the **Data Manager** and focuses on implementing and backtesting the Buy the Dip strategy. The strategy uses benchmark signals to generate buy and sell orders, evaluating performance over historical data.

## Key Features
**Strategy Implementation**: Implements the Buy the Dip strategy based on user-configurable parameters, such as buy threshold and decay days.

## Trading Logic:
- **Buy Logic**: 10% of total initial capital is used for buying when a signal is triggered.
- **Sell Logic**: 30% of the current portfolio holdings are sold when a sell signal is triggered, adding proceeds to cash.
- **Backtesting**: Simulates trades based on historical data and calculates performance metrics such as cumulative return, Sharpe ratio, max drawdown, and win rate.
- **Optimization**: Optimizes strategy parameters using grid search, with parallel processing to speed up execution.
- **Performance Tracking**: Logs all trades, portfolio performance, and signals, generating charts and a PDF report.

## Backtesting Details
**Buy the Dip Strategy**: A strategy that identifies opportunities to buy when the market dips and sell when the price increases after certain conditions are met.
- **Buy Signal**: Triggered when a benchmark like VIX increases by a certain threshold (e.g., 35%) in a single day.
- **Sell Signal**: Triggered after a set number of consecutive down days (e.g., 14 days) in the benchmark.

## Enhancements
- **Missing Plots Tracking**: Introduced logging to track symbols for which charts were not generated, along with the reasons for missing plots.
- **Sell Signals**: Fixed an issue where sell signals were not being triggered or plotted correctly. Sell signals are now included in the trading logic and displayed in the charts.
- **Multi-Core Optimization**: Leveraged Python's ProcessPoolExecutor to speed up optimization by running strategy backtests in parallel on multiple CPU cores.
- **Plotting Updates**: Added stock prices and buy/sell signals to dual-axis charts for easier visualization of performance.

## Trading Logic Clarification
- **Buying**: Each time a buy signal is triggered, 10% of the initial capital is invested in the stock, regardless of how much cash remains.
- **Selling**: Each time a sell signal is triggered, 30% of the total shares held in the portfolio are sold, and the proceeds are added to the available cash.
## How It Works
- **Data Loading**: The platform loads historical price data from the SQLite database.
- **Signal Generation**: Based on the VIX and other benchmarks, buy and sell signals are generated.
- **Backtest Execution**: The strategy is applied over historical data, and trades are executed based on buy and sell signals.
- **Performance Metrics**: Calculates key performance metrics like cumulative return, Sharpe ratio, max drawdown, and win rate.
- **Optimization**: A grid search is performed to optimize strategy parameters, running in parallel for improved performance.
- **Visualization**: Generates charts showing stock prices, buy/sell signals, and cumulative returns.
## Results Output
- **Trade Log**: All executed trades are logged for review.
- **Performance Metrics**: Key metrics such as Sharpe ratio and max drawdown are calculated and saved.
- **Charts**: Visualization of stock prices and portfolio performance over time, with buy/sell signals.
- **PDF Report**: Compiled charts and results are output as a PDF report for easy review.
