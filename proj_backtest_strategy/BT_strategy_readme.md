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


