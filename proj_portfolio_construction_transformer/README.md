# Portfolio Construction Applied Transformer Project 
## Overview
This project constructs and backtests a portfolio using a *transformer-based* machine learning model. The pipeline includes data preprocessing, model training, generating return predictions, optimizing the portfolio, and evaluating portfolio performance. The project utilizes asset price data from various symbols to perform the following:

- **Data preprocessing**: Clean and prepare historical data for multiple symbols.
- **Model training**: Train a transformer time series model to predict asset prices.
- **Portfolio optimization**: Use the predicted returns to optimize the portfolio using covariance matrices and risk-return trade-offs.
- **Backtesting**: Evaluate the portfolio's performance over time with key metrics like Sharpe ratio, cumulative returns, and drawdowns.