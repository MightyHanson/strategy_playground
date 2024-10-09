# strategy_playground

This branch will contain the strategies created and tested through routine observing.

## Project: Yield Curve Construction, Bootstrapping, and Analysis

### Overview
This project involves constructing and analyzing yield curves using historical and updated yield data from the Federal Reserve Economic Data (FRED). The key steps include bootstrapping the yield curve with continuous compounding to derive zero-coupon rates, interpolating the curve to obtain yields for additional maturities, and optimizing the yield curve to match market data more closely. The project generates visualizations to compare the interpolated yield curve with the optimized curve, helping to illustrate the effectiveness of the calibration process.

## Project: Interest Rate Modeling

### Overview
This project implements and compares different interest rate models, specifically the Vasicek, Cox-Ingersoll-Ross (CIR), and Hull-White models. It fetches historical Treasury yield data from the Federal Reserve Economic Data (FRED) API, calibrates the models to the data, and simulates interest rate paths. The results are then visualized and saved to an Excel file for further analysis.
