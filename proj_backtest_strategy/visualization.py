# visualization.py

import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import numpy as np
import matplotlib.ticker as ticker
import logging

# Configure logging at the top of your script or module
logging.basicConfig(filename='plot_generation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def plot_cumulative_returns(portfolio_df, symbol, strategy, output_dir):
    """
    Plot total portfolio value over time with buy and sell signals, and stock price on secondary axis.
    """
    # Check for empty or invalid data
    if portfolio_df.empty:
        message = f"Warning: portfolio_df is empty for symbol {symbol}. No data to plot."
        print(message)
        logging.warning(message)
        return

    if 'total_value' not in portfolio_df.columns or portfolio_df['total_value'].isnull().all():
        message = f"Warning: 'total_value' column is missing or all NaN for symbol {symbol}."
        print(message)
        logging.warning(message)
        return

    if 'close_price' not in portfolio_df.columns or portfolio_df['close_price'].isnull().all():
        message = f"Warning: 'close_price' column is missing or all NaN for symbol {symbol}."
        print(message)
        logging.warning(message)
        return

    if 'action' not in portfolio_df.columns or portfolio_df['action'].isnull().all():
        message = f"No trades executed for symbol {symbol}. Skipping plot."
        print(message)
        logging.info(message)
        return

    try:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('white')  # Ensure background is white

        # Plot total portfolio value on primary y-axis
        ax1.plot(portfolio_df.index, portfolio_df['total_value'], label='Total Portfolio Value', color='blue')

        # Extract buy and sell signals
        buy_signals = portfolio_df[portfolio_df['action'] == 'buy']
        sell_signals = portfolio_df[portfolio_df['action'] == 'sell']

        # Ensure sell signals are not empty
        if sell_signals.empty:
            print(f"No sell signals to plot for symbol {symbol}.")
        else:
            print(f"Plotting {len(sell_signals)} sell signals for symbol {symbol}.")

        # Plot buy signals on primary axis
        ax1.scatter(buy_signals.index, buy_signals['total_value'],
                    marker='^', color='green', label='Buy Signal', s=100, zorder=5)

        # Plot sell signals on primary axis
        ax1.scatter(sell_signals.index, sell_signals['total_value'],
                    marker='v', color='red', label='Sell Signal', s=100, zorder=5)

        # Formatting for primary y-axis
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Portfolio Value')
        ax1.grid(True)

        # Format y-axis with commas
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        # Create secondary y-axis
        ax2 = ax1.twinx()

        # Plot stock price on secondary y-axis
        ax2.plot(portfolio_df.index, portfolio_df['close_price'], label='Stock Price', color='orange', alpha=0.6)

        # Formatting for secondary y-axis
        ax2.set_ylabel('Stock Price')

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        # Improve date formatting on the x-axis
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        plt.title(f'{symbol} Portfolio Performance using {strategy.replace("_", " ").title()} Strategy')
        plt.tight_layout()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot as PNG
        plot_filename = f"{symbol}_{strategy}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        message = f"Error plotting data for symbol {symbol}: {e}"
        print(message)
        logging.error(message)
        if os.path.exists(plot_path):
            os.remove(plot_path)


def compile_charts_to_pdf(charts_dir, output_pdf_path):
    """
    Compile all PNG charts in a directory into a single PDF file.

    Parameters:
    - charts_dir (str): Directory containing PNG chart files.
    - output_pdf_path (str): Path to save the compiled PDF.
    """
    # Collect all PNG files in the charts directory
    png_files = [os.path.join(charts_dir, f) for f in os.listdir(charts_dir) if f.lower().endswith('.png')]

    # Sort the files for organized PDF
    png_files.sort()

    # Initialize PdfPages object
    with PdfPages(output_pdf_path) as pdf:
        for png_file in tqdm(png_files, desc="Compiling charts into PDF"):
            try:
                img = plt.imread(png_file)
                plt.figure(figsize=(14, 7))
                plt.imshow(img)
                plt.axis('off')  # Hide axes
                pdf.savefig()  # Saves the current figure into a pdf page
                plt.close()
            except Exception as e:
                print(f"Error adding {png_file} to PDF: {e}")
