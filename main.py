from matplotlib.dates import YearLocator
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def fetch_stock_data(symbol, start_date, end_date):
    # Fetch historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    return stock_data["Adj Close"]


def calculate_linear_regression(stock_prices):
    # Reshape the data
    X = np.arange(len(stock_prices)).reshape(-1, 1)
    y = stock_prices.values.reshape(-1, 1)

    # Perform linear regression
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = y - y_pred

    return y_pred.flatten(), residuals.flatten()


def plot_stock_data_with_regression(symbol, start_date, end_date):
    # Fetch stock data
    stock_prices = fetch_stock_data(symbol, start_date, end_date)

    if stock_prices.empty:
        print(
            f"Unable to fetch data for {symbol}. Please check the stock symbol or date range."
        )
        return

    # Calculate linear regression
    y_pred, residuals = calculate_linear_regression(stock_prices)

    # Calculate standard deviation of residuals
    std_dev_residuals = np.std(residuals)

    # Plot the stock prices and linear regression line
    plt.figure(figsize=(12, 8))
    plt.plot(stock_prices.index, stock_prices, label="Stock Price")
    plt.plot(
        stock_prices.index,
        y_pred,
        label="Linear Regression",
        linestyle="-",
        color="black",
    )

    # Plot the regression channel
    plt.fill_between(
        stock_prices.index,
        y_pred,
        y_pred + std_dev_residuals,
        color="red",
        alpha=0.2,
    )
    plt.fill_between(
        stock_prices.index,
        y_pred + std_dev_residuals,
        y_pred + 2 * std_dev_residuals,
        color="red",
        alpha=0.4,
    )
    plt.fill_between(
        stock_prices.index,
        y_pred,
        y_pred - std_dev_residuals,
        color="green",
        alpha=0.2,
    )
    plt.fill_between(
        stock_prices.index,
        y_pred - std_dev_residuals,
        y_pred - 2 * std_dev_residuals,
        color="green",
        alpha=0.4,
    )

    # Set plot labels and title
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"Linear Regression channel for {symbol}")

    # Set x-axis ticks to one year intervals
    plt.gca().xaxis.set_major_locator(YearLocator())

    # Display legend
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Take stock symbol, start date, and end date as input
    symbol = input("Enter the stock symbol: ").upper()
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Plot stock data with regression
    plot_stock_data_with_regression(symbol, start_date, end_date)
