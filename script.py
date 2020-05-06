import optim
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 83)
plt.style.use("ggplot")

symbols = ["MSFT", "AMZN", "AAPL", "GOOG", "FB"]

start = datetime(2019, 1, 1)
end = datetime(2019, 5, 1)

stock_data = web.get_data_yahoo(symbols, start, end)

stock_prices = stock_data["Adj Close"]
stock_daily_returns = stock_prices.pct_change()

fig1 = plt.figure(figsize=(15, 9))
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)
ax1.plot(stock_prices)
ax1.set_title("Tech Stocks Closing Prices")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price")
ax2.plot(stock_daily_returns)
ax2.set_xlabel("Date")
ax2.set_ylabel("Rate of Return")
ax2.set_title("Tech Stocks Return Rate")
plt.tight_layout()


fig2 = plt.figure(figsize=(15, 9))
ax3 = fig2.add_subplot(511)
ax4 = fig2.add_subplot(512)
ax5 = fig2.add_subplot(513)
ax6 = fig2.add_subplot(514)
ax7 = fig2.add_subplot(515)
ax3.plot(stock_data['Adj Close']['AMZN'].pct_change())
ax3.set_title("Amazon")
ax4.plot(stock_data['Adj Close']['AAPL'].pct_change())
ax4.set_title("Apple")
ax5.plot(stock_data['Adj Close']['FB'].pct_change())
ax5.set_title("Facebook")
ax6.plot(stock_data['Adj Close']['MSFT'].pct_change())
ax6.set_title("Microsoft")
ax7.plot(stock_data['Adj Close']['GOOG'].pct_change())
ax7.set_title("Google")
plt.tight_layout()
fig2.subplots_adjust(hspace=0.5)

plt.show()
