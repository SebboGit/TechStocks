from optim import return_portfolios, optimal_portfolio
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from datetime import datetime
pd.set_option("display.width", 500)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 83)
plt.style.use("ggplot")

symbols = ["MSFT", "INTC", "AAPL", "AMD", "FB"]

start = datetime(2019, 1, 1)
end = datetime(2019, 8, 1)

stock_data = web.get_data_yahoo(symbols, start, end)

stock_prices = stock_data["Adj Close"]
# daily returns percentage
stock_daily_returns = stock_prices.pct_change()
# daily returns mean
stock_daily_mean = stock_daily_returns.mean()
# daily returns variance
stock_daily_var = stock_daily_returns.var()
# standard deviation
stock_daily_std = stock_daily_returns.std()
# stock correlation
stock_corr = stock_daily_returns.corr()
print(stock_corr)
# covariance matrix
cov_matrix = stock_daily_returns.cov()


# plotting prices and daily returns
fig1 = plt.figure(figsize=(15, 9))
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

for symbol in symbols:
    ax1.plot(stock_prices[symbol], label=symbol)
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price")
ax1.set_title("Tech Stocks Closing Prices")
ax1.legend(loc=2)

for symbol in symbols:
    ax2.plot(stock_daily_returns[symbol], label=symbol)
ax2.set_xlabel("Date")
ax2.set_ylabel("Rate of Return")
ax2.set_title("Tech Stocks Return Rate")
ax2.legend()
plt.tight_layout()

# plotting daily returns by company
fig2 = plt.figure(figsize=(15, 9))
ax3 = fig2.add_subplot(511)
ax4 = fig2.add_subplot(512)
ax5 = fig2.add_subplot(513)
ax6 = fig2.add_subplot(514)
ax7 = fig2.add_subplot(515)
ax3.plot(stock_data['Adj Close']['AMD'].pct_change())
ax3.set_title("AMD Returns")
ax4.plot(stock_data['Adj Close']['AAPL'].pct_change())
ax4.set_title("Apple Returns")
ax5.plot(stock_data['Adj Close']['FB'].pct_change())
ax5.set_title("Facebook Returns")
ax6.plot(stock_data['Adj Close']['MSFT'].pct_change())
ax6.set_title("Microsoft Returns")
ax7.plot(stock_data['Adj Close']['INTC'].pct_change())
ax7.set_title("Google Returns")
plt.tight_layout()
fig2.subplots_adjust(hspace=0.5)

# plotting mean rate of return, variance and standard deviation
bar_heights_mean = []
for key in stock_daily_mean.keys():
    bar_heights_mean.append(stock_daily_mean[key])
bar_heights_var = []
for key in stock_daily_var.keys():
    bar_heights_var.append(stock_daily_var[key])
bar_heights_std = []
for key in stock_daily_std.keys():
    bar_heights_std.append(stock_daily_std[key])

x_pos = np.arange(len(stock_daily_mean.keys()))

fig3 = plt.figure(figsize=(14, 8))
ax8 = fig3.add_subplot(131)
ax9 = fig3.add_subplot(132)
ax10 = fig3.add_subplot(133)
ax8.bar(x_pos, bar_heights_mean, color="#f57e42")
ax8.set_xticks(x_pos)
ax8.set_xticklabels(symbols)
ax8.set_xlabel("Company")
ax8.set_ylabel("Daily mean return")
ax8.set_title("Daily Mean Rate of Return")

ax9.bar(x_pos, bar_heights_var, color="#f56342")
ax9.set_xticks(x_pos)
ax9.set_xticklabels(symbols)
ax9.set_xlabel("Company")
ax9.set_ylabel("Daily variance")
ax9.set_title("Daily Variance")

ax10.bar(x_pos, bar_heights_std, color="#f5b942")
ax10.set_xticks(x_pos)
ax10.set_xticklabels(symbols)
ax10.set_xlabel("Company")
ax10.set_ylabel("Standard Deviation")
ax10.set_title("Standard Deviation")
plt.tight_layout()

random_portfolios = return_portfolios(stock_daily_mean, cov_matrix)
weights, returns, risks = optimal_portfolio(stock_daily_returns[1:])

fig4 = plt.figure(figsize=(9, 6))
ax12 = fig4.add_subplot(111)
ax12.scatter(x=random_portfolios["Volatility"], y=random_portfolios["Returns"], color="#42b9f5")
ax12.plot(risks, returns, "y-o")
ax12.set_xlabel("Volatility")
ax12.set_ylabel("Returns")
ax12.set_title("Effective Frontier")
plt.tight_layout()

plt.show()