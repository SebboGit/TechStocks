import random
import cvxopt
from cvxopt import blas, solvers


def return_portfolios(expected_returns, cov_matrix):
    np.random.seed(1)
    portfolio_returns = []
    portfolio_volatility = []
    stock_weights = []

    selected = expected_returns.axes[0]
    num_assets = len(selected)
    num_portfolios = 5000

    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_returns.append(returns)
        portfolio_volatility.append(volatility)
        stock_weights.append(weights)

        pf = {"Returns": portfolio_returns, "Volatility": portfolio_volatility}

    for counter, symbol in enumerate(selected):
        pf[symbol + " Weight"] = [weight[counter] for weight in stock_weights]

    df = pd.DataFrame(pf)
    column_order = ["Returns", "Volatility"] + [stock + " Weight" for stock in selected]
    df = df[column_order]

    return df


def optimal_portfolio(quarter_returns):
    n = quarter_returns.shape[1]
    quarter_returns = np.transpose(quarter_returns.values)

    N = 100
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(quarter_returns))
    pbar = opt.matrix(np.mean(quarter_returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))  # negative n x n identity matrix
    h = opt.matrix(0.0, (n, 1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu * S, -pbar, G, h, A, b)['x'] for mu in mus]

    # CALCULATE RISKS AND RETURNS FOR FRONTIER
    quarter_returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S * x)) for x in portfolios]

    # CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(quarter_returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])

    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), quarter_returns, risks