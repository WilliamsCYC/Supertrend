import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def calculate_metrics(time_return):
    cumulative_return = (time_return + 1).cumprod()
    '''max drawdown'''
    max_return = cumulative_return.cummax()
    drawdown = - (cumulative_return - max_return)
    max_drawdown = drawdown.cummax()
    '''annualized return'''
    time_multiplier = 365 * 24 / 4
    '''this wired code is to avoid python warning for fractional power over negative number'''
    annualized_return = np.sign(cumulative_return[-1]) * np.abs(cumulative_return[-1]) ** (
                        time_multiplier/len(time_return))
    risk_free_rate = 0.02
    length = len(time_return)
    sharpe_ratio = (time_return.mean() * length - length/6/365 * risk_free_rate) / (
                    time_return.std() * np.sqrt(length) + 1e-10)
    calmar = (time_return.mean() * length - length/6/365 * risk_free_rate) / (max_drawdown.max() + 1e-10)
    return cumulative_return, annualized_return, sharpe_ratio, calmar, max_drawdown


def plot_performance(time_return, data, initial_cash, order_size):

    cumulative_return, annualized_return, sharpe_ratio, calmar, max_drawdown = calculate_metrics(time_return)
    cols_names = ['Cumulative returns',
                  'Annualized return',
                  'Sharpe ratio',
                  'Annual volatility',
                  'Calmar ratio',
                  'Max drawdown',
                  ]
    cell_values = [np.round(cumulative_return[-1], 4),
                   np.round(annualized_return, 4),
                   np.round(sharpe_ratio, 4),
                   np.round(time_return.std() * np.sqrt(252), 4),
                   np.round(calmar, 4),
                   np.round(max_drawdown.max(), 4)
                   ]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )
    '''subplot one: metrics table'''
    fig.add_trace(
        go.Table(
            header=dict(
                values=cols_names,
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=cell_values,
                align="left")
        ),
        row=1, col=1
    )
    '''subplot two: cumulative return'''
    fig.add_trace(
        go.Scatter(
            x=cumulative_return.index,
            y=cumulative_return.values,
            mode="lines",
            name='cumulative return',
            marker=dict(color='green'),
        ),
        row=2, col=1
    )
    '''subplot three: max drawdown'''
    fig.add_trace(
        go.Scatter(
            x=max_drawdown.index,
            y=max_drawdown.values,
            mode="lines",
            name='max drawdown',
            marker=dict(color='red')
        ),
        row=3, col=1
    )
    fig.update_layout(
        height=600,
        title_text='Performance',
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=False)
    fig.show()

    '''plot buy and hold performance'''
    df = data.copy()
    btc_return = df.close.pct_change() + 1
    btc_cumulative_return = btc_return.cumprod()
    initial_equity = initial_cash * order_size / 100
    cash = initial_cash * (1 - order_size / 100)
    equity = initial_equity * btc_cumulative_return
    equity[0] = initial_equity
    bh_return = (equity - equity.shift()) / (equity.shift() + cash)
    cumulative_return, annualized_return, sharpe_ratio, calmar, max_drawdown = calculate_metrics(bh_return)
    cols_names = ['Cumulative returns',
                  'Annualized return',
                  'Sharpe ratio',
                  'Annual volatility',
                  'Calmar ratio',
                  'Max drawdown',
                  ]
    cell_values = [np.round(cumulative_return[-1], 4),
                   np.round(annualized_return, 4),
                   np.round(sharpe_ratio, 4),
                   np.round(time_return.std() * np.sqrt(252), 4),
                   np.round(calmar, 4),
                   np.round(max_drawdown.max(), 4)
                   ]
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )
    '''subplot one: metrics table'''
    fig.add_trace(
        go.Table(
            header=dict(
                values=cols_names,
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=cell_values,
                align="left")
        ),
        row=1, col=1
    )
    '''subplot two: cumulative return'''
    fig.add_trace(
        go.Scatter(
            x=cumulative_return.index,
            y=cumulative_return.values,
            mode="lines",
            name='cumulative return',
            marker=dict(color='green')
        ),
        row=2, col=1
    )
    '''subplot three: max drawdown'''
    fig.add_trace(
        go.Scatter(
            x=max_drawdown.index,
            y=max_drawdown.values,
            mode="lines",
            name='max drawdown',
            marker=dict(color='red')
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=600,
        title_text='Buy & Hold Performance',
        template='plotly_white'
    )
    fig.update_xaxes(showgrid=False)
    fig.show()


def transactions_info(transactions):
    """massage transactions(an output of backtrader) into time series to do visualization"""
    buy_date = []
    buy_price = []
    sell_date = []
    sell_price = []
    """
    transactions is an OrderedDict, key is date, 
    value is a list of [['amount', 'price', 'sid', 'symbol', 'value']]
    """
    for i, j in transactions.items():
        if j[0][0] > 0:
            '''j[0][0] is the amount, if amount > 0, it is a buy order'''
            buy_date.append(i)
            buy_price.append(j[0][1])

        else:
            '''if amount < 0, it is a sell order'''
            sell_date.append(i)
            sell_price.append(j[0][1])

    return buy_date, buy_price, sell_date, sell_price


def plot_transactions(data, transactions, time_return):
    """plot k line and orders"""
    df = data.copy()
    df.reset_index(inplace=True)
    df['time'] = pd.to_datetime(df['time'])

    buy_date, buy_price, sell_date, sell_price = transactions_info(transactions)
    '''the following four lines are just used to synchronize the time for plotting'''
    long = pd.DataFrame(np.vstack([buy_date, buy_price]).T, columns=['time', 'buy_price'])
    short = pd.DataFrame(np.vstack([sell_date, sell_price]).T, columns=['time', 'sell_price'])
    df = pd.merge(df, long, how='left', on='time')
    df = pd.merge(df, short, how='left', on='time')

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.15,
        specs=[[{"type": "Candlestick"}],
               [{"type": "scatter"}]]
    )

    '''subplot one: k line'''
    fig.add_trace(
        go.Candlestick(x=df.time,
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       showlegend=False),
        row=1, col=1
    )
    '''buy orders'''
    fig.add_scatter(
        mode='markers',
        x=df.time,
        y=df.buy_price,
        marker=dict(color='rgba(0,134,254,1)', size=10, symbol='arrow-up'),
        name='buy order',
        row=1, col=1
    )
    '''sell orders'''
    fig.add_scatter(
        mode='markers',
        x=df.time,
        y=df.sell_price,
        marker=dict(color='black', size=10, symbol='arrow-down'),
        name='sell order',
        row=1, col=1
    )
    '''only plot upper band when short and lower band when long'''
    plot_lower = df.mask(df['direction'] == 1)['lower_band']
    plot_upper = df.mask(df['direction'] == -1)['upper_band']
    plot_lower.replace(0, None, inplace=True)
    plot_upper.replace(0, None, inplace=True)
    '''upper band'''
    fig.add_scatter(
        mode='lines',
        x=df.time,
        y=plot_upper,
        marker=dict(color='red'),
        name='Upper Band',
        row=1, col=1
    )
    '''lower band'''
    fig.add_scatter(
        mode='lines',
        x=df.time,
        y=plot_lower,
        marker=dict(color='green'),
        name='Lower Band',
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df.time, y=df.dema, mode="lines", name='DEMA',
                             marker=dict(color='purple')), row=1, col=1)

    cumulative_return = (time_return + 1).cumprod()
    fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return.values, mode="lines",
                             name='cumulative return'),
                  row=2, col=1)

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=2,
                         label="2 month",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6 months",
                         step="month",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True, thickness=0.1),
            type="date"),
        height=700,
        title_text='Transactions',
        template='plotly_white'
    )

    fig.update_xaxes(showgrid=False)

    fig.show()
