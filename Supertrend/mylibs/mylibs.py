import backtrader as bt
import numpy as np
import pandas as pd
import ta
import itertools


def calculate_metrics(time_return):
    """
    input: a time series of return
    output: cumulative_return, annualized_return, sharpe_ratio, calmar, max_drawdown
    note: cumulative_return and max_drawdown is time series and the others are scaler
    """
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


def generate_super_trend(data, factor, atr_period):
    """
    data should contain columns: high, low and close
    output the original dataframe attached with new columns: atr, upper_band, lower_band, direction and super_trend
    """
    df = data.copy()
    '''calculate ATR'''
    df['atr'] = ta.volatility.average_true_range(df.high, df.low, df.close, atr_period)
    '''replace 0 with None for later coding convenience'''
    df['atr'] = df['atr'].replace(0, None)

    '''the foundation of super trend is the mean value of high and low plus ATR multiplied by a factor'''
    src = (df.high + df.low) / 2
    upper_band = src + factor * df.atr
    lower_band = src - factor * df.atr
    upper_band.fillna(0, inplace=True)
    lower_band.fillna(0, inplace=True)

    '''
    the core step to generate super trend
    idea: now we have upper_band and lower_band, we do not want the upper_band to increase when we hold short position
    or lower_band to decrease when we hold long position
    '''
    for i in range(1, len(lower_band)):
        lower_band[i] = lower_band[i] if (lower_band[i] > lower_band[i - 1]) or (
                    df.close[i - 1] < lower_band[i - 1]) else lower_band[i - 1]
        upper_band[i] = upper_band[i] if (upper_band[i] < upper_band[i - 1]) or (
                    df.close[i - 1] > upper_band[i - 1]) else upper_band[i - 1]

    df['upper_band'] = upper_band
    df['lower_band'] = lower_band

    prev_upper_band = upper_band.shift()

    '''
    direction helps generate trading signals
    direction == -1 means we are holding long position
    direction == 1 means we are holding short position
    note: we can only trade after the first -1 or 1, because direction is updated at the end of the day, 
          we can not trade at the first day when direction changes
    '''
    direction = pd.Series([None] * len(upper_band))
    '''super_trend is for plotting'''
    super_trend = pd.Series([None] * len(upper_band))
    for i in range(1, len(direction)):
        if df.atr[i] is None:
            direction[i] = 1
        elif super_trend[i - 1] == prev_upper_band[i]:          # when we are holding short position
            '''close price up cross the upper band, ready to long'''
            direction[i] = -1 if df.close[i] > upper_band[i] else 1
        else:                                                   # when we are holding long position
            '''close price down cross the lower band, ready to short'''
            direction[i] = 1 if df.close[i] < lower_band[i] else -1
        super_trend[i] = lower_band[i] if direction[i] == -1 else upper_band[i]

    df['direction'] = direction.values
    df['super_trend'] = super_trend.values
    return df


def generate_dema(data, dema_length=144):
    df = data.copy()
    e1 = ta.trend.ema_indicator(df.close, dema_length)
    e2 = ta.trend.ema_indicator(e1, dema_length)
    dema = 2 * e1 - e2
    df['dema'] = dema
    return df


def generate_entry_signal(data):
    df = data.copy()
    '''super_trend_signal is the long/short signal of super trend (when we are not using DEMA to support)'''
    super_trend_signal = - (df.direction - df.direction.shift())
    '''at the beginning of the time period, some indicators are None, replace them with 0'''
    super_trend_signal.fillna(0, inplace=True)
    '''use entry_signal to hold long/short signals'''
    df['entry_signal'] = 0
    '''only enter long position when close price is higher than dema'''
    df['entry_signal'] = df['entry_signal'].mask((df.close > df.dema) & (super_trend_signal > 0), 1)
    '''only enter short position when close price is lower than dema'''
    df['entry_signal'] = df['entry_signal'].mask((df.close < df.dema) & (super_trend_signal < 0), -1)

    '''when super trend indicates long position, and the price break up dema'''
    df['entry_signal'] = df['entry_signal'].mask(((df.direction < 0) & (df.close.shift() <= df.dema) &
                                                  (df.close > df.dema)), 1)
    '''when super trend indicates short position, and the price break down dema'''
    df['entry_signal'] = df['entry_signal'].mask(((df.direction > 0) & (df.close.shift() >= df.dema) &
                                                  (df.close < df.dema)), -1)
    '''do not trade in the period when DEMA has no value'''
    df['entry_signal'] = df['entry_signal'].mask(df.dema.isnull(), 0)
    return df


def generate_stop_signal(data):
    df = data.copy()
    real_position = data.entry_signal.replace(0, None)
    real_position.ffill(inplace=True)
    super_trend_signal = - (df.direction - df.direction.shift())
    super_trend_signal.fillna(0, inplace=True)
    '''close long position when close price is above DEMA and super_trend_signal gives short signal'''
    df['stop_long'] = np.where((real_position > 0) & (super_trend_signal < 0) & (df.close > df.dema), 1, 0)
    '''close short position when close price is below DEMA and super_trend_signal gives long signal'''
    df['stop_short'] = np.where((real_position < 0) & (super_trend_signal > 0) & (df.close < df.dema), 1, 0)
    return df


def data_massage(data, factor, atr_period, dema_length, start, end):
    """integrate all data processing functions"""
    df = data.copy()
    df = generate_super_trend(df, factor, atr_period)
    df = generate_dema(df, dema_length)
    df = generate_entry_signal(df)
    df = generate_stop_signal(df)
    df = df[start:end]
    return df


class MyData(bt.feeds.PandasData):
    """
    in backtrader, it is necessary to define a class if user want to feed
    data with pre-defined columns rather than the necessary seven columns
    """
    '''put the names of pre-defined columns in a tuple'''
    lines = ('entry_signal', 'stop_long', 'stop_short')
    '''seven necessary columns'''
    params_list = [
        ('datetime', 'time'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', 'openinterest'),
    ]
    '''add pre-defined columns'''
    for i in lines:
        params_list.append((i, i))
    params = tuple(params_list)


class TestStrategy(bt.Strategy):
    """customize the strategy"""
    params = dict(
        target=0.8
    )

    def __init__(self):
        self.entry_signal = self.data.entry_signal
        self.stop_long = self.data.stop_long
        self.stop_short = self.data.stop_short
        self.order = None

    def next(self):
        """if no position"""
        if not self.position:
            # long signal
            if self.entry_signal > 0:
                self.order = self.order_target_percent(target=self.p.target, info='long')
            # short signal
            elif self.entry_signal < 0:
                self.order = self.order_target_percent(target=-self.p.target)
        # when holding long position
        elif self.getposition(self.data).size > 0:
            # short signal
            if self.entry_signal < 0:
                self.order = self.order_target_percent(target=-self.p.target)
            # stop long signal
            elif self.stop_long == 1:
                self.order = self.close()
        # when holding short position
        else:
            # long signal
            if self.entry_signal > 0:
                self.order = self.order_target_percent(target=self.p.target)
            # stop short signal
            elif self.stop_short == 1:
                self.order = self.close()


class CommInfoFractional(bt.CommissionInfo):
    """set parameter commission to control commission fee"""
    params = (
        ('commission', 0.0003),
    )
    '''set the minimum order as 0.000001 BTC'''
    def getsize(self, price, cash):
        """Returns fractional size for cash operation @price"""
        return np.round(self.p.leverage * (cash / price), 6)


def backtest_results(data, initial_cash=100000.0, order_size=80, commission=0.0003, slippage=0.0001):
    df = data.copy()
    '''the data fed into backtrader should contain a column named 'openinterest'''
    df['openinterest'] = 0
    df.reset_index(inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    '''initialize backtrader engine'''
    cerebro = bt.Cerebro()
    '''feed data'''
    data = MyData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=240)
    cerebro.adddata(data)
    '''add strategy'''
    cerebro.addstrategy(TestStrategy, target=order_size / 100)
    '''initial cash'''
    cerebro.broker.setcash(initial_cash)
    '''set fractional contract and commission'''
    cerebro.broker.addcommissioninfo(CommInfoFractional(commission=commission))
    '''slippage'''
    cerebro.broker.set_slippage_perc(perc=slippage)
    '''record time return and transaction points'''
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Minutes, compression=240, _name='time_return')
    cerebro.addanalyzer(bt.analyzers.Transactions, _name='trade')
    '''start engine'''
    result = cerebro.run(stdstats=False)
    result = result[0]
    '''get time return and transaction points'''
    time_return = pd.Series(result.analyzers.time_return.get_analysis())
    transactions = result.analyzers.trade.get_analysis()
    return time_return, transactions


def grid_search(data, start, end, initial_cash, parameter_dic, commission, slippage):
    """
    to increase scalability
    parameter_dic should be an ordered dictionary, of which keys should be <parameter name>_span,
    values should be a list of int/float
    """
    spans_string = ''
    for i in parameter_dic.values():
        spans_string += str(i) + ','
    '''use itertools.product() to write the loop'''
    combos = itertools.product(*eval(spans_string))

    '''create an empty dataframe to store all results'''
    grid_search_result = pd.DataFrame()

    for i in combos:
        '''i is a tuple of parameter combination'''
        '''create an empty dataframe to store one single search'''
        each_row = pd.DataFrame()
        for n, j in enumerate(parameter_dic.keys()):
            '''
            n is the order of parameters in parameter_dic
            and n should also match the order of parameters in i
            j is something like <parameter name>_span, so j[:-5] can get rid of _span and only <parameter name> is left
            use globals() to create variables automatically with strings
            '''
            globals()[j[:-5]] = i[n]
            '''write parameter values into dataframe'''
            each_row[j[:-5]] = [i[n]]

        '''generate data'''
        df = data_massage(data, factor, atr_period, dema_length, start, end)
        df.dropna()
        '''divide the backtest period into 5 segments and backtest on each segment'''
        step = int(len(df) / 5)
        df_1 = df.iloc[:step]
        df_2 = df.iloc[step: 2 * step]
        df_3 = df.iloc[2 * step: 3 * step]
        df_4 = df.iloc[3 * step: 4 * step]
        df_5 = df.iloc[4 * step:]

        for n, j in enumerate([df_1, df_2, df_3, df_4, df_5]):

            time_return, transactions = backtest_results(j, initial_cash, order_size, commission, slippage)
            cumulative_return, annualized_return, sharpe_ratio, calmar, max_drawdown = calculate_metrics(time_return)
            '''write metrics values into dataframe'''
            each_row[f'{str(n)}_cumulative_return'] = [np.round(cumulative_return[-1], 4)]
            each_row[f'{str(n)}_max_drawdown'] = [np.round(max_drawdown.max(), 4)]
            each_row[f'{str(n)}_sharpe_ratio'] = [np.round(sharpe_ratio, 4)]
            each_row[f'{str(n)}_annual volatility'] = [np.round(time_return.std() * np.sqrt(252), 4)]
            each_row[f'{str(n)}_annualized_return'] = [np.round(annualized_return, 4)]
            each_row[f'{str(n)}_calmar'] = [np.round(calmar, 4)]
            each_row[f'{str(n)}_number_of_transactions'] = [len(transactions)]

        grid_search_result = pd.concat([grid_search_result, each_row])
    return grid_search_result
