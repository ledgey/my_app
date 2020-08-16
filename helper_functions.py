import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import plotly.graph_objects as go

import CVAE as cvae


encoder, _,_ =  cvae.build_cvae()
encoder.load_weights('encoder.h5')
ngb = pickle.load(open('ngboost.pkl', 'rb'))
symbols = pd.read_csv('forex.csv')
ticks = symbols.Symbol.tolist()


def LagguerreRSI(data, gamma):
    # Create base data structures
    out = []
    l =[np.zeros(4), [data.iloc[0]]*4]
    cu =np.zeros(3)
    cd =np.zeros(3)
    for i in data.index:
        # Different calculation for if it is the first price value
        l[0][0] = ((1-gamma)*data[i]) + (gamma*l[1][0])
        l[0][1] = ((-gamma)*l[0][0]) + l[1][0] + (gamma*l[1][1])
        l[0][2] = ((-gamma)*l[0][1]) + l[1][1] + (gamma*l[1][2])
        l[0][3] = ((-gamma)*l[0][2]) + l[1][2] + (gamma*l[1][3])
        # Calculate the Count Up and Count Down   
        cu[0] = l[0][0] - l[0][1] if l[0][0] >= l[0][1] else 0
        cu[1] = cu[0] + (l[0][1] - l[0][2]) if l[0][1] >= l[0][2] else cu[0]
        cu[2] = cu[1] + (l[0][2] - l[0][3]) if l[0][2] >= l[0][3] else cu[1]
        
        cd[0] = 0 if l[0][0] >= l[0][1] else l[0][1] - l[0][0]
        cd[1] = cd[0] if l[0][1] >= l[0][2] else cd[0] + (l[0][2] -l[0][1])
        cd[2] = cd[1] if l[0][2] >= l[0][3] else cd[1] + (l[0][3] - l[0][2])
        # Calculate the RSI from the final CU and CD
        out.append(cu[2]/(cu[2] + cd[2]) if (cu[2] + cd[2]) != 0 else 0)
        # Store the Laguerre values to another list so previous values can be reffed
        l[1] = l[0]
        l[0] = np.zeros(4)
                
    return out


def calculate_k(df):
    close = df.Close
    highest_high = df.High.rolling(10).max()
    lowest_low = df.Low.rolling(10).min()
    df['per_k_10'] = (close - lowest_low)/(highest_high-lowest_low)
    return df


def calculate_d(df):
    df['per_d_10'] = df['per_k_10'].rolling(10).mean()
    return df


def stoichastic_osc(df):
    df_new = calculate_k(df)
    df_new = calculate_d(df_new)
    return df_new


def macd_signal(df):
    exp1 = df.Close.ewm(span=12, adjust=False).mean()
    exp2 = df.Close.ewm(span=26, adjust=False).mean()
    df['macd'] = exp1-exp2
    df['macd_signal'] = df.macd.ewm(span=9, adjust=False).mean()
    return df


def MinMax(x):
    xt = (x-min(x))/(max(x)-min(x))
    return xt.fillna(0)


def getData(tick, lookback, interval, tz):
    ticker = yf.Ticker(tick)
    tmp = ticker.history(period=lookback, interval=interval)
    cols = ['Open','High', 'Low', 'Close']
    tmp = tmp[cols] 
    tmp.index = tmp.index.tz_convert(tz=tz)
    return tmp


def transform(tmp):
    tmp['time'] = (tmp.index.hour*60 + tmp.index.minute) /1440
    tmp['sma'] = tmp.Close.rolling(20).mean()
    tmp = stoichastic_osc(tmp)
    tmp = macd_signal(tmp)
    std = tmp.Close.rolling(20).std()
    tmp['upper_bol'] = tmp['sma'] + (2*std)
    tmp['lower_bol'] = tmp['sma'] - (2*std)
    tmp['RSI_point8'] = tmp[['Close']].apply(LagguerreRSI, gamma = 0.8)
    price_cols = ['Open','High', 'Low', 'Close', 'sma',  'upper_bol', 'lower_bol']
    return tmp.dropna()


def scale(pred_data, seq_length):
    price_cols = ['Open','High', 'Low', 'Close', 'sma',  'upper_bol', 'lower_bol']
    macd_cols = ['macd','macd_signal']
    osc_cols = ['per_k_10', 'per_d_10']
    non_scaled_cols = ['time','RSI_point8']
    idxs = []
    dts = []
    obs_pred = []
    for i in range(len(pred_data)-(seq_length)+1):
        obs = pred_data.iloc[i:i+seq_length]
        if len(obs) == seq_length:
            price_obs = obs[price_cols].values
            price_obs = (price_obs - price_obs.min())/(price_obs.max()-price_obs.min())
            macd_obs = obs[macd_cols].values
            macd_obs = (macd_obs - macd_obs.min())/(macd_obs.max()-macd_obs.min())
            osc_obs = obs[osc_cols].values
            osc_obs = (osc_obs - osc_obs.min())/(osc_obs.max()-osc_obs.min())
            other_obs = obs[non_scaled_cols].values
            obs_pp = np.concatenate([price_obs, macd_obs,osc_obs,other_obs], axis=1)
            obs_pred.append(obs_pp)
            idxs.append([i,i+seq_length])
            dts.append(obs.index.max())

    return obs_pred, idxs, dts


def plotResults(results):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['upper_bol'],
        name="upper_bol",
        mode='lines'

    ))

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['lower_bol'],
        name="lower_bol",
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['sma'],
        name="sma",
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['pred'],
        name="pred",
        line = dict(color='black', width=2),
        yaxis="y2"
    ))


    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['pred']+results['prob'],
        name="pred_upper",
        line = dict(color='black', width=1, dash='dot'),
        yaxis="y2"
    ))
    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['pred']-results['prob'],
        name="pred_lower",
        line = dict(color='black', width=1, dash='dot'),
        yaxis="y2"
    ))

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['macd'],
        name="macd",
        line = dict(color='red', width=2),
        yaxis="y3"
    ))

    fig.add_trace(go.Scatter(
        x=results.index,
        y=results['macd_signal'],
        name="macd_signal",
        line = dict(color='blue', width=2),
        yaxis="y3"
    ))
    # Add traces
    fig.add_trace(go.Candlestick(x=results.index,
                    open=results['Open'],
                    high=results['High'],
                    low=results['Low'],
                    close=results['Close']))

    fig.update_layout(
        title='Trace',
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            title="yaxis title",
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        yaxis2=dict(
            title="Pred Slope",
            titlefont=dict(
                color="#d62728"
            ),
            tickfont=dict(
                color="#d62728"
            ),
            overlaying='y',
            anchor="free",
            side="right"
        ),
        yaxis3={'title': "MACD", 'overlaying': 'y', 'anchor': "free", 'side': "right"})

    return fig


import plotly.figure_factory as ff


def plotNormal(mu, sigma):
    # Add histogram data
    x1 = sigma * np.random.randn(1000) + mu

    # Create distplot with custom bin_size
    fig = ff.create_distplot([x1],group_labels=['Slope'],bin_size=.5,
                         curve_type='normal', show_hist=False,show_rug=False)
    fig.add_shape(
        type="line",
        yref="paper",
        x0=0,
        y0=0,
        x1=0,
        y1=1,
        line=dict(
            color="DarkOrange",
            width=3,
        ),
    )
    return fig


def runPrediction(tick, seq_length=128, lookback = '1d', interval='5m', tz = 'US/Eastern'):
    df = getData(tick, lookback, interval, tz)
    df = transform(df)
    x, idx, dts = scale(df, seq_length)
    _,_,z = encoder.predict(np.array(x), batch_size=64)
    preds = (ngb.predict(z))
    probs = (ngb.pred_dist(z).scale)
    predDict = {'prob': probs,'pred':preds, 'idx':idx, 'times':dts}
    gbpyn = pd.DataFrame(predDict)
    gbpyn.index = pd.DatetimeIndex(gbpyn['times'])
    results = df.join(gbpyn)
    print(results.tail(5))
    dist = results[['pred', 'prob']].iloc[-1].values
    # print(dist)
    dist = plotNormal(dist[0], dist[1])
    graph = plotResults(results.dropna())
    last_time = results.index.max().strftime('%Y-%m-%d %H:%M:%S')
    return graph, dist, last_time

