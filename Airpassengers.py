# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
dateparse = lambda x:pd.datetime.strptime(x,'%Y/%m/%d')
df = pd.read_csv('./data/passengers.csv',parse_dates=['Month'],date_parser=dateparse)
df = df.set_index('Month')
# df['diff1'] = df['Passengers'].diff(1)
# df['diff2'] = df['diff1'].diff(1)
# df['diff3'] = df['diff2'].diff(1)
# fig = plt.figure(figsize=(15,5))
# df.plot(subplots=True)
# plt.show()


import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.seasonal import seasonal_decompose

# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(df['Passengers'],ax=ax1,lags=20)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(df['Passengers'],ax=ax2,lags=20)
# # ax1.xaxis.set_ticks_position('bottom')
# fig.tight_layout()
# plt.show()
# ts_log = np.log(df['Passengers'])
# # decomposition = seasonal_decompose(ts_log,period=12)
# # trend = decomposition.trend #趋势
# # seasonal = decomposition.seasonal  #季节性
# # residual = decomposition.resid     #残差序列
# # residual.dropna(inplace=True)
# #
# # fig,ax = plt.subplots(2,2)
# # ax1 = ax[0,0]
# # ax2 = ax[0,1]
# # ax3 = ax[1,0]
# #
# # ax1.plot(trend)
# # ax2.plot(seasonal)
# # ax3.plot(residual)
# # plt.show()


from statsmodels.tsa.arima_model import ARIMA
model_ARIMA = ARIMA(df,order=(2,0,2)).fit()
predictions_ARIMA = model_ARIMA.predict(start='1950-01',end='1962-04')
plt.plot(df['Passengers'])
plt.plot(predictions_ARIMA)
plt.show()
