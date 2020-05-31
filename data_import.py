
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_excel("dataset/day_data_weather.xlsx", sheet_name="일별데이터_tidy")
data_corr_water_weather = data.loc[:, ['일자', '수분 함유율', '평균기온', '평균 상대습도']]
data_corr_electricity = data.loc[:, ['전력', '도시가스(LNG)', '슬래그파우더', '소계', '수분 함유량', '수분 함유율', '평균 상대습도']]
data_corr_water_weather = data_corr_water_weather.iloc[ 1100:, ]
data_corr_electricity = data_corr_electricity.iloc[1100:, ]
#print(data_corr_water_weather.head(20))
#print(data_corr_electricity.head(20))

data_corr_water_weather=data_corr_water_weather.dropna(axis=0, how='any')
data_corr_electricity=data_corr_electricity.dropna(axis=0, how='any')

#print(data_corr_water_weather.head(20))
#print(data_corr_electricity.head(20))

#print(data_corr_water_weather.corr())
#print(data_corr_electricity.corr())

data_corr_electricity_X = data_corr_electricity.loc[:, ['소계', '수분 함유량']]
data_corr_electricity_Y = data_corr_electricity.iloc[:, 0]

#print(data_corr_electricity_X.head(10)
reg = LinearRegression().fit(data_corr_electricity_X, data_corr_electricity_Y)
y_pred = reg.predict(data_corr_electricity_X)

#plt.plot(data_corr_electricity_X, data_corr_electricity_Y, alpha=0.7)
#plt.plot(data_corr_electricity_X, y_pred, color='red')
#plt.show()

print(reg.coef_)
print("MSE : ", mean_squared_error(data_corr_electricity_Y, y_pred))