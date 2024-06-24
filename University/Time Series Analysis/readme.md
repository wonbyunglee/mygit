# ARIMA model Fitting

Data name : 한국남부발전(주) 성산풍력발전실적

Data period : 2020.01.01 ~ 2022.12.31

Data source : 공공데이터포털 "한국남부발전(주) 성산풍력발전실적" https://www.data.go.kr/data/15043393/fileData.do

Analytical method : 시간별 발전량을 일평균한 값을 변수로 추세 및 계절성 조정을 통한 시계열 분석

## Data Preprocessing
```
import pandas as pd

df = pd.read_csv("시계열df.csv", encoding="euc-kr")
df.head(3)
```
```
import matplotlib.pyplot as plt

df['date'] = pd.to_datetime(df['date'])

monthly_mean = df.groupby(df['date'].dt.to_period("M")).mean()

monthly_mean_unique = monthly_mean[~monthly_mean.index.duplicated(keep='first')]

df = monthly_mean_unique
df = df.drop('date', axis=1)
df
```
- 일별 'mean' column을 월별 평균으로 계산

- 새로운 df 생성

## Sequence Chart
![image](https://github.com/wonbyunglee/mygit/assets/134191686/45ab2503-46b9-47de-bd6c-1f57ae75d188)

- 계절성이 뚜렷하게 보이는 것으로 판단

- 따라서 계절 차분 진행

## ACF, PACF Chart
```
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df['seasonal_diff'] = df['mean'].diff(12)

# 결측치 제거
df = df.dropna()

# ACF 및 PACF 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 계절 차분 적용 전 ACF 및 PACF
plot_acf(df['mean'], ax=ax1, lags=12, title='before ACF')
plot_pacf(df['mean'], ax=ax2, lags=10, title='before PACF')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 계절 차분 적용 후 ACF 및 PACF
plot_acf(df['seasonal_diff'], ax=ax1, lags=12, title='after ACF')
plot_pacf(df['seasonal_diff'], ax=ax2, lags=10, title='after PACF')

plt.show()
```
![image](https://github.com/wonbyunglee/mygit/assets/134191686/6be00216-e089-4e36-811a-eb1cb5705ec4)
![image](https://github.com/wonbyunglee/mygit/assets/134191686/7c60c284-c3e2-49dd-8407-996ff05bb4c0)

- df에 대한 계절 차분 진행

- 자기상관 그래프 확인 (계절 차분 전, 후 구분)

## SARIMA Model Fitting
![image](https://github.com/wonbyunglee/mygit/assets/134191686/6028a771-2f67-489a-bc13-8eb5337cd835)
![image](https://github.com/wonbyunglee/mygit/assets/134191686/3e601551-241d-4464-8cd7-53b0e6774122)

- acf, pacf 차트를 반영해 arima(2,1,2) * (1,1,1,12) 모델을 초기 모델로 적합 시작

- p-value 0.05을 기준으로 p-value가 0.05보다 높은 변수를 하나씩 차례로 제거해가며 모델 업데이트

- 최종 arima(0,1,1) * (1,1,0,12) 모델 적합

- acf 그래프를 보면 전보다 꽤 안정적인 것으로 보인다.

## Data Prediction

![image](https://github.com/wonbyunglee/mygit/assets/134191686/7e21fecc-514c-4d36-9a9e-6154d105aec7)

- 앞서 적합한 최종 모델 arima(0,1,1) * (1,1,0,12)을 적용하여 2023년 1년동안의 값 예측

## Data Comparison

![image](https://github.com/wonbyunglee/mygit/assets/134191686/98da265c-2636-43fe-8903-07d3a9e1c64a)
![image](https://github.com/wonbyunglee/mygit/assets/134191686/8ef9fd73-53cc-4ee6-a197-ef1a1a1fd7f5)

- R-studio을 통해 분석한 (주)두산의 주식 데이터와 비교

- 같은 시기인 빨간 원끼리 비교하였을 때 비슷한 추세를 보이고 있다.

- 따라서 (주)두산의 주식 값과 한국남부발전(주)의 풍력발전실적에 어느정도 상관관계가 있다고 볼 수 있다.








