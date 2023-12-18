## Control Chart about KORAIL operation data
Data source : https://www.data.go.kr/data/15079935/fileData.do '한국교통안전공단_국내 철도사고 및 운행장애 발생 정보'

Data Period : 2018.01.06 ~ 2022.12.27

About Data : KORAIL에서 제공하는 열차 운행 관련 정보들 중 열차 '지연(장애) 발생 건수'와 그에 따른 '피해액' 두 가지 데이터를 활용하여 Shewart 관리도를 적합, 상태 모니터링

### X-Chart ('피해액')

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df18 = pd.read_csv("18all.csv", encoding='euc-kr')
df19 = pd.read_csv("19all.csv", encoding='euc-kr')
df20 = pd.read_csv("20all.csv", encoding='euc-kr')
df21 = pd.read_csv("21all.csv", encoding='euc-kr')
df22 = pd.read_csv("22all.csv", encoding='euc-kr')

df18 = df18[['일자', '총 피해액(백만원)']]
df19 = df19[['일자', '총 피해액(백만원)']]
df20 = df20[['일자', '총 피해액(백만원)']]
df21 = df21[['일자', '총 피해액(백만원)']]
df22 = df22[['일자', '총 피해액(백만원)']]
```
2018~2022년 자료를 업로드하고 그 중 적합하고자 하는 column '일자'와 '피해액'만으로 데이터 재구성

```
df18['일자'] = pd.to_datetime(df18['일자'])
monthly_mean = df18.groupby(df18['일자'].dt.to_period("M"))['총 피해액(백만원)'].mean()

# 새로운 열로 추가
df18['월별 평균 피해액'] = df18['일자'].dt.to_period("M").map(monthly_mean)
monthly_mean_unique = monthly_mean[~monthly_mean.index.duplicated(keep='first')]

# '일자' 열을 데이터프레임의 인덱스로 설정
monthly_mean_unique = monthly_mean_unique.reset_index()
df18 = monthly_mean_unique
df18
```
```
dfs = [df18, df19, df20, df21]
phase1 = pd.concat(dfs, ignore_index=True)
phase1
```
```
phase2 = df22
phase2
```
- df18부터 df22까지의 데이터의 '일자' column을 월별로 하나씩만 표시하도록 설정

- '총 피해액(백만원)' column의 값들은 각 월별 평균 계산

- df18~df21 까지는 phase1, df22는 phase2로 설정 (phase1을 이전 4년 동안의 기간으로 설정하여 관리도의 평균 및 관리한계를 구하고 22년도 데이터를 phase2로 설정하여 phase1을 통해 결정한 평균과 관리한계에 데이터를 적합, 관리상태에 있는지 모니터링)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/02844b11-8e37-4928-a7b7-756d540c7108)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/ddaa25b5-49ca-4de1-b5df-1c66c02f84a2)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/3d5a3568-aeb4-4a92-af20-9575389edfa5)

- 앞서 설정한 phase1 데이터를 활용하여 R-Chart, X-Chart 적합

- X-Chart는 정규분포를 따라야 하므로 QQ-Plot 작성

- R-Chart, X-Chart 적합 결과는 초록색 3sigma 관리상한선을 넘지 않아 정상인 것으로 판단

- 하지만 QQ-Plot을 보면 값들이 정규분포를 따르지 않는 모습

 ```
phase1['총 피해액(백만원)'] = np.log(phase1['총 피해액(백만원)'])
phase1['총 피해액(백만원)'] = phase1['총 피해액(백만원)'].replace(-np.inf, 0.001)
phase1
```
정규분포 근사를 위해 로그변환 수행

![image](https://github.com/wonbyunglee/mygit/assets/134191686/f31382ca-4699-4f07-9fcb-bacb9bc00e73)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/ce7e7106-c875-467f-95ee-0bdd1cb9fb45)

- 로그변환 후 X-Chart와 QQ-Plot 적합

- 전보다 정규분포를 유사하게 따르는 것으로 확인

![image](https://github.com/wonbyunglee/mygit/assets/134191686/0231e10c-1995-43d9-a053-39f1c2bc0328)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/30f9e009-95e6-4f98-b2c0-2b6d41485271)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/59cbc916-0594-40fd-9c66-dbe525bd34b1)

- Phase2 데이터도 마찬가지로 로그변환 후 R-Chart, X-Chart, QQ-Plot 적합

- R-Chart, X-Chart 결과를 통해 값들이 관리한계 내에 존재하여 정상 상태라고 판단

- QQ-Plot을 통해 phase2 데이터 역시도 정규분포에 근사함을 확인

![image](https://github.com/wonbyunglee/mygit/assets/134191686/4b445f59-407f-465c-93c0-fa132030d783)

- 최종 X-Chart 적합

- 전체적으로 3sigma 한계선 내에 분포하는 모습이다. 중간중간 2sigma 경고한계선을 넘는 값이 확인되는데, 저 시기에 많은 피해액이 발생한 사고가 있었던 것으로 추측 가능하다. 하지만 그런 몇몇 큰 사건들이 공정상태를 이상상태로까지 영향을 미치지는 않는다.
