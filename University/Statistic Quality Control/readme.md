# Control Chart about KORAIL operation data
Data source : https://www.data.go.kr/data/15079935/fileData.do '한국교통안전공단_국내 철도사고 및 운행장애 발생 정보'

Data Period : 2018.01.06 ~ 2022.12.27

About Data : KORAIL에서 제공하는 열차 운행 관련 정보들 중 열차 '지연(장애) 발생 건수'와 그에 따른 '피해액' 두 가지 데이터를 활용하여 Shewart 관리도를 적합, 상태 모니터링

Detail Code : 

X-Chart - https://github.com/wonbyunglee/mygit/blob/c9e6149d6fab339a392b7d5d21517e52bfc98da7/X-Chart.ipynb

C-Chart - https://github.com/wonbyunglee/mygit/blob/c9e6149d6fab339a392b7d5d21517e52bfc98da7/C-Chart.ipynb
## X-Chart ('피해액')
계량형 관리도에서 가장 널리 쓰이는 X-Chart는 주어진 데이터의 공정 평균을 모니터링하여 공정의 불안정성을 식별하고 수정하고자 하는 목표를 가진다. 이번 주제에서는 철도의 여러 원인으로 발생하는 월별 총 피해액의 평균을 모니터링하여 피해가 적정선을 넘지 않고 있는지를 확인하려고 한다. 일반적으로 공정 평균을 체크하는 X-Chart는 R-Chart와 세트로 적합한다. 여기서 R-Chart는 데이터의 범위를 모니터링하며, X-Chart를 적합하기 전 먼저 적합한 후 정상상태인지 확인해야 한다. 또한, X-Chart는 데이터들이 정규분포를 따라야 하므로 정규분포 여부를 꼭 확인해야 한다.

### Data Preprocessing

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
- 2018~2022년 자료를 업로드하고 그 중 적합하고자 하는 column '일자'와 '피해액'만으로 데이터 재구성

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

### Chart Fitting (phase1)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/02844b11-8e37-4928-a7b7-756d540c7108)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/ddaa25b5-49ca-4de1-b5df-1c66c02f84a2)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/3d5a3568-aeb4-4a92-af20-9575389edfa5)

- 앞서 설정한 phase1 데이터를 활용하여 R-Chart, X-Chart 적합

- X-Chart는 정규분포를 따라야 하므로 QQ-Plot 작성

- R-Chart, X-Chart 적합 결과는 초록색 3sigma 관리상한선을 넘지 않아 정상인 것으로 판단

- 하지만 QQ-Plot을 보면 값들이 정규분포를 따르지 않는 모습

### Log Conversion

 ```
phase1['총 피해액(백만원)'] = np.log(phase1['총 피해액(백만원)'])
phase1['총 피해액(백만원)'] = phase1['총 피해액(백만원)'].replace(-np.inf, 0.001)
phase1
```
- 정규분포 근사를 위해 로그변환 수행

### Chart Fitting after log conversion (phase1)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/f31382ca-4699-4f07-9fcb-bacb9bc00e73)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/ce7e7106-c875-467f-95ee-0bdd1cb9fb45)

- 로그변환 후 X-Chart와 QQ-Plot 적합

- 전보다 정규분포를 유사하게 따르는 것으로 확인

### Chart Fitting after log conversion (phase2)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/0231e10c-1995-43d9-a053-39f1c2bc0328)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/30f9e009-95e6-4f98-b2c0-2b6d41485271)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/59cbc916-0594-40fd-9c66-dbe525bd34b1)

- Phase2 데이터도 마찬가지로 로그변환 후 R-Chart, X-Chart, QQ-Plot 적합

- R-Chart, X-Chart 결과를 통해 값들이 관리한계 내에 존재하여 정상 상태라고 판단

- QQ-Plot을 통해 phase2 데이터 역시도 정규분포에 근사함을 확인

### Final X-Chart Fitting
![image](https://github.com/wonbyunglee/mygit/assets/134191686/4b445f59-407f-465c-93c0-fa132030d783)

- 전체적으로 3sigma 한계선 내에 분포하는 모습이다. 중간중간 2sigma 경고한계선을 넘는 값이 확인되는데, 저 시기에 많은 피해액이 발생한 사고가 있었던 것으로 추측 가능하다. 하지만 그런 몇몇 큰 사건들이 공정상태를 이상상태로까지 영향을 미치지는 않는다.

## C-Chart ('지연 수')

불량품은 규격을 하나 이상 만족하지 못한 생산품인데, 규격을 만족하지 못한 부분을 결점이라고 한다. 즉, 불량품은 하나 이상의 결점을 포함하고 있다. 공정관리의 많은 상황에서 불량률보다는 결점수로 직접 작업하는 것이 선호된다. 여기서 다룰 열차의 지연(장애) 수도 이에 해당한다. 우선 전처리를 통해 주별 열차 지연 수의 합계를 계산하고, 이를 C-Chart로 적합 시켜 열차의 지연 수가 적절히 유지되고 있는지 확인한다. C-Chart를 적합 시키기 위해서는 데이터가 포아송 분포를 따라야 한다. 따라서 C-Chart 적합 후 추가로 plot을 통해 포아송 분포를 따르는 지 검증하기로 한다.
### Data Preprocessing

```
df = pd.read_csv("21_delay.csv")
df = df[['week', 'total']]
df
```
```
df1 = pd.read_csv("22_delay.csv")
df1 = df1[['week', 'total']]
df1
```
- 21_delay, 22_delay 파일은 각각 2021, 2022년 열차 지연 수를 주별로 합산하여 정리한 데이터이다.

- C-Chart는 주별 지연 수의 합을 데이터로 사용

- 21_delay와 22_delay 파일 업로드

- '지연 수' 의 경우 과거와 최근 값의 편차가 심해 18~20년 데이터는 제외하고 21년 데이터를 phase1, 22년 데이터를 phase2로 사용

### C-Chart Fitting

![image](https://github.com/wonbyunglee/mygit/assets/134191686/041b677d-0ade-4ff3-87a8-ec6a8ccacb40)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/934586b0-ec42-4fb8-8cc5-03c66678b790)

- phase1과 phase2의 C-Chart 적합 결과 두 기간 모두 관리상한을 넘는 값이 존재, 즉 공정이 이상상태

- 따라서 이상상태로 탐지된 값들을 모두 제거 후 C-Chart 재적합 해야한다.

### Poisson Checking

![image](https://github.com/wonbyunglee/mygit/assets/134191686/5c7c0891-58c6-4c05-9312-c1b570099de2)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/36c49f65-2a17-43ab-a3a7-8cf31f7e1ac2)

- 히스토그램 결과 포아송 plot이 왼쪽으로 치우쳐져 있어 포아송 분포를 잘 따른다고 볼 수 없다.

### Removing Outliers

```
df['Mean'] = df['total'].mean()
df['UCL'] = df['Mean'] + 3 * np.sqrt(df['Mean'])
new_df = df[df['total'] <= df['UCL']]
new_df
```
```
df1['Mean'] = df1['total'].mean()
df1['UCL'] = df1['Mean'] + 3 * np.sqrt(df1['Mean'])
new_df1 = df1[df1['total'] <= df1['UCL']]
new_df1
```
- phase1과 phase2 데이터에서 각각 관리한계를 초과한 값들 제거 후 새로운 데이터 생성

### New C-Chart Fitting, poisson Checking

![image](https://github.com/wonbyunglee/mygit/assets/134191686/74c62e9c-dba6-46b2-978f-366d06445bcb)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/0f0ac394-0151-44a3-a7b2-71698371f97a)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/a82fe06b-383d-4722-9ac2-34683f03271e)

![image](https://github.com/wonbyunglee/mygit/assets/134191686/ae8eb396-8bff-4353-8787-d4e1c950bd10)

- 1차적으로 적합한 C-Chart, poisson 플롯과 비교해보면 C-Chart에서는 여전히 관리상한을 넘는 이상치가 보인다. 대신 poisson 플롯은 앞서 적합한 것보다 상당히 가운데 형성된 것을 확인할 수 있다. 즉, 포아송 분포에 근사한다는 의미이다.

### Final C-Chart Fitting

![image](https://github.com/wonbyunglee/mygit/assets/134191686/7b2a2125-df22-4baf-8d3d-fd57b1ad32c5)

- 최종 C-Chart를 적합해보면 phase1과 phase2 부분에서 이상상태로 탐지되는 값들이 몇 개 관측된다.

- 또한 phase2의 뒷부분을 보면 X-Chart의 동일기간에서 2sigma를 넘어 평균보다 꽤 높은 값으로 측정되었던 부분과 C-Chart에서 관리한계를 넘는 부분들이 일치한다. 이는 그 시기에 대형 사고 등의 이유로 피해액이 많이 발생했을 것이라는 추측과 대립되는 점이다. 즉, 어떠한 이유로 지연된 열차 자체가 늘었고 그로 인해 피해액도 증가했을 것이라는 추측이 더 합리적이다. 그 원인에 대해서는 추가적인 분석을 통해 근거를 마련할 필요성도 있다.
