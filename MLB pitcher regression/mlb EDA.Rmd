---
title: "EDA 기말과제"  
author: "응용통계학과 32193335 이원병"  
output: 
  html_document:
    css: bootstrap.css
---

```{r, include=FALSE}
# setting default options
knitr::opts_chunk$set(echo = TRUE)  # show R commands
options(width = 100)
```


<br/>  
<!--- #1 --->  

| <span style="color:darkblue">#1. 분석대상 데이터에 대한 소개</span> |
|---|
| 과제에서 분석할 데이터를 간략히 소개하고 분석목표를 제시하시오.|

<span style="color:darkred">__answer)__</span>  
<!--- You can write your answer #1 here. --->

## 데이터 소개 및 출처

이 데이터는 미국프로야구(MLB) 2022년도 정규시즌(포스트시즌 포함x) 전체 투수들의 기록이다. Mlb.com에 기록된 각종 투수들의 데이터들 중 연관성이 있거나 의미를 도출할 수 있을 만한 데이터변수들을 추려서 최종 데이터로 생성하였다. 또한, 1경기라도 등판한 기록이 있는 모든 선수들을 포함할 시 데이터의 양이 너무 커지는 관계로, 규정이닝인 162이닝의 약 50%이상을 충족하는 선수들의 데이터로 한정하였다.

출처: https://baseballsavant.mlb.com/leaderboard/custom?year=2022&type=pitcher&filter=&sort=8&sortDir=asc&min=250&selections=player_age,p_game,p_formatted_ip,p_total_hits,p_home_run,p_strikeout,p_walk,p_k_percent,p_bb_percent,batting_avg,slg_percent,on_base_percent,on_base_plus_slg,p_earned_run,p_run,p_save,p_win,p_loss,p_era,p_quality_start,p_complete_game,p_hit_by_pitch,p_hold,in_zone_percent,edge_percent,f_strike_percent,fastball_avg_speed,fastball_avg_spin,breaking_avg_speed,breaking_avg_spin,&chart=false&x=player_age&y=player_age&r=no&chartType=beeswarm

## 데이터의 각 변수들에 대한 설명

Raw 데이터는 위에서 설명한 바와 같이 규정이닝의 50%이상을 충족하는 선수들로 한정한 결과 총 275명의 선수로 구성되어 있다. Column 변수는 총 30개로, 전체 275x30=8,250개의 데이터가 존재한다.

### Column 변수 정의

* Last_name : 성
* First_name : 이름
* Player_id : mlb.com에서 임의로 정해놓은 선수별 식별번호
* Year : 2022시즌의 기록이므로 전부 2022로 동일
* Player_age : 나이
* P_game : 등판한 경기 횟수
* P_formatted_ip : 투구 이닝
* P_total_hits : 전체 피안타(상대 타자에게 허용한 안타)
* P_home_run : 전체 피홈런(상대 타자에게 허용한 홈런)
* P_strikeout : 탈삼진(상대 타자로부터 잡아낸 삼진)
* P_walk : 사사구
* P_k_percent : 탈삼진율(잡아낸 모든 아웃카운트 중 탈삼진 비율)
* P_bb_percent : 사사구율(기록한 모든 결과 중 사사구 비율)
* Batting_avg : 피안타율
* Slg_percent : 피장타율
* On_base_percent : 피출루율
* On_base_plus_slg : 피OPS(출루율+장타율)
* P_earned_run : 자책점(마운드에 있는 동안 야수의 실책 등으로 인한 실점을 제외한 허용한 점수)
* P_run : 실점(마운드에 있는 동안 허용한 총 점수)
* P_save : 세이브(특정요건에 의해 리드 시 등판하여 경기를 승리로 마무리한 경우)
* P_win : 승리
* P_loss : 패배
* P_era : 평균자책점, 방어율(9이닝당 자책점의 평균)
* P_quality_start : QS(6이닝 이상 3자책점 이하 기록 시)
* P_complete_game : 완투(한 경기를 혼자 끝낸 경우)
* P_hit_by_pitch : 사구
* P_hold : 홀드(리드 시 그 리드를 유지한 채로 다음 투수에게 넘겨준 경우)
* In_zone_percent : 스트라이크존에 던질 확률
* Edge_percent : 스트라이크존 각 방향의 모서리에 던질 확률
* F_strike_percent : 초구에 스트라이크를 던질 확률
* Fastball_avg_speed : 패스트볼의 평균 구속(Mile)
* Fastball_avg_spin : 패스트볼의 평균 회전수(RPM)
* Breaking_avg_speed : 변화구의 평균 구속(Mile)
* Breaking_avg_spin : 변화구의 평균 회전수(RPM)

### 변수 타입
```{r}
mlbp <- read.csv("mlb.csv")
str(mlbp)
```

선수의 성과 이름을 제외한 나머지 변수들은 모두 numeric(숫자형)과 integer(정수형)타입으로 구성되어 있다. 또한 대부분의 변수들이 연속형 변수이며, 범주형 변수가 존재하지 않는다. 따라서 범주형 변수를 추가하기 위해 각 선수들이 뛴 리그를 AL(American League)와 NL(National League)로 구분해 AL을 0, NL을 1로 하는 변수를 추가하였다.

### 분석 목표

투수의 데이터에서 가장 주목해야 하는 변수는 p_era이다. p_era의 값이 투수의 능력과 성적을 전반적으로 대변하는 데이터로 보기 때문이다. 따라서 p_era변수와 나머지 변수간의 관계를 살펴보고 어떤 변수가 p_era에 영향을 미치는 지, **궁극적으로 p_era를 낮추기 위해 필요한 부분은 무엇인지 살펴보고자 한다.**

<!--- end of #1 --->  

 
<br/>  
<!--- #2 --->  

| <span style="color:darkblue">#2. 개별 변수에 대한 탐색적 자료분석</span> |
|---|
| 개별 변수에 대한 탐색적 자료분석을 수행하고 결과를 설명해보시오.|

<span style="color:darkred">__answer)__</span>  
<!--- You can write your answer #2 here. --->

## 각 변수들의 특징 분석

* 총 30개의 변수 중 분석 목표에 맞는 중요 변수를 일부 선택해 그 변수들의 특징을 분석한다.

| <위에서 살펴본 변수들의 타입 중 numeric(숫자형) 변수들은 대부분 확률을 나타낸다.> 
|---|

### 평균자책점(p_era)
```{r}
summary(mlbp$p_era)
```

p_era 변수는 위의 변수들과 달리 확률을 나타내지 않는다. 그 선수가 던진 총 이닝과 기록한 총 자책점을 바탕으로 9이닝당 자책점으로 환산하여 나타낸 수치가 바로 평균자책점이다. 평균자책점은 투수를 설명하는 가장 대표적인 지표이다. 물론 다른 지표도 종합적으로 봐야겠지만, 평균자책점이 2.5 이하이면 mvp급 투수로 평가받는다. 위의 수치만 봐도 평균이 3.89, 1사분위수도 3.1로 나타나 낮은 평균자책점을 기록하는 것이 얼마나 어려운 일인지 알 수 있다.

```{r}
hist(mlbp$p_era, freq=T)
boxplot(mlbp$p_era)
```

투수의 평균자책점을 의미하는 p_era 변수에 대한 히스토그램과 박스플롯을 확인해 보면, p_era는 대체로 정규분포를 따른다. 하지만 박스플롯에서 자세히 나와있듯이, p_era가 7 이상으로 보이는 이상점 3개가 있다. 따라서, 추후 분석에서는 이 3개의 이상점을 제외해야 할 필요성이 있어 보인다.

### 리그(league)

```{r}
str(mlbp$league)
mlbp$league <- as.factor(mlbp$league)
league <- table(mlbp$league)
label <- paste(names(league), ':', league)
pie(league, labels=label)
```

새로 추가한 league 변수는 각 선수들이 뛰었던 리그를 의미한다. mlb의 소속팀은 총 30개로 연고지의 위치에 따라 2개의 큰 리그로 나누어 경기를 진행한다. 0='American League', 1='National League'이다. 파이차트를 보면 0, 즉 AL 소속 선수가 약간 더 많다.

### 이닝(p_formatted_ip)
```{r}
mlbp1 <- mlbp[mlbp$p_formatted_ip<100,]
mlbp2 <- mlbp[mlbp$p_formatted_ip>=100,]
ip1 <- c(nrow(mlbp1), nrow(mlbp2))
ip2 <- c("100이닝 미만", "100이닝 이상")
barplot(ip1, names=ip2) 
```

ip_formatted_ip 변수를 100이닝을 기준으로 나눠 막대그래프를 그려보면 100이닝 이상인 투수가 미세하게 더 많다. 이 때, 100이닝의 기준은 선발투수와 구원투수의 역할을 대략적으로 나누기 위함이다. 분석의 복잡함을 줄이기 위해 100이닝 미만은 구원투수, 100이닝 이상은 선발투수로 간주하기로 한다.

### 피안타율(batting_avg)
```{r}
summary(mlbp$batting_avg)
boxplot(mlbp$batting_avg)
```

batting_avg 변수는 피안타율을 뜻하는데, 이 변수의 범위는 이론적으로 0~1 사이의 값을 가진다. 하지만 안타를 단 하나도 허용하지 않거나, 무조건 안타를 허용하는 경우는 사실상 불가능에 가깝기 때문에 0과 1은 없다고 봐도 무방하다. 275명의 선수들의 2022시즌 피안타율의 평균은 약 0.242이다. 피안타율의 경우 보통 2할 초반대 이하를 기록하는 경우 수준급 선수라고 평가한다. 1사분위수인 0.220 근방이 그 기준이 될 수 있을 것이다. 또한 피안타율이 낮을수록 상대타자가 그 투수의 공을 안타로 만들기 어렵다는 의미이므로 공의 힘이나 위력과 연관이 있을 가능성이 높다.

### 피장타율(slg_percent)
```{r}
summary(mlbp$slg_percent)
boxplot(mlbp$slg_percent)
```

slg_percent 변수는 피장타율로, 장타를 허용한 비율을 나타낸다. 위의 batting_avg 변수는 안타의 종류를 구분하지 않고 허용한 안타의 비율을 구한 지표라면, slg_percent는 안타 중 2루타, 3루타, 홈런을 허용한 비율이다. 단타에 비해 2루타 이상의 장타를 허용하게 되면 실점할 확률이 높아질 것이다. 따라서 피장타율이 높은 것은 투수에게 좋지 않은 신호이다.

### 사사구율(p_bb_percent)
```{r}
summary(mlbp$p_bb_percent)
boxplot(mlbp$p_bb_percent)
```

p_bb_percent는 기록한 모든 결과 중 사사구의 비율을 나타낸다. 기록한 모든 결과는 아웃 여부와는 상관없다. 사사구율은 투수의 제구력과 깊은 연관이 있다고 볼 수 있다. 사사구를 허용하는 비율이 높은 경우 그만큼 스트라이크를 잘 던지지 못한다고 할 수 있기 때문이다. 또한, 사사구율이 높으면 출루허용률도 높아질 것이고 결국 실점할 확률도 높아질 가능성이 있다.

### 직구평균구속(fastball_avg_speed)
```{r}
summary(mlbp$fastball_avg_speed)
boxplot(mlbp$fastball_avg_speed)
```

fastball_avg_speed 변수는 말그대로 직구평균구속을 나타낸다. 275명의 직구평균구속은 약 93마일로, km로 환산 시 약 149.7km이다. 한국프로야구인 KBO리그에서 평균구속이 150km이 넘는 투수가 한해에 다섯손가락 안에 드는 것과 비교하면 mlb의 평균구속은 상당히 빠르다.

### 변화구평균회전수(breaking_avg_spin)
```{r}
summary(mlbp$breaking_avg_spin)
boxplot(mlbp$breaking_avg_spin)
```

breakind_avg_spin 변수는 변화구의 평균회전을 나타낸다. 단위는 rpm으로 계산하는데, rpm은 분당회전수를 뜻한다. 변화구는 그 종류가 많고, 투수에 따라 회전을 많이 주는 변화구를 선호하는 경우와 회전이 적은 대신 속도에 중점을 둔 변화구를 선호하는 경우가 있다. 따라서 변화구를 온전히 rpm만으로 평가하기는 무리가 있다. 하지만 변화구의 위력을 가장 잘 나타내는 대표적인 지표이기에 분석에 사용할 예정이다.    

### 초구스트라이크비율(f_strike_percent)
```{r}
summary(mlbp$f_strike_percent)
boxplot(mlbp$f_strike_percent)
```

f_strike_percent 변수는 투수의 초구스트라이크비율을 나타낸다. 여기서 초구는 각 타자에게 던진 첫 번째 공을 뜻하는데, A타자에게 초구를 던져 아웃을 잡아내고 B타자에게 다시 첫 번째 공을 던지면 그 공도 마찬가지로 초구로 기록된다. 야구에서 초구는 흔히 투수가 상대타자를 잡아낼 수 있는지의 여부를 가르는 중요한 '키' 라는 말을 많이 한다. 초구가 스트라이크냐 볼이냐에 따라 승부결과가 정해지는 경우가 많기 때문이다. 초구스트라이크비율은 투수의 제구력과도 연관이 깊다. 따라서, f_strike_percent가 높을수록 그 투수의 성적이나 평가가 좋아질 가능성이 충분히 있다.  

| <Numeric(숫자형) 변수와 다르게 integer(정수형) 변수는 기록한 수치의 개수를 있는 그대로 나타내는 경우가 많다.> 
|---|

### 피안타 수(p_total_hits)
```{r}
summary(mlbp$p_total_hits)
boxplot(mlbp$p_total_hits)
hist(mlbp$p_total_hits)
```

p_total_hits 변수는 각 투수가 허용한 안타의 개수, 즉 피안타 수를 의미한다. 이번 데이터에 포함된 투수들은 단순하게 한 시즌동안 평균 100개의 안타를 허용했다. 또한 분포를 살펴보면 미세하게 피안타 수가 늘어날수록 해당 값의 투수의 수는 줄어든다. 하지만 p_total_hits 값을 있는 그대로 해석해서는 안된다. 피안타 수는 이닝을 고려하지 않은 수치형 변수이기 때문이다. 아래의 예시를 살펴보자.
```{r}
df <- data.frame(mlbp$last_name, mlbp$p_formatted_ip, mlbp$p_total_hits, mlbp$batting_avg)
df[1:2,]
```

이 데이터는 현재 분석하고 있는 데이터의 일부를 발췌한 것으로 Wainwright 선수와 Greinke 선수의 이닝, 피안타 수 그리고 피안타율을 비교한 자료이다. p_total_hits 값을 먼저 확인해보면 Wainwright가 Greinke보다 안타를 더 많이 허용하였다. 하지만 batting_avg는 Greinke가 0.286으로 Wainwright의 0.261보다 높다. 이유는 p_formatted_ip에 있다. Wainwright가 투구한 이닝이 Greinke보다 월등히 많았기 때문에 피안타 수가 많더라도 피안타율은 상대적으로 낮을 수 있었다. 이처럼 p_total_hits는 확률이 아닌 누적된 수치변수이므로 분석 시 다른 변수를 고려할 필요성이 있다.

### 홀드(p_hold), 세이브(p_save)
```{r}
hist(mlbp$p_hold)
hist(mlbp$p_save)
```

p_hold와 p_save변수는 구원투수의 대표적인 지표이다. "셋업맨" 이라고 부르는 구원투수의 보직은 선발투수의 뒤를 이어 등판하여 리드를 지키고 내려간다. 홀드를 주로 기록하게 되는 선수가 바로 "셋업맨" 역할을 하는 선수이다. "클로저" 라고 칭하는 보직은 말그대로 경기의 마무리를 담당한다. 선발투수와 셋업맨을 비롯한 여러 구원투수들이 지켜온 리드를 승리로 마무리짓는 역할을 한다. 이 때, 세이브가 기록된다. 위의 히스토그램에서도 알 수 있듯이 홀드나 세이브를 많이 기록한 선수는 흔하지 않다. 즉, 홀드나 세이브를 많이 기록했다는 것은 각 팀에서 중요한 비중을 차지하고 있다는 의미임과 동시에 그 선수의 성적이 상위권이라는 것을 증명하는 셈이다. 
<!--- end of #2 --->  


<br/>  
<!--- #3 --->  

| <span style="color:darkblue">#3. 둘 이상의 변수들 사이의 관계에 대한 탐색적 자료분석</span> |
|---|
| 둘 이상의 변수들 사이의 관계에 대한 탐색적 자료분석을 수행하고 결과를 설명해보시오.|

<span style="color:darkred">__answer)__</span>  
<!--- You can write your answer #3 here. --->

| 이제 앞에서 개별적으로 살펴본 변수들을 이용해 둘 이상의 변수들 사이의 관계를 알아보고, 최종적으로 분석하고자 하는 p_era 변수와의 연관성, 상관관계를 파악한다. 
|---

### 피안타 수(p_total_hits)와 피안타율(battting_avg)
```{r}
plot(mlbp$batting_avg ~ mlbp$p_total_hits)
aes <- lm(mlbp$batting_avg ~ mlbp$p_total_hits)
abline(aes)
```

앞에서도 한번 설명했듯이, p_total_hits 변수 분석 시에는 이닝 변수를 고려해야 한다. batting_avg 변수는 p_total_hits 변수와 의미는 비슷하지만 이닝과 피안타 수의 비율을 나타내는 지표로, 분석 시 좀 더 용이한 부분이 있다. 두 변수의 관계를 살펴보기 위해 선형모형을 예측하고 회귀직선을 그린 결과, 미세하지만 양의 상관관계가 있음을 알 수 있다. 즉, 피안타 수의 증가가 어느정도는 피안타율의 상승에 영향을 미친다고 할 수 있다.

### 이닝(p_formatted_ip)과 피안타 수(p_total_hits)
```{r}
plot(p_total_hits ~ p_formatted_ip, mlbp, col=3)
```

이닝이 증가할수록 총 피안타의 수도 증가하는 것을 확인할 수 있다. 즉, 이닝과 피안타 수 사이에는 선형적인 관계가 있는 것 처럼 보인다. 하지만 50~100이닝 사이의 데이터가 많이 겹쳐 있어 정확한 판단이 어렵기 때문에 세부적인 분석이 필요하다.
```{r}
mlbp1 <- mlbp[mlbp$p_formatted_ip<100,]
plot(p_total_hits~p_formatted_ip, mlbp1, col=3)
```

이닝 수가 100이닝이 안되는 선수들의 데이터만으로 그래프를 다시 그려보면 뚜렷한 선형관계가 나타나지 않는다. 조건을 100이닝으로 설정한 것은 보통 100이닝 이하일 경우 전문선발투수가 아닐 가능성이 높기 때문이다. 따라서 이닝이 상대적으로 적은 구원투수들의 경우 이닝이 늘어날수록 피안타의 수도 많아진다고 보긴 힘들다.
```{r}
mlbp2 <- mlbp[mlbp$p_formatted_ip>=100,]
plot(p_total_hits~p_formatted_ip, mlbp2, col=3)
```

100이닝 이상을 기록한 선수들의 경우 확실한 선형관계가 보인다. 따라서 선발투수의 경우 많은 이닝을 투구한 선수들이 피안타도 많이 맞는다는 추측은 어느정도 적절하다고 볼 수 있다.

| 본격적으로 p_era 변수와의 상관관계, 연관성을 분석한다.
|---

### 평균자책점(p_era)과 이닝(p_formatted_ip)
```{r}
plot(p_era ~ p_formatted_ip, mlbp, pch=16, col=5)
mlbp1 <- mlbp[mlbp$p_formatted_ip<100,]
plot(p_era ~ p_formatted_ip, mlbp1, pch=16, col=6)
mlbp2 <- mlbp[mlbp$p_formatted_ip>100,]
plot(p_era ~ p_formatted_ip, mlbp2, pch=16, col=7)
```

p_era와 p_formatted_ip는 전체 구간에서도, 100이닝 이하에서도, 100이닝 이상에서도 뚜렷한 상관관계가 보이지 않는다. 따라서, 이닝은 투수의 평균자책점에 미치는 영향이 크게 없다고 판단할 수 있다.

### 평균자책점(p_era)과 피안타 수(p_total_hits)
```{r}
plot(p_era ~ p_total_hits, mlbp, pch=16, col=1)
plot(p_era ~ p_total_hits, mlbp1, pch=16, col=1)
plot(p_era ~ p_total_hits, mlbp2, pch=16, col=1)
```

p_era와 p_total_hits의 산점도에서 유의미한 결과가 나타나는 조건은 두 번째 산점도, 즉 100이닝 이하를 기록한 선수들의 경우이다. 이를 통해, 상대적으로 짧은 이닝을 투구하는 구원투수들의 피안타 수가 평균자책점과 연관성이 있다고 추측할 수 있다. 

### 평균자책점(p_era)과 피안타율(batting_avg)
```{r}
plot(p_era ~ batting_avg, mlbp, pch=16, col=2)
plot(p_era ~ batting_avg, mlbp1, pch=16, col=2)
plot(p_era ~ batting_avg, mlbp2, pch=16, col=2)
```

이번에는 앞서 설명한 피안타 수와 의미는 비슷하지만 분석에 좀 더 용이한 batting_avg 변수를 p_era와 비교하였다. 그 결과, 3가지 경우 모두 피안타율이 증가할수록 평균자책점도 상승하는 양상을 보인다. 따라서, batting_avg는 p_era와 강한 양의 상관관계를 가진다고 할 수 있다. 

### 평균자책점(p_era)과 피장타율(slg_percent)
```{r}
plot(p_era ~ slg_percent, mlbp, pch=16, col=3)
plot(p_era ~ slg_percent, mlbp1, pch=16, col=3)
plot(p_era ~ slg_percent, mlbp2, pch=16, col=3)
```

개별 변수에 대한 설명 당시, slg_percent 값이 높으면 실점할 확률이 급격히 상승할 것으로 예측했다. 그래프로 살펴본 결과, 예측대로 피장타율이 증가하면 평균자책점도 상승했다. 따라서, slg_percent 변수도 p_era와 강한 양의 상관관계가 있어 보인다.

### 평균자책점(p_era)과 사사구율(p_bb_percent)
```{r}
plot(p_era ~ p_bb_percent, mlbp, pch=16, col=4)
plot(p_era ~ p_bb_percent, mlbp1, pch=16, col=4)
plot(p_era ~ p_bb_percent, mlbp2, pch=16, col=4)
```

사사구율 또한 투수의 제구력과 연관이 있어 실점에 영향을 미칠 것으로 예측했다. 하지만 선발투수와 구원투수로 기준을 나누어 살펴보아도 평균자책점과의 상관관계가 뚜렷하게 보이지 않는다. 따라서, p_bb_percent는 p_era를 결정하는 유의한 변수라고 볼 수 없다.

### 평균자책점(p_era)과 초구스트라이크비율(f_strike_percent)
```{r}
plot(p_era ~ f_strike_percent, mlbp, pch=16, col=5)
plot(p_era ~ f_strike_percent, mlbp1, pch=16, col=5)
plot(p_era ~ f_strike_percent, mlbp2, pch=16, col=5)
```

초구스트라이크비율을 나타내는 f_strike_percent와 p_era간의 관계를 살펴보면, 모든 구간에서 상관관계가 보이지 않는다. 따라서, f_strike_percent는 p_era에 영향을 주지 않는다고 판단할 수 있다.

### 평균자책점(p_era)과 직구평균구속(fastball_avg_speed)
```{r}
plot(p_era ~ fastball_avg_speed, mlbp, pch=16, col=6)
plot(p_era ~ fastball_avg_speed, mlbp1, pch=16, col=6)
plot(p_era ~ fastball_avg_speed, mlbp2, pch=16, col=6)
```

fastball_avg_speed는 직구계열의 평균 구속을 의미한다. 여기서 직구계열이라 함은 크게 포심패스트볼, 투심패스트볼, 컷패스트볼(커터), 싱킹패스트볼(싱커) 4가지 종류로 나눌 수 있다. 직구의 구속은 통상적으로 투수의 구위와 연관성이 있다고 보는데, 평균 구속이 높다는 것은 그만큼 타자가 공을 맞추기 어려울 가능성이 높다. fastball_avg_speed와 p_era의 관계를 살펴보면, 상관관계가 뚜렷하게 보이지 않는다. 하지만, 평균구속이 높을수록 미세하게 점들이 우하향하는 것처럼 보인다. 따라서 평균구속의 범위를 나누어 추가적으로 분석하고자 한다.
```{r}
f1 <- mlbp[mlbp$fastball_avg_speed<90,]
f2 <- mlbp[mlbp$fastball_avg_speed>=90 & mlbp$fastball_avg_speed<95, ]
f3 <- mlbp[mlbp$fastball_avg_speed>=95, ]
m_p_era <- c(mean(f1$p_era), mean(f2$p_era), mean(f3$p_era))
barplot(m_p_era, names.arg=c("~90", "90~95", "95~"), xlab="fastball_avg_speed", ylab="p_era의 평균")
```

평균구속을 "90마일 미만", "90~95마일", "95마일 이상" 세 그룹으로 나누어 각 그룹의 p_era의 평균을 구해보았다. 그 결과, 평균구속이 높은 그룹의 p_era의 평균이 낮아지는 것을 확인할 수 있다. 첫 번째 plot에서 예측했던 평균구속 증가에 따른 p_era의 미세한 우하향이 어느정도 증명되는 부분이다.

### 평균자책점(p_era)과 변화구평균회전수(breaking_avg_spin)
```{r}
plot(p_era ~ breaking_avg_spin, mlbp, pch=16, col=7)
plot(p_era ~ breaking_avg_spin, mlbp1, pch=16, col=7)
plot(p_era ~ breaking_avg_spin, mlbp2, pch=16, col=7)
```

breaking_avg_spin 변수는 변화구의 평균 회전을 의미한다. 직구와 달리 변화구는 그 종류가 많고, 각각의 변화구마다 속도의 차이가 심해 구속을 기준으로 하지 않았다. 변화구의 회전과 p_era의 관계를 살펴보면, 직구의 구속과 마찬가지로 크게 관계가 없어 보인다. 
```{r}
b1 <- mlbp[mlbp$breaking_avg_spin<2200,]
b2 <- mlbp[mlbp$breaking_avg_spin>=2200 & mlbp$breaking_avg_spin<2600, ]
b3 <- mlbp[mlbp$breaking_avg_spin>=2600, ]
m_p_era1 <- c(mean(b1$p_era), mean(b2$p_era), mean(b3$p_era))
barplot(m_p_era, names.arg=c("~2200", "2200~2600", "2600~"), xlab="breaking_avg_spin", ylab="p_era의 평균")
```

breaking_avg_spin 변수도 그룹을 나누어 추가로 분석한 결과, plot에서는 명확히 나타나지 않았지만 변화구의 회전이 많은 그룹에서의 p_era 평균이 낮게 측정되었다. 따라서, breaking_avg_spin이 p_era과 아무 연관이 없다고는 할 수 없어보인다.   
### 평균자책점(p_era)과 리그(league)
```{r}
l0 <- mlbp[mlbp$league=="0",]
l1 <- mlbp[mlbp$league=="1",]
l_era <- c(mean(l0$p_era), mean(l1$p_era))
l_name <- c("AL 소속", "NL 소속")
b <- barplot(l_era, names=l_name, ylim=c(0,4), xlab="league", ylab="p_era 평균")
text(b, l_era, labels = round(l_era, 2))
```

league 변수에 따른 p_era의 차이를 알아보기 위해 barplot을 그려보면, AL 소속 투수들의 평균자책점의 평균이 NL 소속 투수들의 평균보다 약 0.05 낮다. 그러나 평균자책점 0.05의 차이에 큰 의미를 두긴 힘들다. 따라서 league와 p_era간의 상관관계는 없다고 판단한다.

### 평균자책점(p_era)과 홀드(p_hold), 세이브(p_save)
```{r}
h1 <- mlbp[mlbp$p_hold<=10,]
h2 <- mlbp[mlbp$p_hold>10 & mlbp$p_hold<=20 ,]
h3 <- mlbp[mlbp$p_hold>20 & mlbp$p_hold<=30 ,]
h4 <- mlbp[mlbp$p_hold>30,]
era1 <- mean(mlbp$p_era)
era2 <- mean(h1$p_era)
era3 <- mean(h2$p_era)
era4 <- mean(h3$p_era)
era5 <- mean(h4$p_era)
hold <- c("전체", "~10", "10~20", "20~30", "30~")
eram <- c(era1, era2, era3, era4, era5)
df <- data.frame(hold, eram)
barplot(eram, names=hold, xlab="p_hold", ylab="p_era 평균")
```

p_hold의 수를 10개 단위로 barplot을 그려본 결과 홀드가 10개 이하인 경우 평균자책점이 p_era의 전체 평균보다도 높았지만, 홀드가 늘어날 수록 평균자책점은 낮아지는 것을 확인할 수 있다. 홀드는 구원투수의 주요 수치 중 하나로 홀드가 많다는 것은 그만큼 팀에서 중요한 위치에 있다는 것을 의미한다. 일명 '필승조' 라고 불리는 경우가 많다. 따라서 홀드의 증가는 평균자책점이 낮아지는 데 어느정도는 영향을 미친다고 판단할 수 있다.
```{r}
s1 <- mlbp[mlbp$p_save<=10,]
s2 <- mlbp[mlbp$p_save>10 & mlbp$p_save<=20 ,]
s3 <- mlbp[mlbp$p_save>20,]
era1 <- mean(mlbp$p_era)
era7 <- mean(s1$p_era)
era8 <- mean(s2$p_era)
era9 <- mean(s3$p_era)
save <- c("전체", "~10", "10~20", "20~")
eram1 <- c(era1, era7, era8, era9)
df1 <- data.frame(save, eram1)
barplot(eram1, names=save, xlab="p_save", ylab="p_era 평균")
```

p_save도 마찬가지로 세이브 수가 늘어날 수록 미세하지만 p_era가 낮아진다. 한시즌에 20세이브 이상을 기록한 선수는 각 팀의 주전 마무리 투수인 경우가 대부분이다. 따라서 세이브가 많은 투수는 주전 마무리 투수이고, 주전 마무리 투수는 평균자책점이 낮을 확률이 높다.

<!--- end of #3 --->  


<br/>  
<!--- #4 --->  

| <span style="color:darkblue">#4. 탐색적 자료분석 결과 및 시사점</span> |
|---|
| 지금까지의 탐색적 자료분석 결과를 정리하고 앞으로의 통계분석 방향에 내용들을 제시해보시오. |

<span style="color:darkred">__answer)__</span>  
<!--- You can write your answer #4 here. --->

### p_era와 나머지 변수들의 상관계수
* 지금까지 분석한 내용을 토대로 p_era와 변수들간의 상관계수를 구해 연관이 있다고 판단한 변수들 중에서도 어떤 변수가 가장 상관관계가 강한지 살펴본다. 상관계수는 -1에서 1사이의 값을 가지며, -1에 가까울수록 음의 상관관계, 1에 가까울수록 양의 상관관계, 그리고 0에 가까울수록 선형적인 관계가 없음을 나타낸다.

```{r}
cor(mlbp[c('p_era','p_total_hits','batting_avg', 'slg_percent')])
cor(mlbp1[c('p_era', 'p_total_hits')])
```

위 변수들은 값이 증가할 때, p_era도 같이 증가하는 양의 상관관계를 가지는 변수들이다. p_total hits 변수의 경우 앞서 100이닝 이하를 기록한 구원투수에 대해서만 뚜렷한 상관관계가 나타나는 것으로 확인되어 데이터를 mlbp1으로 변경하였다. 이 중에는 slg_percent 변수의 상관계수가 약 0.83으로 p_era에 가장 큰 영향을 미치는 것으로 확인된다. 
```{r}
cor(mlbp[c('p_era', 'fastball_avg_speed','breaking_avg_spin','p_hold',
'p_save')], use="complete.obs")

```

이번에는 p_era와 음의 상관관계를 가지는 변수들의 상관계수를 나타낸 표이다. breaking_avg_spin 데이터에 결측값이 존재해 complete.obs 옵션을 통해 결측값을 제거하고 상관계수를 계산하였다. 음의 상관계수를 가지는 변수들 중에서는 p_hold와 p_save 변수가 약 -0.25로 가장 큰 상관관계를 가진다.
```{r}
mlbp$league <- as.numeric(mlbp$league)
cor(mlbp[c('p_era','p_formatted_ip','p_bb_percent','f_strike_percent','league')])
```

이 변수들은 앞서 진행한 p_era와의 상관관계 분석에서 뚜렷한 관계가 보이지 않아 연관성이 없다고 판단한 변수들이다. league변수를 제외하면 약하게나마 상관관계가 있다는 결과가 나온다. 

### 종합적인 결과

시각화와 상관계수를 통한 p_era와 나머지 변수들의 상관관계를 분석한 죄종결과, slg_percent 변수가 p_era가 높아지는 데 가장 큰 영향을 미쳤다. 즉, 피장타율이 평균자책점 상승의 가장 큰 원인이었다. batting_avg, 피안타율도 만만치않게 p_era에 영향을 미쳤다. 그 외에도 p_total_hits, p_bb_percent가 p_era의 증가와 상관관계가 있었다.
반면, p_hold와 p_save변수는 p_era와 가장 큰 음의 상관관계를 보였으며, fastball_avg_speed, breaking_avg_spin, p_formatted_ip, f_strike_percent 등의 변수도 p_era의 감소에 영향이 있다는 결과가 나왔다.
종합적으로, p_era는 그 값이 작을수록 좋은 지표이다. 따라서 p_era와 양의 상관관계를 가지는 변수는 그 값이 증가할수록 p_era에 부정적인 영향을 미침을 뜻하고, 음의 상관관계를 가지는 변수는 p_era에 긍정적인 영향을 미친다는 의미가 있다. **결국 평균자책점을 낮추기 위해서는 피장타율을 비롯한 피안타율, 피안타 수, 사사구 허용 등을 줄여야 한다. 또한 구원투수에게 주로 해당하는 홀드와 세이브는 많이 기록할수록, 전체적으로는 직구평균구속, 변화구평균회전수, 초구스트라이크비율 등이 높을수록 평균자책점이 낮아질 가능성이 높다.**

### 한계점

**가장 큰 한계점은 한정적인 데이터이다.** 다른 종목도 마찬가지지만 야구라는 스포츠가 1년만 하는 것이 아님에도 불구하고 분석의 원활함을 위해 데이터를 일부만 선택해야 했다. 또한 선택한 1년치의 데이터 안에서도 데이터의 양을 조절하기 위해 규정이닝의 50% 이상의 선수들만을 추렸다. 나아가 p_era와 관련이 있다고 예상되는 변수를 주관적인 기준으로 골라 분석을 진행하였다. 이러한 데이터의 한정적인 부분은 무엇보다도 분석의 정확성을 떨어뜨린다. 앞서 시각화를 통한 분석과 상관계수로 구한 분석의 결과값에 차이가 발생한 것도 데이터의 양이 적었기 때문이었을 것으로 추정된다. 

### 보완점 및 향후 분석방향

**우선 데이터의 양을 늘려야한다.** 공식적으로 메이저리그가 시작한 1901년부터 현재까지 수많은 경기기록과 선수들의 데이터가 존재할 것이다. 프로그램이 감당할 수 있는 최대한의 데이터를 활용하는 것이 중요하다. 또한 아무리 세세하고 다양한 기록이 존재하고 생겨난다지만 하나의 변수가 특정상황을 완벽하게 설명하는 것은 불가능하다. **따라서 지금까지의 데이터로 예측모델을 완성했어도 다음, 그다음해의 기록, 상황과 비교하여 끊임없는 피드팩과 수정을 향후 분석방향에 필수적으로 포함시킬 필요가 있다.**
<!--- end of #4 --->  


<br/>
