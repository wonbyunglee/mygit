# Analyzing the Correlation Between Penalty Shootout Success and Player Statistical Data

### Abstract
This study proposes a data-driven approach to increase the probability of success of a penalty shootout in a soccer game. If the game is not decided even after extra time, the penalty shootout, which is introduced, is an important factor in determining the outcome of the game, and it is known that the kicker's probability of goal success is usually over 70%. However, goal success is not always guaranteed, and often failures occur. In this study, based on the penalty shootout data accumulated around the world, we present a model that can classify success or failure using various statistics and information of kickers. Through techniques such as logistic regression, random forest, LightGBM, and artificial neural network, the correlation between variables and the importance of variables are measured, and the factors that have the greatest influence on the success of the shootout are derived. The results of this analysis suggest the applicability of prediction models through data analysis not only in soccer matches but also in various sports events, and will lay the foundation for contributing to improving game results.

### Dataset
The dataset used in this study includes penalty shootout data collected from the five major European leagues, UEFA club competitions, and major national competitions. The leagues analyzed are the English Premier League (EPL), Spain La Liga (La Liga), Italy Serie A (Serie A), Germany's Bundesliga (Bundesliga), and France League 1 (League 1) and used penalty shootout data from cup competitions in these leagues. It also included data from the Champions League, Europa League, and Europa Conference League organized by UEFA, and data from the last three tournaments, including the World Cup, Europe, and Copa America, were used as national competitions.
Data were collected by Transfermarkt and Whoscored, and 20,624 data from a total of 262 games, including statistics such as age, position, and number of points of each player, were used for analysis. VIF (Variance Expansion Index) analysis was conducted in advance to review the multicollinearity of independent variables to be used for analysis, and as a result, it was confirmed that the VIF indices were all less than 5, which does not significantly affect the model.

변수 명
변수 설명
Name
대회명
Season
해당 대회의 개최시즌
Team
해당경기에 참가한 팀명
Player
승부차기를 진행한 선수명
Position
해당 선수의 주포지션
Age
해당 선수의 당시 나이
Goals
해당 선수의 당시까지의 통산 골 수
Order
해당 선수의 승부차기 순서
Last
해당 선수의 순서가 마지막이었는지 여부
Period(Club)
해당 선수가 해당 클럽에 소속되어 있던 기간 (년차)
Squad(International)
해당 선수가 국가대표팀에 소속되어 있던 기간 (등록 경기 수)
Score
해당 선수의 승부차기 성공/실패 여부
