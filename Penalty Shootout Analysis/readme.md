# Analyzing the Correlation Between Penalty Shootout Success and Player Statistical Data

* This is the code for paper submitted in 2024 KSII (Korean Society for Internet Information) Fall Conference.
   
### Abstract

This study proposes a data-driven approach to increase the probability of success of a penalty shootout in a soccer game. If the game is not decided even after extra time, the penalty shootout, which is introduced, is an important factor in determining the outcome of the game, and it is known that the kicker's probability of goal success is usually over 70%. However, goal success is not always guaranteed, and often failures occur. In this study, based on the penalty shootout data accumulated around the world, we present a model that can classify success or failure using various statistics and information of kickers. Through techniques such as logistic regression, random forest, LightGBM, and artificial neural network, the correlation between variables and the importance of variables are measured, and the factors that have the greatest influence on the success of the shootout are derived. The results of this analysis suggest the applicability of prediction models through data analysis not only in soccer matches but also in various sports events, and will lay the foundation for contributing to improving game results.

### Introduction

 International competitions such as the World Cup are competitions where people from all over the world become one every four years, and people from each country eagerly support their national players and immerse themselves in the games. Soccer has two consequences: victory and defeat, which have a great emotional effect not only on the players playing but also on the fans watching it. In particular, a penalty shootout is an important factor in determining the outcome of a game, creating a moment that focuses everyone's attention with a single kick [1]. While it may seem simple in theory to score a goal from a distance of 11 meters, it is by no means easy considering psychological factors. However, it is very difficult to quantify psychological factors because they vary widely from individual to individual and are subjective.
 Excluding these psychological factors, this study aims to analyze the penalty shootout success rate using the numerical official data [2]. This should be objective data that can be referenced when a coach and a coach decide the order of the penalty shootout and assign players in an actual game. Through this study, we intend to understand the relationship between various data and the success of the penalty shootout, and derive key data that can contribute to increasing the probability of success. Ultimately, this analysis aims to help strategic decision-making to win the game [3].

### Dataset

 The dataset used in this study includes penalty shootout data collected from the five major European leagues, UEFA club competitions, and major national competitions. The leagues analyzed are the English Premier League (EPL), Spain La Liga (La Liga), Italy Serie A (Serie A), Germany's Bundesliga (Bundesliga), and France League 1 (League 1) and used penalty shootout data from cup competitions in these leagues. It also included data from the Champions League, Europa League, and Europa Conference League organized by UEFA, and data from the last three tournaments, including the World Cup, Europe, and Copa America, were used as national competitions.
 Data were collected by Transfermarkt and Whoscored, and 20,624 data from a total of 262 games, including statistics such as age, position, and number of points of each player, were used for analysis. VIF (Variance Expansion Index) analysis was conducted in advance to review the multicollinearity of independent variables to be used for analysis, and as a result, it was confirmed that the VIF indices were all less than 5, which does not significantly affect the model.

| 변수 명                | 변수 설명                                              |
|-----------------------|-----------------------------------------------------|
| Name                  | 대회명                                                 |
| Season                | 해당 대회의 개최 시즌                                    |
| Team                  | 해당 경기에 참가한 팀                                    |
| Player                | 승부차기를 진행한 선수명                                  |
| Position              | 해당 선수의 포지션                                       |
| Age                   | 해당 선수의 당시 나이                                     |
| Goals                 | 해당 선수의 당시까지의 통산 골 수                            |
| Order                 | 해당 선수의 승부차기 순서                                  |
| Last                  | 해당 선수의 순서가 마지막이었는지 여부                        |
| Period(Club)          | 해당 클럽에 소속되어 있던 기간 (년차)                        |
| Squad(International)  | 해당 선수가 국가대표팀에 소속되어 있던 기간 (등록 경기 수)        |
| Score                 | 해당 선수의 승부차기 성공/실패 여부                          |

### Method

In this experiment, a total of five models are learned to classify and predict the 'score' variables of the PA dataset. The models used are logistic regression, GLM (Generalized Linear Model), random forest, Light Gradient Boosting Machine (LGBM), and MLP (Multi-layer Perceptron, Hidden layer : 4). These are all models suitable for classification problems, and the performance of each model is evaluated through four evaluation indicators: accuracy, F1-Score, ROC-AUC, and MCC.
F1-Score is a harmonized average of Precision and Recall (recall) and is a particularly useful performance indicator on unbalanced datasets. In addition, MCC evaluates predictive performance by considering all of True Positive, True Negative, False Positive, and False Negative, and has values from -1 to 1 as an indicator robust to class imbalance [4]. MCC is calculated as follows. 

![image](https://github.com/user-attachments/assets/db02fef4-da1e-4650-9de2-7d80c184da77)

Additionally, logistic regression and GLM analyze the effect of independent variables on dependent variables through regression coefficients after learning and P-value values for each variable, and random forest and LightGBM evaluate the importance of independent variables through feature importance [5]. In the case of artificial neural networks, performance is evaluated by applying only basic evaluation indicators.

### Result1 (Club)

|     Model     |  Acc  |  F1-score  |  Roc_Auc  |  MCC   |
|---------------|-------|------------|-----------|--------|
| LR            | 0.591 | **0.438**      | **0.615**     | **0.196**  |
| GLM           | 0.561 | 0.401      | 0.576     | 0.131  |
| RF            | **0.674** | 0.330      | 0.558     | 0.115  |
| LGBM          | 0.627 | 0.311      | 0.532     | 0.060  |
| MLP           | 0.580 | 0.352      | 0.559     | 0.129  |

|     Model     |   Age   |   Goals   |  Period  |  Order  |  Last   |
|---------------|---------|-----------|----------|---------|---------|
| LR            | -0.001  | -0.008*   | -0.017   | -0.101* |  0.795* |
| GLM           |  0.013  | -0.466*   | -0.089   | -0.136* |  0.959* |

(* : P < 0.05)

![image](https://github.com/user-attachments/assets/c186dfcb-8caf-4d25-8a51-061418815572)

 In this experiment, it's the 5th League Cup and UEFA Club competition
 We evaluated the performance of five models using data. In terms of accuracy, the random forest (RF) model showed the best performance, and the logistic regression (LR) model showed the best performance in the indicators of F1-Score, ROC-AUC, and MCC. In the LR and GLM models, the 'last' variable showed a high regression coefficient, indicating a tendency to decrease the probability of success when the kick order was the last. On the other hand, the variables 'order', 'goals', and 'period' showed negative correlations with the probability of success. In addition, when looking at the feature importance, the 'goals', 'age', and 'period' variables were found to have the greatest influence on the success or failure of the penalty shootout in the random forest (RF) and LightGBM models.

### Result2 (International)

|     Model     |  Acc   |  F1-score  |  Roc_Auc  |  MCC   |
|---------------|--------|------------|-----------|--------|
| LR            |  0.546 |  0.355     |  0.515    |  0.026 |
| GLM           |  0.553 |  0.358     |  0.519    |  0.035 |
| RF            |  **0.592** |  0.225  |  0.476    | -0.050 |
| LGBM          |  0.559 |  0.130     |  0.425    | -0.162 |
| MLP           |  0.471 |  **0.422** | **0.545** |  **0.089** |

| Model |   Age   |   Goals   |  Squad  |  Order  |  Last   |
|-------|---------|-----------|---------|---------|---------|
| LR    |  0.080* |  -0.600*  | -0.008* | -0.037* |  0.767* |
| GLM   |  0.080* |  -0.600*  | -0.008* | -0.037* |  0.767* |

(* : P < 0.05)

![image](https://github.com/user-attachments/assets/5930b211-2f3a-4f6d-ab57-113d1f72ba39)

 As a result of model training on national competition data, the overall performance was lower than that learned with club competition data. This seems to be a performance degradation caused by a relatively small number of samples of national competition data, and performance improvement can be expected by collecting more data in future studies.
 As a result of the regression coefficient analysis, similar to the previous club competition, the probability of failure was high in the case of the last order, and the total number of goals and kick order showed a negative correlation. However, in the variable importance analysis, the importance was high in the order of 'squad', 'goals', and 'age', and in particular, 'squad' was the most important variable. This is the difference from the results of the previous club competition data, suggesting that the experience as a national team has a great influence on the outcome of the penalty shootout.

### Conclusion

 In this paper, by analyzing the data of the players in the penalty shootout in various models, the accuracy of up to about 67% was achieved, and through correlation analysis, key variables that influence success or failure could be identified. It is significant that there is room for performance improvement through further research and that meaningful insights have been derived in the field of sports where little research has been done.
 This study presented the possibility of data-based decision-making and laid the foundation for use in other sports, and proposed a model suitable for penalty shootout prediction by comparing various models. It also makes an important contribution in that it has the potential to develop into a practical tool that can be used by coaches and coaching staff in actual games.
 If learning and analysis are conducted using more data and advanced models through future research, the possibility of performance improvement is sufficient. This will be of practical help to the decision-making of the director and coaching staff in the field.

### Reference

[1] Horn, M., de Waal, S., & Kraak, W. “In-match penalty kick analysis of the 2009/10 to 2018/19 English Premier League competition”. International Journal of Performance Analysis in Sport, 21(1), 139-155, 2021.

[2] Pinheiro, G. D. S., Nascimento, V. B., Dicks, M., Costa, V. T., & Lames, M. “Design and validation of an observational system for penalty kick analysis in football (OSPAF)”. Frontiers in Psychology, 12, 661179, 2021.

[3] Pinheiro, P., & Cavique, L., “A bi-objective procedure to deliver actionable knowledge in sport services”, Expert Systems, 37(6), e12617, 2020.

[4] Chicco, D., & Jurman, G. “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation”. BMC genomics, 21, 1-13, 2020.

[5] Rajbahadur, G. K., Wang, S., Oliva, G. A., Kamei, Y., & Hassan, A. E. “The impact of feature importance methods on the interpretation of defect classifiers”. IEEE Transactions on Software Engineering, 48(7), 2245-2261, 2021.
