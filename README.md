-----------------Classification tree-------------------------------------------------------------------------------------------------
Introduction:

See how stats like PA, AB, H, 2B, 3B, HR, RBI, SB, BA, OBP, SLG, etc influence postseason birth

Methods:

Gathering MLB regular season team stats from 2012-2019, including 

Using classification tree 

Result:

constrained tree, max_depth=100 : Test set accuracy= 0.83

unconstrained tree(entropy criterion) : Test set Accuracy = 0.8333333333333334

unconstrained tree(gini criterion) : Test set Accuracy = 0.8333333333333334

-----------------Regression tree-------------------------------------------------------------------------------------------------

Introduction:

see if plate discipline in baseball affected other stats including regular ones like BB%, K%, AVG, OBP, SLG, OPS, ISO, wOBA, wRC+, per game WAR

Methods:

Plate Discipline defined as (ZSwing%-OSwing%)/Swing%

Using Regression tree

Result:

1.Plate Discipline vs BB%

Test set RMSE of dt: 0.01
Linear Regression test set RMSE: 0.01
Regression Tree test set RMSE: 0.01

2.Plate Discipline vs K%
Test set RMSE of dt: 0.05
Linear Regression test set RMSE: 0.05
Regression Tree test set RMSE: 0.05

3.Plate Discipline vs AVG
Test set RMSE of dt: 0.02
Linear Regression test set RMSE: 0.02
Regression Tree test set RMSE: 0.02

4.Plate Discipline vs OBP
Test set RMSE of dt: 0.02
Linear Regression test set RMSE: 0.02
Regression Tree test set RMSE: 0.02

5.Plate Discipline vs ISO
Test set RMSE of dt: 0.04
Linear Regression test set RMSE: 0.04
Regression Tree test set RMSE: 0.04

6.Plate Discipline vs wOBA
Test set RMSE of dt: 0.02
Linear Regression test set RMSE: 0.02
Regression Tree test set RMSE: 0.02

7.Plate Discipline vs wRC+
Test set RMSE of dt: 15.68
Linear Regression test set RMSE: 15.32
Regression Tree test set RMSE: 15.68

8.7.Plate Discipline vs per game WAR
Test set RMSE of dt: 0.00
Linear Regression test set RMSE: 0.00
Regression Tree test set RMSE: 0.00

Right now as the result shows, there is not much difference betweeen using general linear regression and regression tree based on the same RMSE.

