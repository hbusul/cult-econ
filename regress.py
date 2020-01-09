import pandas as pd
import numbers as np
import matplotlib.pyplot as plt
from sklearn import linear_model

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    feature_superset = [['P'], ['C'], ['N'], ['P', 'C', 'N']]
    for feature in feature_superset:
        reg = linear_model.LinearRegression()
        reg.fit(df[feature], df['GDPPC'])
        r_square = reg.score(df[feature], df['GDPPC'])
        print(
            f"Experiment {feature}, Constant {reg.intercept_}, Coefficients {reg.coef_}, R^2  {reg.score(df[feature], df['GDPPC'])}")
