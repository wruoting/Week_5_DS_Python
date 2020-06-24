import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import deque
from assignment_5_wang_utils import print_confusion_matrix, transform_trading_days_to_trading_weeks, \
    make_trade, trading_strategy

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    scaler = StandardScaler()
    X_2018 = df_2018[['mean_return', 'volatility']].values
    # Need to scale the training data
    X_2018_Scaled = scaler.fit_transform(X_2018)
    Y_2018 = df_2018[['Classification']].values

    X_2019 = df_2019[['mean_return', 'volatility']].values
    X_2019_Scaled = scaler.fit_transform(X_2019)
    Y_2019 = df_2019[['Classification']].values

    # We will map GREEN to 1 and RED to 0
    logisticRegression = LogisticRegression()
    logisticRegression.fit(X_2018_Scaled, Y_2018.ravel())
    coefficients = logisticRegression.coef_[0]
    print('\nQuestion 1')
    print('Coefficients are: {}'.format(coefficients))
    print('Equation is: Y = {} * mean_return + {} * volatility'.format(coefficients[0], coefficients[1]))
    print('\nQuestion 2')
    predict_2019 = logisticRegression.predict(X_2019_Scaled)
    error_rate =  np.round(np.multiply(np.mean(predict_2019 != Y_2019.T), 100), 2)
    accuracy = 100 - error_rate
    print("Accuracy is: {}%".format(accuracy))
    print('\nQuestion 3')
    confusion_matrix_array = confusion_matrix(Y_2019, predict_2019)
    confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
    print(confusion_matrix_df)
    print('\nQuestion 4')
    print_confusion_matrix(Y_2019, confusion_matrix_df)
    print('\nQuestion 5')
    # Import the CSV necessary for 2019 data
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    trading_weeks_2019.reset_index(inplace=True)
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels", predict_2019, allow_duplicates=True)
    print('Trading strategy was based on the one created in Assignment 3')
    print('With logistic regression:')
    predicted_trading_df = trading_strategy(trading_weeks_2019, 'Predicted Labels')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('With buy and hold:')
    predicted_trading_df = trading_strategy(trading_weeks_2019, 'Buy and Hold')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('Buy and hold has the greater profits at the end of the year. Logistic regression seems to ')
    print('mark more RED weeks over green weeks, and does not accurately train on true positives.')

if __name__ == "__main__":
    main()