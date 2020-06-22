from assignment_5_wang_custom_knn_class import Custom_knn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import deque

def print_confusion_matrix(Y_2019, confusion_matrix_df):
    '''
    Y_2019: input vector for the confusion matrix
    confusion_matrix_df: the input confusion df
    '''
    total_data_points = len(Y_2019)
    true_positive_number = confusion_matrix_df['Predicted: GREEN']['Actual: GREEN']
    true_positive_rate = np.round(np.multiply(np.divide(true_positive_number, total_data_points), 100), 2)
    true_negative_number = confusion_matrix_df['Predicted: RED']['Actual: RED']
    true_negative_rate = np.round(np.multiply(np.divide(true_negative_number, total_data_points), 100), 2)
    print("True positive rate: {}%".format(true_positive_rate))
    print("True negative rate: {}%".format(true_negative_rate))

def transform_trading_days_to_trading_weeks(df):
    '''
    df: dataframe of relevant data
    returns: dataframe with processed data, only keeping weeks, their open and close for said week
    '''
    trading_list = deque()
    # Iterate through each trading week
    for trading_week, df_trading_week in df.groupby(['Year','Week_Number']):
        classification =  df_trading_week.iloc[0][['Classification']].values[0]
        opening_day_of_week = df_trading_week.iloc[0][['Open']].values[0]
        closing_day_of_week = df_trading_week.iloc[-1][['Close']].values[0]
        trading_list.append([trading_week[0], trading_week[1], opening_day_of_week, closing_day_of_week, classification])
    trading_list_df = pd.DataFrame(np.array(trading_list), columns=['Year', 'Trading Week', 'Week Open', 'Week Close', 'Classification'])
    return trading_list_df

def make_trade(cash, open, close):
    '''
    cash: float of cash on hand
    open: float of open price
    close: float of close price
    returns: The cash made from a long position from open to close
    '''
    shares = np.divide(cash, open)
    return np.multiply(shares, close)

def trading_strategy(trading_df, prediction_label, weekly_balance=100):
    '''
    trading_df: dataframe of relevant weekly data
    prediction_label: the label for which we're going to trade off of
    returns: A df of trades made based on Predicted Labels
    '''
    # The weekly balance we will be using
    weekly_balance_acc = weekly_balance
    trading_history = deque()
    index = 0
    
    while(index < len(trading_df.index) - 1):
        trading_week_index = index
        if weekly_balance_acc != 0:
            # Find the next consecutive green set of weeks and trade on them
            while(trading_week_index < len(trading_df.index) - 1 and trading_df.iloc[trading_week_index][[prediction_label]].values[0] == 'GREEN'):
                trading_week_index += 1
            green_weeks = trading_df.iloc[index:trading_week_index][['Week Open', 'Week Close']]
            # Check if there are green weeks, and if there are not, we add a row for trading history
            if len(green_weeks.index) > 0:
                # Buy shares at open and sell shares at close of week
                green_weeks_open = float(green_weeks.iloc[0][['Week Open']].values[0])
                green_weeks_close = float(green_weeks.iloc[-1][['Week Close']].values[0])
                # We append the money after we make the trade
                weekly_balance_acc = make_trade(weekly_balance_acc, green_weeks_open, green_weeks_close)
            # Regardless of whether we made a trade or not, we append the weekly cash and week over
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                weekly_balance_acc])
        else:
            # If we have no money we will not be able to trade
            trading_history.append([trading_df.iloc[trading_week_index][['Year']].values[0],
                    trading_df.iloc[trading_week_index][['Trading Week']].values[0],
                    weekly_balance_acc])
        index = trading_week_index+1
    trading_hist_df = pd.DataFrame(np.array(trading_history), columns=['Year', 'Trading Week', 'Balance'])
    trading_hist_df['Balance'] = np.round(trading_hist_df[['Balance']].astype(float), 2)

    return trading_hist_df

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]

    print('\nQuestion 1')
    X_2018 = df_2018[['mean_return', 'volatility']].values
    Y_2018 = df_2018[['Classification']].values

    error_rate_custom = {}
    error_rate = {}
    # The highest accuracy from our knn classifiers was k = 9
    for p in [1, 1.5, 2]:
        X_train, X_test, Y_train, Y_test = train_test_split(X_2018, Y_2018, test_size=0.6, random_state=3)
        # Custom Classifier
        knn_custom_classifier = Custom_knn(number_neighbors_k=9, distance_parameter_p=p)
        knn_custom_classifier.fit(X_train, Y_train.ravel())
        prediction_custom = knn_custom_classifier.predict(X_test)
        # As a percentage
        error_rate_custom[p] = np.round(np.multiply(np.mean(prediction_custom != Y_test), 100), 2)

        # This is to validate that we are getting the same error rate across the KNN classifier as well
        # KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=9, p=p)
        knn_classifier.fit(X_train, Y_train.ravel())
        prediction = knn_classifier.predict(X_test)
        # As a percentage
        error_rate[p] = np.round(np.multiply(np.mean(prediction != Y_test), 100), 2)
    print("Confirm that the error rate for both the custom and scipy classifiers are the same: {}".format(str(error_rate == error_rate_custom)))
    print("The error rate of the different p's are {}".format(error_rate_custom))
    plt.plot(np.fromiter(error_rate_custom.keys(), dtype=float), np.subtract(100, np.fromiter(error_rate_custom.values(), dtype=float)))
    plt.title('P value vs Accuracy - Training 2018, Testing 2018')
    plt.xlabel('P value')
    plt.ylabel('Accuracy (%)')
    plt.savefig(fname='KNN_Classifiers_Q1')
    plt.show()
    plt.close()
    print('The P value of 2 gives the best accuracy of {}%'.format(float(100-error_rate_custom[2])))
    
    print('\nQuestion 2')
    print('The question is unclear, as to which set to use as training data, but I am repeating this with year 2 and using year 1 data to train.')
    error_rate_custom = {}
    X_2019 = df_2019[['mean_return', 'volatility']].values
    Y_2019 = df_2019[['Classification']].values
    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=9, distance_parameter_p=p)
        knn_custom_classifier.fit(X_2018, Y_2018.ravel())
        prediction_custom = knn_custom_classifier.predict(X_2019)
        # As a percentage
        error_rate_custom[p] = np.round(np.multiply(np.mean(prediction_custom != Y_2019), 100), 2)
    print("The error rate of the different p's are {}".format(error_rate_custom))
    print('The P value of 1.5 and 2 give the best accuracy of {}%'.format(float(100-error_rate_custom[2])))
    plt.plot(np.fromiter(error_rate_custom.keys(), dtype=float), np.subtract(100, np.fromiter(error_rate_custom.values(), dtype=float)))
    plt.title('P value vs Accuracy - Training 2018, Testing 2019')
    plt.xlabel('P value')
    plt.ylabel('Accuracy (%)')
    plt.savefig(fname='KNN_Classifiers_Q2')
    plt.show()
    plt.close()
    print('Using 2018 data to test 2019 showed significantly lower accuracy. Changing the distance metric between Minkovski and Euclidean did ')
    print('not seem to make a difference in clustering label selection.')
    print('\nQuestion 3')
    # Train on 2018 data
    knn_custom_classifier = Custom_knn(number_neighbors_k=9, distance_parameter_p=1.5)
    knn_custom_classifier.fit(X_2018, Y_2018.ravel())
    prediction_custom = knn_custom_classifier.predict(X_2019)
    print('Labels for 2019')
    print(prediction_custom)
    # Pick two points with different labels in 2019
    # Week 1 is GREEN and Week 3 is RED
    print('Label for Week 1 is Green')
    print('The graph presented shows a majority of green local points')
    knn_custom_classifier.draw_decision_boundary(X_2019[0])
    print('Label for Week 3 is Red')
    print('The graph presented shows a majority of red local points')
    knn_custom_classifier.draw_decision_boundary(X_2019[2])
    
    print('\nQuestion 4 and Question 5')
    print('2019 is predicted with 2018 trained data.')
    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=9, distance_parameter_p=p)
        knn_custom_classifier.fit(X_2018, Y_2018.ravel())
        prediction_custom = knn_custom_classifier.predict(X_2019)
        confusion_matrix_array = confusion_matrix(Y_2019, prediction_custom)
        confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
        print('Confusion matrix for p = {}'.format(p))
        print(confusion_matrix_df)
        print_confusion_matrix(Y_2019, confusion_matrix_df)
    print('For question 5, there are some differences between the Euclidean/Minkowski and Manhattan ')
    print('distance clustering, but the large number for K probably made the distance calculation ')
    print('less impactful between p = 1.5 and p = 2. There was an accuracy jump as p increased, but ')
    print('more continuous p values need to be explored in order to see if this is true.')

    print('\nQuestion 6')
    # Import the CSV necessary for 2019 data
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    trading_weeks_2019.reset_index(inplace=True)
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=9, distance_parameter_p=p)
        knn_custom_classifier.fit(X_2018, Y_2018.ravel())
        prediction_custom = knn_custom_classifier.predict(X_2019)
        # Add columns for each of the different clustering methods
        trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Predicted Labels {}".format(p), prediction_custom, allow_duplicates=True)
    trading_weeks_2019.insert(len(trading_weeks_2019.columns), "Buy and Hold", buy_and_hold, allow_duplicates=True)
    print('Trading Strategy for 2019 for $100 starting cash:')
    print('Trading strategy was based on the one created in Assignment 3')
    print('With p = 1')
    predicted_trading_df = trading_strategy(trading_weeks_2019, 'Predicted Labels 1')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('With p = 1.5')
    predicted_trading_df = trading_strategy(trading_weeks_2019, 'Predicted Labels 1.5')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('With p = 2')
    predicted_trading_df = trading_strategy(trading_weeks_2019, 'Predicted Labels 2')
    print('${}'.format(predicted_trading_df[['Balance']].iloc[-1].values[0]))
    print('Buy and Hold')
    predicted_trading_buy_and_hold = trading_strategy(trading_weeks_2019, "Buy and Hold")
    print('${}'.format(predicted_trading_buy_and_hold[['Balance']].iloc[-1].values[0]))
    print('The best trading strategy is still buy and hold.')
if __name__ == "__main__":
    main()