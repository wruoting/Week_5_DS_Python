from assignment_5_wang_custom_knn_class import Custom_knn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from assignment_5_wang_utils import print_confusion_matrix, transform_trading_days_to_trading_weeks, make_trade, trading_strategy

def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]
    scaler = StandardScaler()
    
    print('\nQuestion 1')
    X_2018 = df_2018[['mean_return', 'volatility']].values
    Y_2018 = df_2018[['Classification']].values
    X_2019 = df_2019[['mean_return', 'volatility']].values
    Y_2019 = df_2019[['Classification']].values

    # Need to scale the training data
    X_2018_Scaled = scaler.fit_transform(X_2018)
    X_2019_Scaled = scaler.fit_transform(X_2019)

    error_rate_custom = {}
    error_rate = {}
    # The highest accuracy from our knn classifiers was k = 5
    for p in [1, 1.5, 2]:
        X_train, X_test, Y_train, Y_test = train_test_split(X_2018_Scaled, Y_2018, test_size=0.6, random_state=3)
        # Custom Classifier
        knn_custom_classifier = Custom_knn(number_neighbors_k=5, distance_parameter_p=p)
        knn_custom_classifier.fit(X_train, Y_train.ravel())
        prediction_custom = knn_custom_classifier.predict(X_test)
        # As a percentage
        error_rate_custom[p] = np.round(np.multiply(np.mean(prediction_custom != Y_test.T), 100), 2)

        # This is to validate that we are getting the same error rate across the KNN classifier as well
        # KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5, p=p)
        knn_classifier.fit(X_train, Y_train.ravel())
        prediction = knn_classifier.predict(X_test)
        # As a percentage
        error_rate[p] = np.round(np.multiply(np.mean(prediction != Y_test.T), 100), 2)
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
    print('I am repeating this with year 2 and using year 1 data to train.')
    error_rate_custom = {}

    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=5, distance_parameter_p=p)
        knn_custom_classifier.fit(X_2018_Scaled, Y_2018.ravel())
        prediction_custom = knn_custom_classifier.predict(X_2019_Scaled)
        np.set_printoptions(threshold=np.inf)
        # As a percentage
        error_rate_custom[p] = np.round(np.multiply(np.mean(prediction_custom != Y_2019.T), 100), 2)
    print("The error rate of the different p's are {}".format(error_rate_custom))
    print('The P value of 1 and 2 give the best accuracy of {}%'.format(float(100-error_rate_custom[2])))
    plt.plot(np.fromiter(error_rate_custom.keys(), dtype=float), np.subtract(100, np.fromiter(error_rate_custom.values(), dtype=float)))
    plt.title('P value vs Accuracy - Training 2018, Testing 2019')
    plt.xlabel('P value')
    plt.ylabel('Accuracy (%)')
    plt.savefig(fname='KNN_Classifiers_Q2')
    plt.show()
    plt.close()
    print('Using 2018 data to test 2019 showed slightly higher accuracy. Changing the distance metric between Manhattan and Euclidean did ')
    print('not seem to make a difference in clustering label selection. Minkowski distance showed a slightly lower accuracy.')
    print('\nQuestion 3')
    # Train on 2018 data
    knn_custom_classifier = Custom_knn(number_neighbors_k=5, distance_parameter_p=1.5)
    knn_custom_classifier.fit(X_2018_Scaled, Y_2018.ravel())
    prediction_custom = knn_custom_classifier.predict(X_2019_Scaled)
    print('Labels for 2019')
    print(prediction_custom)
    # Pick two points with different labels in 2019
    # Week 11 is GREEN and Week 1 is RED
    print('Label for Week 11 is Green')
    print('The graph presented shows a majority of green local points')
    knn_custom_classifier.draw_decision_boundary(X_2019_Scaled[10])
    print('Label for Week 1 is Red')
    print('The graph presented shows a majority of red local points')
    knn_custom_classifier.draw_decision_boundary(X_2019_Scaled[0])
    
    print('\nQuestion 4 and Question 5')
    print('2019 is predicted with 2018 trained data.')
    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=5, distance_parameter_p=p)
        knn_custom_classifier.fit(X_2018_Scaled, Y_2018.ravel())
        prediction_custom = knn_custom_classifier.predict(X_2019_Scaled)
        confusion_matrix_array = confusion_matrix(Y_2019, prediction_custom)
        confusion_matrix_df = pd.DataFrame(confusion_matrix_array, columns= ['Predicted: GREEN', 'Predicted: RED'], index=['Actual: GREEN', 'Actual: RED'])
        print('Confusion matrix for p = {}'.format(p))
        print(confusion_matrix_df)
        print_confusion_matrix(Y_2019, confusion_matrix_df)
    print('For question 5, there are significant differences in true positives vs. true negatives. Predicted GREEN ')
    print('and actual GREEN values show almost no accuracy, which indicates that this method is not particularly good at predicting making trades.')
    print('It does however, show better accuracy for weeks to not trade. The different methods don\'t show significantly different accuracy, ')
    print('and the true positive rate remains low regardless of distance calculation.')

    print('\nQuestion 6')
    # Import the CSV necessary for 2019 data
    df = pd.read_csv(file_name_self_labels, encoding='ISO-8859-1')
    df_trading_weeks = transform_trading_days_to_trading_weeks(df)
    trading_weeks_2019 = df_trading_weeks[df_trading_weeks['Year'] == '2019']
    trading_weeks_2019.reset_index(inplace=True)
    buy_and_hold = np.full(len(trading_weeks_2019.index), 'GREEN')
    for p in [1, 1.5, 2]:
        # Train on 2018 data
        knn_custom_classifier = Custom_knn(number_neighbors_k=5, distance_parameter_p=p)
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