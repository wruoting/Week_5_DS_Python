from assignment_5_wang_custom_knn_class import Custom_knn
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main():
    ticker='WMT'
    file_name = '{}_weekly_return_volatility.csv'.format(ticker)
    file_name_self_labels = 'WMT_Labeled_Weeks_Self.csv'

    # Read from that file for answering our questions
    df = pd.read_csv(file_name, encoding='ISO-8859-1')
    df_2018 = df[df['Year'] == 2018]
    df_2019 = df[df['Year'] == 2019]

    print('\nQuestion 1')
    X = df_2018[['mean_return', 'volatility']].values
    Y = df_2018[['Classification']].values

    error_rate = {}
    error_rate_custom = {}
    for n in range(3, 13, 2):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.6, random_state=3)
        # KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=n)
        knn_classifier.fit(X_train, Y_train.ravel())
        prediction = knn_classifier.predict(X_test)

        knn_custom_classifier = Custom_knn(number_neighbors_k=n)
        knn_custom_classifier.fit(X_train, Y_train.ravel())
        prediction_custom = knn_custom_classifier.predict(X_test)
        # As a percentage
        error_rate[n] = np.round(np.multiply(np.mean(prediction != Y_test), 100), 2)
        error_rate_custom[n] = np.round(np.multiply(np.mean(prediction_custom != Y_test), 100), 2)
    print(error_rate)
    print(error_rate_custom)

if __name__ == "__main__":
    main()