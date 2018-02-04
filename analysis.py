import numpy as np
import matplotlib.pyplot as plt


def perform_analysis(X_cols, Y_col, np_dataset):
    Y_col = Y_col[0]
    for X_col in X_cols:
        all_x = np_dataset[:, X_col]
        all_y = np_dataset[:, Y_col]

        mean_x = np.mean(all_x)
        mean_y = np.mean(all_y)

        print ('mean x: ', mean_x)
        print ('mean y: ', mean_y)

        x_diff = 0
        y_diff = 0
        covariance = 0
        for x, y in zip(all_x, all_y):
            x_diff += pow(x - mean_x, 2)
            y_diff += pow(y - mean_y, 2)
            covariance += ((x - mean_x) * (y - mean_y))

        x_diff /= (len(all_x) - 1)
        y_diff /= (len(all_y) - 1)
        covariance /= (len(all_y) - 1)

        print('var x: ', x_diff, np.var(all_x))
        print('var y: ', y_diff, np.var(all_y))

        print ('stddev x: ', pow(x_diff, 0.5), np.std(all_x))
        print('stddev y: ', pow(y_diff, 0.5), np.std(all_y))

        print('covariance: ', covariance, np.cov(all_x, all_y))

        pcc = covariance / (pow(x_diff, 0.5) * pow(y_diff, 0.5))

        print('pcc: ', pcc, np.corrcoef(all_x, all_y))

        plt.scatter(all_x, all_y)
        plt.show()


    from sklearn import datasets
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    # load the iris datasets
    dataset = datasets.load_iris()
    # create a base classifier used to evaluate a subset of attributes
    model = LinearRegression()
    # create the RFE model and select 3 attributes
    rfe = RFE(estimator=model, n_features_to_select=3, step=1)
    rfe = rfe.fit(np_dataset[:, X_cols], np.ravel(np_dataset[:, Y_col]))
    # summarize the selection of the attributes
    print(rfe.n_features_)
    print(rfe.support_)
    print(rfe.ranking_)