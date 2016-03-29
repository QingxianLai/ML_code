import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

def main():
    #  data load and precessing
    train_data_set = pd.read_csv("data/banana_train.csv",header=None)
    test_data_set = pd.read_csv("data/banana_test.csv",header=None)

    train_data_X = train_data_set.iloc[:,1:3]
    train_data_y = train_data_set.iloc[:,0]

    test_data_X = test_data_set.iloc[:,1:3]
    test_data_y = test_data_set.iloc[:,0]

    X_mean = train_data_X.mean(axis=0)
    X_std = train_data_X.std(axis = 0)
    train_data_X = (train_data_X-X_mean)/X_std
    test_data_X = (test_data_X-X_mean)/X_std



    #
    #
    # #
    # clf = DecisionTreeClassifier(max_depth=5).fit(train_data_X,train_data_y)
    # plt.figure(figsize=(12,10))
    # plot_step = 0.01
    #
    # x_min, x_max = train_data_X.iloc[:, 0].min()-0.1, train_data_X.iloc[:, 0].max()+0.1
    # y_min, y_max = train_data_X.iloc[:, 1].min()-0.1, train_data_X.iloc[:, 1].max()+0.1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))
    #
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #
    # plt.axis("tight")
    # for i, color in [(-1,'b'),(1,'r')]:
    #     idx = np.where(train_data_y == i)
    #     plt.scatter(train_data_X.iloc[idx[0], 0], train_data_X.iloc[idx[0], 1], c=color,
    #                     cmap=plt.cm.Paired)
    #
    # plt.show()
    #
    #
    #
    # # incredible stupid plot
    #
    # max_depth = range(1,11)
    # for depth in max_depth:
    #     clf = DecisionTreeClassifier(max_depth=depth).fit(train_data_X,train_data_y)
    #     plt.subplot(2, 5, depth)
    #
    #     plot_step = 0.05
    #
    #     x_min, x_max = train_data_X.iloc[:, 0].min()-0.1, train_data_X.iloc[:, 0].max()+0.1
    #     y_min, y_max = train_data_X.iloc[:, 1].min()-0.1, train_data_X.iloc[:, 1].max()+0.1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                              np.arange(y_min, y_max, plot_step))
    #
    #     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #     cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #
    #     plt.axis("tight")
    #     for i, color in [(-1,'b'),(1,'r')]:
    #         idx = np.where(train_data_y == i)
    #         plt.scatter(train_data_X.iloc[idx[0], 0], train_data_X.iloc[idx[0], 1], c=color,
    #                         cmap=plt.cm.Paired)
    #
    # plt.show()
    #
    #
    #
    #
    # max_depth = range(1,11)
    # train_errors = []
    # test_errors = []
    # for depth in max_depth:
    #     clf = DecisionTreeClassifier(max_depth=depth).fit(train_data_X,train_data_y)
    #
    #     train_errors.append(np.sum(0.5*np.abs(clf.predict(train_data_X)-train_data_y))/float(len(train_data_y)))
    #     test_errors.append(np.sum(0.5*np.abs(clf.predict(test_data_X)-test_data_y))/float(len(test_data_y)))
    #
    # plt.figure(figsize=(12,10))
    # plt.xlabel("max depth",{'size':'15'})
    # plt.ylabel("percent error",{'size':'15'})
    # plt.plot(train_errors,'b-',label="train_errors")
    # plt.plot(test_errors,'r*-',label="test_errors")
    # plt.legend()
    # plt.show()
    #
    #
    #
    # # 3.3.4 [optional]
    # clf = DecisionTreeClassifier(max_depth=5).fit(train_data_X,train_data_y)
    # # min_samples_split : int, optional (default=2)
    # # The minimum number of samples required to split an internal node.
    #
    # # min_samples_leaf : int, optional (default=1)
    # # The minimum number of samples required to be at a leaf node.
    #
    # # max_leaf_nodes : int or None, optional (default=None)
    # # Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
    # # If None then unlimited number of leaf nodes. If not None then max_depth will be ignored.
    #
    # # random_state : int, RandomState instance or None, optional (default=None)
    # # If int, random_state is the seed used by the random number generator;
    # # If RandomState instance, random_state is the random number generator; If None, the random number generator
    # #     is the RandomState instance used by np.random.
    #
    #
    # # ===== plot ==========================================================================
    # plt.figure(figsize=(12,10))
    # plot_step = 0.01
    #
    # x_min, x_max = train_data_X.iloc[:, 0].min()-0.1, train_data_X.iloc[:, 0].max()+0.1
    # y_min, y_max = train_data_X.iloc[:, 1].min()-0.1, train_data_X.iloc[:, 1].max()+0.1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    #                          np.arange(y_min, y_max, plot_step))
    #
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    # cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    #
    # plt.axis("tight")
    # for i, color in [(-1,'b'),(1,'r')]:
    #     idx = np.where(train_data_y == i)
    #     plt.scatter(train_data_X.iloc[idx[0], 0], train_data_X.iloc[idx[0], 1], c=color,
    #                     cmap=plt.cm.Paired)
    # plt.show()
    #
    #
    # #===== test error ====================================================================
    #
    # print np.sum(0.5*np.abs(clf.predict(test_data_X)-test_data_y))/float(len(test_data_y))




    # 5.1 AdaBoost
    train_err = []
    test_err = []

    for rounds in range(1,10):
        w = np.array([1.0/len(train_data_X)]*len(train_data_X))
        train_predict = []
        test_predict = []

        for i in range(rounds):

            clf = DecisionTreeClassifier(max_depth=3).fit(train_data_X,train_data_y,sample_weight=w)
            err = np.sum(w*(0.5*np.abs(clf.predict(train_data_X)-train_data_y)))/float(np.sum(w))

            alpha = np.log((1-err)/err)

            w = alpha*w*(0.5*np.abs(clf.predict(train_data_X)-train_data_y.values))+w*0.5*np.abs((clf.predict(train_data_X)+train_data_y.values))

            train_predict.append(alpha * clf.predict(train_data_X))
            test_predict.append(alpha * clf.predict(test_data_X))

    #         train_error.append(np.sum(0.5*np.abs(clf.predict(train_data_X)-train_data_y))/float(len(train_data_y)))
    #         test_error.append(np.sum(0.5*np.abs(clf.predict(test_data_X)-test_data_y))/float(len(test_data_y)))
        train_predict_sum = np.sum(np.array(train_predict),axis=0)
        test_predict_sum = np.sum(np.array(test_predict),axis=0)
        train_err.append(np.sum(0.5*np.abs(np.sign(train_predict_sum)-train_data_y.values))/float(len(train_data_X)))
        test_err.append(np.sum(0.5*np.abs(np.sign(test_predict_sum)-test_data_y.values))/float(len(test_data_X)))



if __name__ == "__main__":
    main()