__author__ = 'LaiQX'
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import random
import time


# ================================================================================================
# 1.1
#================================================================================================

def data_construct():
    m = 150  # number of examples
    d = 75  # number of feature

    X = np.random.rand(m, d)

    theta = [-10, -10, -10, 10, 10, -10, 10, 10, 10, -10] + [0] * 65
    theta = np.array(theta).reshape(d, 1)

    epsilon = 0.1 * np.random.randn()
    y = X.dot(theta) + epsilon

    # split dataset into train validation and test data set
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=50, random_state=10)
    X_train, X_vali, y_train, y_vali = train_test_split(X_tv, y_tv, test_size=20, random_state=11)

    print X_train.shape, y_train.shape, X_vali.shape, y_vali.shape, X_test.shape, y_test.shape
    # print theta.T
    return X_train, y_train, X_vali, y_vali, X_test, y_test


#================================================================================================
# 1.2
#================================================================================================
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    m = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)

    temp = X.dot(theta) - y.reshape(m, 1)
    loss = (temp.T.dot(temp)) / float(2 * m)
    return loss


def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    m = X.shape[0]  # num_instances
    y = y.reshape((m, 1))
    theta = theta.reshape(X.shape[1], 1)

    grad = (1 / float(m)) * (X.dot(theta) - y).T.dot(X) + 2 * lambda_reg * theta.T
    return grad


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    Compute the loss of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        loss - the square loss, scalar
    """
    m = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)
    y = y.reshape(m, 1)
    temp = X.dot(theta) - y
    loss = (temp.T.dot(temp)) / float(2 * m) + lambda_reg * (theta.T).dot(theta)
    return loss


def regularized_grad_descent(X_train, y_train, X_val, y_val, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter + 1)  #initialize loss_hist
    theta = np.ones(num_features)  #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_regularized_square_loss(X_train, y_train, theta, lambda_reg)[0, 0]
    val_hist = np.zeros(num_iter + 1)
    val_hist[0] = compute_square_loss(X_val, y_val, theta)[0, 0]
    for i in range(1, num_iter + 1):
        theta_hist[i] = theta_hist[i - 1] - alpha * compute_regularized_square_loss_gradient(X_train, y_train,
                                                                                             theta_hist[i - 1],
                                                                                             lambda_reg)
        loss_hist[i] = compute_regularized_square_loss(X_train, y_train, theta_hist[i],lambda_reg)[0, 0]
        val_hist[i] = compute_square_loss(X_val, y_val, theta_hist[i])[0, 0]
    return theta_hist, loss_hist, val_hist


def ridge_regression(X_train, y_train, X_vali, y_vali, X_test, y_test):
    """
    this method is used to choose alpha and lambda
    :param X_train:
    :param y_train:
    :param X_vali:
    :param y_vali:
    :param X_test:
    :param y_test:
    :return:
    """
# ################# search for the range of alpha and lambda ########################
#     la = range(-8, 0, 2)
#     La = [10**k for k in la]
#     Alpha = [k for k in np.arange(0.01,0.101,0.02)]
#     r = lambda: random.randint(0, 255)
#     for alpha in Alpha:
#         vali_loss = []
#         for lam in La:
#             theta_hist, loss_hist, val_hist = regularized_grad_descent(X_train, y_train, X_vali, y_vali, alpha,
#                                                                                                       lam, 300)
#             # theta = theta_hist[-1]
#             # train_loss.append(compute_square_loss(X_train,y_train,theta)[0,0])
#             vali_loss.append(val_hist[-1])
#         kolor = '#%02X%02X%02X' % (r(), r(), r())
#         plt.plot(la,vali_loss, color=kolor, label="alpha= "+str(alpha))
#     plt.legend()
#     plt.show()
# ################# result: alpha = 0.1, lambda = 1e-4 ###################################


    la = range(-8, 0, 2)
    La = [10**k for k in la]
    vali_loss = []
    train_loss = []
    for lam in La:
        theta_hist, loss_hist, val_hist = regularized_grad_descent(X_train, y_train, X_vali, y_vali, alpha=0.1,
                                                                                                  lambda_reg=lam, num_iter=300)
        # theta = theta_hist[-1]
        # train_loss.append(compute_square_loss(X_train,y_train,theta)[0,0])
        vali_loss.append(val_hist[-1])
        train_loss.append(loss_hist[-1])
    plt.plot(la,vali_loss,'b-',label='validation')
    plt.plot(la,train_loss,'r-',label="train")
    plt.legend()
    plt.xlabel("log(lambda)")
    plt.ylabel("loss")
    plt.savefig("T_1_2.png")
    plt.show()


def compare_model_cofficients(X_train, y_train, X_vali, y_vali, threshold=0,show_true_cof=False):

    true_value = np.array([-10, -10, -10, 10, 10, -10, 10, 10, 10, -10] + [0] * 65)
    if show_true_cof:
        print true_value

    #### >> defined
    theta_list,train_loss,validation_loss = regularized_grad_descent(X_train, y_train, X_vali, y_vali,alpha=0.1,lambda_reg=1e-4,num_iter=1000)
    print theta_list[-1]
    count1 = 75 - np.sum(np.abs(theta_list[-1]-true_value)<=threshold)
    print "Report <GD methods>, with threshold = "+str(threshold)
    print "wrong value count: "+str(count1)


    ### >> sklearn linear_model Ridge
    clf = Ridge(alpha=0.1,fit_intercept=False,max_iter=1000,normalize=False,solver='auto')
    clf.fit(X_train,y_train)
    print clf.coef_
    count2 = 75 - np.sum(np.abs(clf.coef_-true_value)<=threshold)
    print "Report <sklearn ridge>, with threshold = "+str(threshold)
    print "wrong value count: "+str(count2)

#================================================================================================
# 2.1
#================================================================================================
def lasso_loss(X,y,theta,lambda_reg):
    """
    compute the lasso loss with given data and parammaters
    :param X:
    :param y:
    :param theta:
    :param lambda_reg:
    :return:
    """
    n,d = X.shape
    theta = theta.reshape(d,1)
    y = y.reshape(n,1)
    temp = X.dot(theta)-y

    l1 = np.abs(theta)
    l2 = np.sum(l1)
    regular_term = np.sum(np.abs(theta))
    loss = temp.T.dot(temp)[0,0]+lambda_reg*regular_term
    return loss


def soft(a,b):
    si = np.sign(a)
    tmp = np.abs(a)-b
    if tmp>0:
        pass
    else:
        tmp = 0
    ans = si*tmp
    return ans


def test_soft():
    a = np.arange(-10,20,0.01)
    b = [soft(x,5) for x in a]
    b = np.array(b)
    plt.plot(a,b)
    plt.show()


def shooting_method(X_train, y_train, X_vali, y_vali,w_init,lam):

    n,d = X_train.shape
    loss = [lasso_loss(X_train,y_train,w_init,lambda_reg=lam)]
    W  = w_init.copy()    # W is a numpy array
    vali_loss = []
    # iter = 0
    # tolerance = min(1e-4,lasso_loss(X_train,y_train,W,lam)/100.0)
    tolerance = 1e-2
    while True:
        W_last = W.copy()
        a = np.zeros(d)
        c = np.zeros(d)
        for j in range(d):
            # a[j]=0
            # for i in range(n):
            #     tmp = X_train[i,j]*X_train[i,j]
            #     a[j] = a[j] + tmp
            # a[j]=2*a[j]
            a[j] = 2*(np.linalg.norm(X_train[:,j])**2)
            c[j] = 2*(X_train[:,j].dot(y_train)[0]-X_train[:,j].reshape(n,1).T.dot(X_train.dot(W.reshape(d,1)))[0,0]+W[j]*(np.linalg.norm(X_train[:,j]))**2)
            # c[j]=0
            # for i in range(n):
            #     tmp = X_train[i,j]*(y_train[i]-W.T.dot(X_train[i])+W[j]*X_train[i,j])
            #     c[j] = c[j] + tmp
            # c[j] = 2*c[j]
            # test = c[j]/a[j]
            W[j] = soft(c[j]/a[j],lam/a[j])
        # print lasso_loss(X_train,y_train,W,lam),"     ", lasso_loss(X_train,y_train,W_last,lam)
        # w = np.array(w)z4
        vali_loss.append(compute_square_loss(X_vali,y_vali,W))
        # iter = iter +1
        if (abs(lasso_loss(X_train,y_train,W,lam)-lasso_loss(X_train,y_train,W_last,lam))<tolerance):
            break
        # if (abs(compute_square_loss(X_train,y_train,W)-compute_square_loss(X_train,y_train,W_last)<tolerance)):
        #     break
        # if iter>300: break
    W = W.reshape(d,1)
    vali_loss = compute_square_loss(X_vali,y_vali,W)[0,0]
    lass_loss = lasso_loss(X_train,y_train,W,lam)
    return lass_loss,vali_loss,W


def lasso(X_train, y_train, X_vali, y_vali):
    la = np.arange(-5,5,1)
    La = [10**k for k in la]
    # La = [k*1e4 for k in la]
    # La = la
    train_loss = []
    validation_loss = []
    plt.figure()
    w_initial = np.zeros(X_train.shape[1])
    for lamb in La:
        a,b,w = shooting_method(X_train,y_train,X_vali,y_vali,w_init=w_initial,lam=lamb)
        train_loss.append(a)
        validation_loss.append(b)
    plt.plot(la,validation_loss,'r-',label="validation")
    print validation_loss
    print w
    # plt.ylim(0,30)
    plt.xlabel("log(lambda)")
    plt.ylabel("loss")
    plt.yscale('log')
    # plt.show()
    plt.savefig("T_21_lasso_by_lambda.png")
    ##############################  lambda = 1e4 or 1e2 ?  #######


def lasso_report(X_train,y_train,X_vali,y_vali,w_init,lam=1e2,threshold=0):
    # w_initial = np.ones(X_train.shape[1])   if not specified
    a,b,w = shooting_method(X_train,y_train,X_vali,y_vali,w_init=w_init,lam=lam)
    true_value = np.array([-10, -10, -10, 10, 10, -10, 10, 10, 10, -10] + [0] * 65).reshape(X_train.shape[1],1)
    count = 75 - np.sum(np.abs(w-true_value)<=threshold)
    print w.T
    print b
    print "lasso ",a
    print "Report <Lasso>, with threshold = "+str(threshold)
    print "wrong value count: "+str(count)


def homotopy_methods(X_train,y_train,X_vali,y_vali,lam_list,w_init):
    vali_list = []
    w_list = []
    w = w_init
    for lam in lam_list:
        lass_loss,vali_loss,w = shooting_method(X_train,y_train,X_vali,y_vali,w,lam)
        vali_list.append(vali_loss)
        w_list.append(w)
    return vali_list,w_list


def non_homotopy_methods(X_train,y_train,X_vali,y_vali,lam_list,w_init):
    vali_list = []
    w_list = []

    for lam in lam_list:
        lass_loss,vali_loss,w = shooting_method(X_train,y_train,X_vali,y_vali,w_init,lam)
        vali_list.append(vali_loss)
        w_list.append(w)
    return vali_list,w_list


def compare_homotopy_nonhomotopy(X_train,y_train,X_vali,y_vali):
    a =np.abs(X_train.T.dot(y_train-np.mean(y_train)))
    lambda_max = 2 * a.max()
    La = np.arange(lambda_max,0,-20)
    w = np.zeros(X_train.shape[1])

    ###### >> time the homotopy method
    start = time.time()
    vali_list,w_list = homotopy_methods(X_train,y_train,X_vali,y_vali,lam_list=La,w_init=w)
    elapse_time = time.time() - start
    print vali_list
    print "homotopy method : ",elapse_time," seconds"
    plt.plot(La,vali_list,'b-',label="homotopy")

    ###### >> time the homotopy method
    start = time.time()
    vali_list,w_list = non_homotopy_methods(X_train,y_train,X_vali,y_vali,lam_list=La,w_init=w)
    elapse_time = time.time() - start
    print vali_list
    plt.plot(La,vali_list,'r-',label="non-homotopy")
    print "non-homotopy method : ",elapse_time," seconds"

    plt.yscale("log")
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

def vectorization_timer(X_train,y_train,X_vali,y_vali):
    a =np.abs(X_train.T.dot(y_train-np.mean(y_train)))
    lambda_max = 2 * a.max()
    La = np.arange(lambda_max,0,-20)
    w = np.zeros(X_train.shape[1])

    ###### >> time the homotopy method
    start = time.time()
    vali_list,w_list = homotopy_methods(X_train,y_train,X_vali,y_vali,lam_list=La,w_init=w)
    elapse_time = time.time() - start
    print vali_list
    print "homotopy method : ",elapse_time," seconds"

#================================================================================================
# 2.2
#================================================================================================

def projected_SGD_p(X,y,theta_p,theta_n,lamb):
    a = len(X)
    ans = (X.dot(theta_p-theta_n)-y)*X+lamb*np.ones(a)
    return ans

def projected_SGD_n(X,y,theta_p,theta_n,lamb):
    a = len(X)
    ans = (-1)*(X.dot(theta_p-theta_n)-y)*(X)+lamb*np.ones(a)
    return ans


def stochastic_grad_descent(X, y, X_vali,y_vali, alpha=0.05, lambda_reg=1, num_iter=100):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """

    num_instances, num_features = X.shape[0], X.shape[1]
    rand = np.arange(num_instances)
    theta_p_hist = []
    theta_n_hist = []
    loss_hist = []
    theta_p = np.random.rand(num_features)
    theta_n = np.random.rand(num_features)
    np.random.shuffle(rand)
    t = 1


    for i in range(num_iter):
        theta_n_list = []
        theta_p_list = []
        loss_list = []

        for j in range(num_instances):
            # alpha = 0.01/t
            # t = t+1
            alpha = 0.01

            x = X[rand[j]]
            yy = y[rand[j]]

            tmp1 = projected_SGD_p(x,yy,theta_p,theta_n,lambda_reg)
            theta_p_tmp = theta_p - alpha*tmp1
            a = [theta_p_tmp>=0,theta_p_tmp<0]
            choice = [theta_p_tmp,0]
            theta_p_tmp = np.select(a,choice)




            tmp2 = projected_SGD_n(x,yy,theta_p,theta_n,lambda_reg)
            theta_n_tmp = theta_n - alpha*tmp2
            a = [theta_n_tmp>=0,theta_n_tmp<0]
            choice = [theta_n_tmp,0]
            theta_n_tmp = np.select(a,choice)
            #


            theta_p = theta_p_tmp
            theta_n = theta_n_tmp
            theta_p_list.append(theta_p)
            theta_n_list.append(theta_n)


            loss = compute_square_loss(X_vali,y_vali,(theta_p-theta_n))[0,0]
            loss_list.append(loss)
        theta_p_hist.append(theta_p_list)
        theta_n_hist.append(theta_n_list)
        loss_hist.append(loss_list)
    theta_p_hist = np.array(theta_p_hist)
    theta_n_hist = np.array(theta_n_hist)
    loss_hist = np.array(loss_hist)
    return theta_p_hist,theta_n_hist, loss_hist

##### For TESTING ##################

def test_stochastic_grad_descent(X, y, X_vali,y_vali, alpha=0.05, lambda_reg=50, num_iter=100):


    num_instances, num_features = X.shape[0], X.shape[1]
    rand = np.arange(num_instances)

    theta_p = np.random.rand(num_features)
    theta_n = np.random.rand(num_features)
    # theta_p = np.ones(num_features)
    # theta_n = np.ones(num_features)

    np.random.shuffle(rand)
    t = 1
    losses = []

    for i in range(num_iter):

        for j in range(num_instances):
            # alpha = 0.01/t
            # t = t+1
            alpha = 0.01

            x = X[rand[j]]
            yy = y[rand[j]]

            tmp1 = projected_SGD_p(x,yy,theta_p,theta_n,lambda_reg)
            theta_p_tmp = theta_p - alpha*tmp1
            a = [theta_p_tmp>=0,theta_p_tmp<0]
            choice = [theta_p_tmp,0]
            theta_p_tmp = np.select(a,choice)

            tmp2 = projected_SGD_n(x,yy,theta_p,theta_n,lambda_reg)
            theta_n_tmp = theta_n - alpha*tmp2
            a = [theta_n_tmp>=0,theta_n_tmp<0]
            choice = [theta_n_tmp,0]
            theta_n_tmp = np.select(a,choice)

            theta_p = theta_p_tmp.copy()
            theta_n = theta_n_tmp.copy()

        # a = [theta_p>=0,theta_p<0]
        # choice = [theta_p,0]
        # theta_p = np.select(a,choice)
        #
        # a = [theta_n>=0,theta_n<0]
        # choice = [theta_n,0]
        # theta_n = np.select(a,choice)

        loss = compute_square_loss(X_vali,y_vali,(theta_p-theta_n))[0,0]
        losses.append(loss)

    return losses,theta_p-theta_n


def compare_SGD_shooting(X_train, y_train, X_vali, y_vali):
    la = np.arange(-5,5,1)
    La = [10**k for k in la]
    # La = [k*1e4 for k in la]
    # La = la
    train_loss = []
    validation_loss = []
    plt.figure()
    w_initial = np.zeros(X_train.shape[1])
    for lamb in La:
        a,b,w = shooting_method(X_train,y_train,X_vali,y_vali,w_init=w_initial,lam=lamb)
        train_loss.append(a)
        validation_loss.append(b)
    plt.plot(la,validation_loss,'r-',label="Shooting-Method")
    # print validation_loss
    # plt.ylim(0,30)

    validation_loss= []
    for lamb in La:
        loss,w = test_stochastic_grad_descent(X_train,y_train,X_vali,y_vali,alpha=0.01,lambda_reg=lamb,num_iter=500)
        validation_loss.append(loss[-1])
    plt.plot(la,validation_loss,'b-',label="Projected-SGD")


    plt.ylabel("validation_loss")
    plt.xlabel("log(lambda")
    plt.yscale('log')
    plt.legend()
    # plt.show()
    plt.savefig("compare_SGD_shouting.png")


#================================================================================================
# 2.3
#================================================================================================


if __name__ == "__main__":
    X_train, y_train, X_vali, y_vali, X_test, y_test = data_construct()

    # ridge_regression(X_train, y_train, X_vali, y_vali,X_test, y_test )
    # # 1.2
    # compare_model_cofficients(X_train, y_train, X_vali, y_vali,threshold=0.1)

    # # 2.1
    # lasso(X_train, y_train, X_vali, y_vali)
    # test_soft()


    # # test lasso and count zero-non-zero
    # lasso_report(X_train,y_train,X_vali,y_vali,w_init=np.zeros(X_train.shape[1]),lam=1)

    compare_homotopy_nonhomotopy(X_train,y_train,X_vali,y_vali)

    # 2.2
    # a,b,c = stochastic_grad_descent(X_train, y_train, X_vali, y_vali)
    #
    # print (a-b)[-1,-1,:]
    # print c[:,-1]



    #
    # loss,theta = test_stochastic_grad_descent(X_train, y_train, X_vali, y_vali,lambda_reg=0.01,num_iter=1000)
    # print loss[-1]
    # print theta
    # true_value = np.array([-10, -10, -10, 10, 10, -10, 10, 10, 10, -10] + [0] * 65)
    # count1 = np.sum(np.abs(theta -true_value)<=0.001)
    # print count1

    # compare_SGD_shooting(X_train, y_train, X_vali, y_vali)

    # w_init = np.zeros(X_train.shape[1])
    # a,b,c = shooting_method(X_train, y_train, X_test, y_test,w_init,1)
    # print c.T
    # true_value = np.array([-10, -10, -10, 10, 10, -10, 10, 10, 10, -10] + [0] * 65)
    # count1 = 75 - np.sum(np.abs(c -true_value.reshape(75,1))<=0.001)
    # print count1
    # compare_homotopy_nonhomotopy(X_train, y_train, X_vali, y_vali)

    # vectorization_timer(X_train, y_train, X_vali, y_vali)



    # a,b,w = shooting_method(X_train,y_train,X_vali,y_vali,w_init=np.zeros(X_train.shape[1]),lam=1)
    # print w
    # print b