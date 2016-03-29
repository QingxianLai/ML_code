import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
import time


### Assignment Owner: Hao Xu

#######################################
####Q2.1: Normalization

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    lmax = train.max(axis=0)
    lmin = train.min(axis=0)
    lrange = lmax - lmin

#     l = [lrange == 0.,lrange != 0]
#     choice = [1,lrange]
#     lrange = np.select(l,choice)

    train_normalized = (train-lmin)/lrange     # use broadcasting here
    test_normalized = (test-lmin)/lrange
    return train_normalized, test_normalized
    

    
########################################
####Q2.2a: The square loss function

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
    loss = 0       #initialize the square_loss
    m = X.shape[0]
    theta = theta.reshape(X.shape[1], 1)

    temp = X.dot(theta)-y.reshape(m, 1)
    loss = (temp.T.dot(temp))/float(2*m)
    return loss


########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)  1 X 51
    """
    m = X.shape[0]
    theta = theta.reshape(X.shape[1],1)
    grad = ((X.dot(theta)-y.reshape(m, 1)).T.dot(X))/float(m)
    return grad
       
        
###########################################
###Q2.3a: Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=0.1):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient  1x51
    num_features = theta.shape[0]
    e = np.identity(num_features)
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    for i in range(num_features):
        test_theta_plus = theta+e[i]*epsilon
        test_theta_minus = theta-e[i]*epsilon
        approx_grad[i] = (compute_square_loss(X, y, test_theta_plus)-compute_square_loss(X, y, test_theta_minus))/(2*epsilon)
    approx_grad = np.array(approx_grad).reshape(true_gradient.shape)
    E_distance = np.sqrt((approx_grad-true_gradient).dot((approx_grad-true_gradient).T))
    return E_distance <= tolerance
    
#################################################
###Q2.3b: Generic Gradient Checker


def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    true_gradient = gradient_func(X, y, theta) #the true gradient
    num_features = theta.shape[0]

    e = np.identity(num_features)
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate

    for i in range(num_features):
        test_theta_plus = theta+e[i]*epsilon
        test_theta_minus = theta-e[i]*epsilon
        approx_grad[i] = (objective_func(X, y, test_theta_plus)-objective_func(X, y, test_theta_minus))/(2*epsilon)
    approx_grad = np.array(approx_grad).reshape(true_gradient.shape)
    E_distance = np.sqrt((approx_grad-true_gradient).dot((approx_grad-true_gradient).T))
    return E_distance <= tolerance



####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, grad_checkerr=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        grad_checker - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))   #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)                    #initialize loss_hist
    theta = np.ones(num_features)                       #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)[0,0]
    for i in range(1, num_iter+1):
        temp = theta_hist[i-1]-alpha*compute_square_loss_gradient(X, y, theta_hist[i-1])
        theta_hist[i] = temp
        if grad_checkerr:
            if not grad_checker(X, y, theta_hist[i]):
                raise Exception("do not pass the gradient test on the iteration %s" % (i))
#              print "grad check failed at iteration %s" % (i)
        loss_hist[i] = compute_square_loss(X, y, theta_hist[i])[0,0]
    return theta_hist, loss_hist



def batch_alpha_plot():
    print('loading the dataset')
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))     #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))        #Add bias term

    r = lambda: random.randint(0, 255)
    alpha_list = np.arange(0.01, 0.11, 0.01)
    plt.figure()
    for alpha in alpha_list:
        theta_hist, loss_hist = batch_grad_descent(X_train,y_train,alpha)
        kolor = '#%02X%02X%02X' % (r(),r(),r())
        plt.plot(loss_hist,color=kolor,label="alpha = "+str(alpha))
    # plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    # plt.title("batch gradient descent with different step size")
    plt.legend()
    plt.xlim(0,20)
    # plt.show()
    plt.savefig("batch_gradient_descent_with_different_step_size.png")
    # print loss_hist



####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
def batch_grad_descent_backtracking_line(X, y, alpha_init, num_iter=1000, grad_checkerr=False):
    """
    implement Backtracking-Armijo line search to improve the batch gradient desent

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        num_iter - number of iterations to run
        grad_checkerr - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))   #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)                    #initialize loss_hist
    theta = np.ones(num_features)                       #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_square_loss(X, y, theta)
    beta = 1e-4
    c = 0.5
    for i in range(1, num_iter+1):
        alpha = alpha_init
        last_loss = compute_square_loss(X,y,theta_hist[i-1])
        last_direction = compute_square_loss_gradient(X, y, theta_hist[i-1])
        theta_hist[i] = theta_hist[i-1]-alpha*last_direction

        while compute_square_loss(X,y,theta_hist[i]) > last_loss + alpha*beta*last_direction.dot(last_direction.T):
            alpha = alpha * c
            theta_hist[i] = theta_hist[i-1]-alpha*compute_square_loss_gradient(X,y,theta_hist[i-1])


        loss_hist[i] = compute_square_loss(X, y, theta_hist[i])

    return theta_hist, loss_hist
    



###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
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
    m = X.shape[0]   # num_instances
    y = y.reshape((m, 1))
    theta = theta.reshape(X.shape[1], 1)

    grad = (1/float(m))*(X.dot(theta)-y).T.dot(X)+2*lambda_reg*theta.T
    return grad




def compute_regularized_square_loss(X,y,theta,lambda_reg):
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
    y = y.reshape(m,1)
    temp = X.dot(theta)-y
    loss = (temp.T.dot(temp))/float(2*m)+lambda_reg*(theta.T).dot(theta)
    return loss






###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
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
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))   #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)                    #initialize loss_hist
    theta = np.ones(num_features)                       #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_regularized_square_loss(X, y, theta,lambda_reg)[0,0]

    for i in range(1,num_iter+1):
        theta_hist[i]=theta_hist[i-1]-alpha*compute_regularized_square_loss_gradient(X, y, theta_hist[i-1], lambda_reg)
        loss_hist[i] = compute_square_loss(X, y, theta_hist[i])[0,0]
    return theta_hist, loss_hist





def regularized_grad_descent_timer():
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
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term


    alpha=0.05
    lambda_reg=0.01
    num_iter=1000
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))   #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1)                    #initialize loss_hist
    theta = np.ones(num_features)                       #initialize theta
    theta_hist[0] = theta
    loss_hist[0] = compute_regularized_square_loss(X_train, y_train, theta,lambda_reg)

    start_time = time.time()
    for i in range(1,num_iter+1) :
        theta_hist[i]=theta_hist[i-1]-alpha*compute_regularized_square_loss_gradient(X_train,y_train,theta_hist[i-1],lambda_reg)
        # loss_hist[i] = compute_regularized_square_loss(X_train, y_train, theta_hist[i])
        # if np.abs(loss_hist[i]-loss_hist[i-1])<0.0001:
        #     break
    elapsed_time = time.time() - start_time
    print "time per iteration is : ", elapsed_time/1000.0




#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss
def visualize_reg_loss():

    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    La = [10**k for k in range(-10, 0)]

    # La = [k*10**(-3) for k in range(1,20)]


    train_loss = []
    theta_list = []
    validation_loss = []
    for lam in La:
        theta_hist,loss_hist = regularized_grad_descent(X_train,y_train,alpha=0.05, lambda_reg=lam)
        train_loss.append(compute_square_loss(X_train, y_train, theta_hist[-1])[0, 0])
        theta_list.append(theta_hist[-1])
        validation_loss.append(compute_square_loss(X_test, y_test, theta_hist[-1])[0, 0])
    La = np.array(La)
    train_loss = np.array(train_loss)
    theta_list = np.array(theta_list)
    validation_loss = np.array(validation_loss)

    ### plot type 1 ####
    x = np.log10(La)
    plt.plot(x,train_loss,'r',label='train_loss')
    plt.plot(x,validation_loss,'b',label = 'val_loss')
    plt.xlabel("log(lambda)")
    plt.ylabel("loss")

    plt.legend(loc=2)
    plt.savefig("RGD_loss_by_lambda.png")

    #### plot type 2 ####
    # La = [0.01*x for x in range(-2,2)]
    # La = [10**k for k in range(-3, 0)]
    # r = lambda: random.randint(0, 255)
    # plt.figure()
    # for lam in La:
    #     theta_hist,loss_hist = regularized_grad_descent(X_train,y_train,alpha=0.1, lambda_reg=lam)
    #     kolor = '#%02X%02X%02X' % (r(),r(),r())
    #     plt.plot(loss_hist,color=kolor,label=str(lam)
    #
    # # plt.yscale('log')
    # plt.legend()
    # plt.xlim(0,200)
    # plt.show()




def grid_search_B():
    #Loading the dataset
    # set lambda = -0.02
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)

    # r = lambda: random.randint(0, 255)
    # plt.figure()
    #
    B_list = [k for k in range(1, 5)]

    train_loss = []
    theta_list = []
    validation_loss = []

    for B in B_list:
        X_train_in = np.hstack((X_train, B*np.ones((X_train.shape[0], 1)))) #Add bias term
        X_test_in = np.hstack((X_test, B*np.ones((X_test.shape[0], 1)))) #Add bias term


        theta_hist,loss_hist = regularized_grad_descent(X_train_in,y_train, alpha=0.05, lambda_reg=0.01)
        train_loss.append(compute_square_loss(X_train_in, y_train, theta_hist[-1])[0, 0])
        theta_list.append(theta_hist[-1])
        validation_loss.append(compute_square_loss(X_test_in, y_test, theta_hist[-1])[0, 0])

        # kolor = '#%02X%02X%02X' % (r(),r(),r())
        # plt.plot(loss_hist,color=kolor,label=str(B))
    # plt.legend()
    # plt.xlim(0,20)
    # plt.show()


    plt.plot(B_list,train_loss,'r',label='train_loss')
    plt.plot(B_list,validation_loss,'b',label = 'val_loss')
    plt.xlabel("B")
    plt.ylabel("loss")

    plt.legend(loc=2)
    plt.savefig("RGD_loss_by_B.png")
    """
    The result shows that B should be 0 or -0.3 !!!???
    """





#############################################
###Q2.6a: Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.05, lambda_reg=0.01, num_iter=1000):
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
    theta = np.zeros(num_features) #Initialize theta
    rand = np.arange(num_instances)
    theta_hist = []
    loss_hist = []
    theta = np.ones(num_features)
    np.random.shuffle(rand)
    t = 1
    for i in range(num_iter):
        theta_list = []
        loss_list = []

        for j in range(num_instances):
            alpha = 0.05/t
            t = t+1

            x = X[rand[j]]
            yy = y[rand[j]]

            tmp = sgd_loss_gradient(x,yy,theta,lambda_reg)
            theta = theta - alpha*tmp
            # theta = theta - alpha * sgd_loss_gradient(x,yy,theta,lambda_reg)
            theta_list.append(theta)
            loss = compute_regularized_square_loss(X,y,theta,0.01)
            loss_list.append(loss)
        theta_hist.append(theta_list)
        loss_hist.append(loss_list)
    theta_hist = np.array(theta_hist)
    loss_hist = np.array(loss_hist)
    return theta_hist, loss_hist




def SGD_train_validation():
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    plt.figure()
    SGD_alpha_plot(X_train,y_train,'-','SGD_loss')
    # SGD_alpha_plot(X_test,y_test,'--','validation loss')
    plt.yscale('log')
    plt.xlim(0,200)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("SGD_comparation.png")


def SGD_timer():
    print('loading the dataset')

    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    start_time = time.time()
    a,b = stochastic_grad_descent(X_train,y_train)
    elapes_time = time.time()-start_time
    print elapes_time/1000.0


def SGD_alpha_plot(X,y,lines,labels):

    num_instances, num_features = X.shape[0], X.shape[1]
    rand = np.arange(num_instances)

    theta = np.ones(num_features) #Initialize theta
    theta_hist = [theta]
    loss_hist = [compute_regularized_square_loss(X,y,theta,0.01)[0,0]]


    lambda_reg = 0.01

    t = 1
    for i in range(1000):
        np.random.shuffle(rand)
        theta_list = []
        for j in range(num_instances):
            alpha = 0.01/t
            t = t+1
            x = X[rand[j]]
            yy = y[rand[j]]

            theta = theta - alpha * sgd_loss_gradient(x,yy,theta,lambda_reg)
            theta_list.append(theta)

        theta_hist.append(theta_list)
        loss_hist.append(compute_regularized_square_loss(X,y,theta_list[-1],0.01)[0,0])
    theta_hist = np.array(theta_hist)
    loss_hist = np.array(loss_hist)
    plt.plot(loss_hist,color='r',linestyle=lines,label=labels+" (alpha = 1/t)")


    theta = np.ones(num_features) #Initialize theta
    theta_hist = [theta]
    loss_hist = [compute_regularized_square_loss(X,y,theta,0.01)[0,0]]


    for i in range(1000):
        np.random.shuffle(rand)
        theta_list = []

        alpha = 0.001

        for j in range(num_instances):
            x = X[rand[j]]
            yy = y[rand[j]]

            theta = theta - alpha * sgd_loss_gradient(x,yy,theta,lambda_reg)
            theta_list.append(theta)

        theta_hist.append(theta_list)
        loss_hist.append(compute_regularized_square_loss(X,y,theta_list[-1],0.01)[0,0])
    theta_hist = np.array(theta_hist)
    loss_hist = np.array(loss_hist)
    plt.plot(loss_hist,color='b',linestyle=lines,label=labels+" (alpha = 0.005)")


    theta = np.ones(num_features) #Initialize theta
    theta_hist = [theta]
    loss_hist = [compute_regularized_square_loss(X,y,theta,0.01)[0,0]]

    t = 1
    for i in range(1000):
        np.random.shuffle(rand)
        theta_list = []
        for j in range(num_instances):
            alpha = 0.01/np.sqrt(t)
            t = t+1
            x = X[rand[j]]
            yy = y[rand[j]]

            theta = theta - alpha * sgd_loss_gradient(x,yy,theta,lambda_reg)
            theta_list.append(theta)

        theta_hist.append(theta_list)
        loss_hist.append(compute_regularized_square_loss(X,y,theta_list[-1],0.01)[0,0])
    theta_hist = np.array(theta_hist)
    loss_hist = np.array(loss_hist)
    plt.plot(loss_hist,color='k',linestyle=lines,label=labels+" (alpha = 1/sqrt(t))")

    c,d = regularized_grad_descent(X,y,0.01,0.01,1000)
    plt.plot(d,color='y',label = "GD loss")

def main2():
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term

    m = X_train.shape[0]
    theta_hist, loss_hist = stochastic_grad_descent(X_train,y_train,lambda_reg=0.01)
    loss_sum = np.sum(loss_hist,axis=1)
    loss_sum = loss_sum/(2.0*m)
    plt.plot(loss_sum)
    plt.yscale('log')
    plt.xlim(0,100)
    print loss_sum
    plt.show()




def sgd_loss(x,y,theta,lambda_reg=0.01):
    """
    input :
        x - 1 X num of features array
        y - scalar
        theta - num of features X 1 array

    return:
        loss - scalar
    """

    loss = (x.dot(theta)-y)**2+lambda_reg*theta.dot(theta)
    return loss



def sgd_loss_gradient(x,y,theta,lambda_reg=0.01):
    """
    input :

        x - 1 X num of features array
        y - scalar
        theta - num of features X 1 array
        lambda - regularing parameter

    return :
        grad - 1 X num of features array
    """

    grad =2 * (x.dot(theta)-y)*x - lambda_reg*theta
    return grad



################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)

def main():
    #Loading the dataset
    print('loading the dataset')
    
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term
    a,b = regularized_grad_descent(X_train,y_train,alpha=0.05)
    print a,b
    plt.plot(b)
    plt.xlim(0,20)
    plt.show()




def main3():
    # batch_alpha_plot()
    print('loading the dataset')
    df = pd.read_csv('hw1-data.csv', delimiter=',')
    X = df.values[:, :-1]
    y = df.values[:, -1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))     #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))        #Add bias term

    # theta = np.ones(X_train.shape[1])
    # theta=theta-0.1*compute_square_loss_gradient(X_train,y_train,theta)
    # theta = theta.reshape(49,1)
    #
    # print compute_square_loss(X_train,y_train,theta)

    # a,b = batch_grad_descent_backtracking_line(X_train,y_train,alpha_init=0.1)
    # print a,b
    r = lambda: random.randint(0, 255)
    alpha_list = np.arange(0.01, 1, 0.12)
    plt.figure()
    for alpha in alpha_list:
        theta_hist, loss_hist = batch_grad_descent(X_train,y_train,alpha)
        kolor = '#%02X%02X%02X' % (r(),r(),r())
        plt.plot(loss_hist,color=kolor,label=str(alpha))
    plt.legend()
    plt.xlim(0,50)
    plt.show()


if __name__ == "__main__":
    SGD_train_validation()