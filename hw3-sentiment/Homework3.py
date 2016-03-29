from __future__ import division
import pickle
from util import dotProduct,increment
import numpy as np
import matplotlib.pyplot as plt
import operator


#====================================================================================
#   2.1
#====================================================================================
def read_files():

    file1 = open("train_data","r")
    file2 = open("test_data","r")

    train_data = pickle.load(file1)
    test_data = pickle.load(file2)

    file1.close()
    file2.close()

    return train_data, test_data

#====================================================================================
#   3.1
#====================================================================================
def convert_to_word_bag(words_list):
    a = {}
    for words in words_list[:-1]:
        a[words] = a.get(words,0)+1

    # ##improve
    # a["word_count"] = len(words)-1
    return a

def data_label(data_set):
    n = len(data_set)
    X = []
    y = []
    for i in range(n):
        X.append(convert_to_word_bag(data_set[i]))
        y.append(data_set[i][-1])
    return X,y

#====================================================================================
#   3.2
#====================================================================================
def pegasos_grad(X,y,w,lamb):
    tmp = y*dotProduct(w,X)
    if 1-tmp > 0:
        an1 = increment({},lamb,w)
        ans = increment(an1,y,X)
    else:
        ans = increment({},lamb,w)
    return ans

def pegasos_loss(X,y,w,lamb):
    ans = (lamb/2.0)*dotProduct(w,w)+max(0,1-y*dotProduct(w,X))
    return ans

def generic_gradient_checker(X,y,w,lamb,grad_fun,loss_fun,epsilon=0.01,tolerance=0.1):
    true_gradient = grad_fun(X,y,w,lamb)
    approx_grad = {}
    distance = 0
    for key in w.keys():
        test_w_p = w.copy()
        test_w_n = w.copy()
        test_w_p[key] += epsilon
        test_w_n[key] -= epsilon
        approx_grad[key] = (loss_fun(X,y,test_w_p,lamb)-loss_fun(X,y,test_w_n,lamb))/(2*epsilon)
        distance += (true_gradient[key]-approx_grad[key])*(true_gradient[key]-approx_grad[key])
    distance = np.sqrt(distance)
    # print distance
    # print true_gradient
    # print approx_grad
    return distance <= tolerance



#====================================================================================
#   4.2
#====================================================================================

def l_de(x):
    if x<1:
        return -1
    else:
        return 0


def pegasos_SGD(X,y,lamb,num_iter):

    w = {}
    t = 1
    s = 1
    for i in range(num_iter):

        for j in range(len(X)):

            t += 1
            alpha = 1.0/(t*lamb)
            tmp = y[j] * s * dotProduct(X[j], w)
            g = l_de(tmp)
            s *= (1 - alpha * lamb)
            w = increment(w, -(alpha*y[j]*g/s), X[j])
        print "epoch "+str(i)
    return increment({},s,w)




#====================================================================================
#   4.3
#====================================================================================

def percent_error(X, y, w):
    correct = 0
    pos = 0
    total = len(y)
    for i in range(total):
        sign_value = np.sign(dotProduct(X[i], w))
        if y[i] == sign_value:
            correct += 1
        if sign_value > 0:
            pos += 1
    print pos
    return 1-float(correct)/total




#====================================================================================
#   4.4
#====================================================================================

def search_lambda():
    train_data,test_data = read_files()
    X_train, y_train = data_label(train_data)
    X_test, y_test = data_label(test_data)
    la = np.arange(-1,0)
    La = [10**x for x in la]
    p_error = []
    w_list = []
    for lam in La:
        print lam
        b = pegasos_SGD(X_train,y_train,lam,30)
        err = percent_error(X_test,y_test,b)
        p_error.append(err)
        w_list.append(b)
    plt.plot(la,p_error)
    plt.xlabel("log(lambda)")
    plt.ylabel("percent error")
    # plt.savefig("t44_search_lambda.png")
    print p_error
    # plt.show()

#====================================================================================
#   4.5
#====================================================================================

def score_interval():
    train_data,test_data = read_files()
    X_train, y_train = data_label(train_data)
    X_test, y_test = data_label(test_data)

    lamb = 1e-1

    w = pegasos_SGD(X_train,y_train,lamb,30)

    a = []
    test_len = len(X_test)
    for i in range(test_len):
        a.append(dotProduct(X_test[i],w))
    a.sort()
    print min(a)
    print max(a)
    plt.plot(a)
    plt.show()
    # all scores are between -8.8935 and 8.1989.

def score_confidence():
    train_data,test_data = read_files()
    X_train, y_train = data_label(train_data)
    X_test, y_test = data_label(test_data)

    lamb = 1e-1

    # w = pegasos_SGD(X_train,y_train,lamb,30)
    #
    # pickle.dump(w,open("weight",'wb'))
    w = pickle.load(open("weight",'rb'))

    a = [0]*18     #correct count
    b = [0]*18     #incorrect count
    c = [0]*18     #total number
    test_len = len(X_test)

    for i in range(test_len):
        tmp = dotProduct(X_test[i],w)
        c[int(tmp+9)] += 1
        if (np.sign(tmp)==y_test[i]):
            a[int(tmp+9)] += 1
        else:
            b[int(tmp+9)] += 1

    # for j in range(18):
    #     if b[j]==0:
    #         print "interval [%s,%s] has %s points and %s are correct, the ratio is %s "%(j-9,j-8,b[j],a[j],0)
    #     else:
    #         print "interval [%s,%s] has %s points and %s are correct, the ratio is %4.2f "%(j-9,j-8,b[j],a[j],float(a[j])/b[j])

    wide = 0.35
    p1 = plt.bar(np.arange(18),a,width=wide, color='g')
    p2 = plt.bar(np.arange(18),b,width=wide, color='r', bottom=a)

    plt.ylabel("Frequency")
    plt.xlabel("Score Intervals")
    # plt.title("Score Confidence")
    X_ticks = ["[%s,%s]"%(k-9,k-8) for k in np.arange(18)]

    plt.xticks(np.arange(18)+wide/2,X_ticks,rotation=45)
    plt.legend((p1[0],p2[0]),("Correct","Incorrect"))

    plt.savefig("t45_score_confidence.png")


#====================================================================================
#   5.1
#====================================================================================
def dotProduct_vector(d1_a, d2_a):
    d1 = d1_a.copy()
    d2 = d2_a.copy()
    d3 = {}
    for f, v in d2.items():
        if f in d1:
            d3[f] = d1.get(f,0)*v
    return d3




def show_incorrect_case():
    train_data,test_data = read_files()
    X_train, y_train = data_label(train_data)
    X_test, y_test = data_label(test_data)

    lamb = 0.1
    w_o= pickle.load(open("weight",'rb'))

    test_len = len(X_test)

    example = 0

    for i in range(test_len):
        w = w_o
        tmp = dotProduct(X_test[i],w)
        if (np.sign(tmp)!=y_test[i]):
            example += 1
            print "predicted_score: ",np.sign(tmp)
            print "true vote: ",y_test[i]
            dict_tmp = dotProduct_vector(X_test[i],w)
            sorted_dict = sorted(dict_tmp.items(),key=lambda x: abs(x[1]),reverse=True)
            print_list_wx = sorted_dict[:8]
            print_list_abs_wx = [(a,abs(b)) for a,b in print_list_wx]
            print_list_x = [X_test[i][x] for (x,k) in print_list_wx ]
            print_list_w = [w[x] for (x,k) in print_list_wx ]

            print "wx:"
            print print_list_wx
            print "\n"


            print "abs_wx"
            print print_list_abs_wx
            print "\n"


            print "x"
            print print_list_x
            print "\n"


            print "w"
            print print_list_w
            if example == 3:
                break

#====================================================================================
#   6.1
#====================================================================================
def std_error(percent_error):
    ans = np.sqrt(percent_error*(1-percent_error)/500)
    return ans


def improve_test():
    train_data,test_data = read_files()
    X_train, y_train = data_label(train_data)
    X_test, y_test = data_label(test_data)
    la = np.arange(-1,0)
    La = [10**x for x in la]
    lam = 0.1

    print lam
    b = pegasos_SGD(X_train,y_train,lam,30)
    err = percent_error(X_test,y_test,b)

    # plt.plot(la,p_error)
    # plt.xlabel("log(lambda)")
    # plt.ylabel("percent error")
    # plt.savefig("t44_search_lambda.png")

    print err

    stderr = std_error(err)
    print stderr



if __name__ == "__main__":
    # score_interval()
    # score_confidence()
    # search_lambda()
    # show_incorrect_case()
    improve_test()