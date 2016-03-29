import os
import numpy as np
import pickle
import random

'''
Note: No obligation to use this code, though you may if you like.  Skeleton code is just a hint for people who are not
familiar with text processing in python.
It is not necessary to follow. 
'''



def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review


def read_data(file):
    '''
    Read each file into a list of strings. 
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on', 
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    eliminate = ["and","to","the","you","on","will","is","of","have","as","back","such","been","why","it's","that",
                 "most","this","a","be","then"]
    # eliminate = ["and","to","the"]
    f = open(file)
    lines = f.read().split(' ')
    # symbols = '${}()[].,:;+-*/&|<>=~" '
    symbols = '${}[]" '
    # symbols = ' '
    words = map(lambda Element: Element.translate(None, symbols).strip(), lines)
    words = filter(None, words)

    for i in range(len(eliminate)):
        while 1:
            try:
                words.remove(eliminate[i])
            except:
                break

    i = 0
    n = len(words)
    while i < n-1:
        if words[i] == "not":
            words[i] = words[i]+" "+words[i+1]
            print words[i]
            words.remove(words[i+1])
            n = n-1
        i += 1
    return words



def shuffle_data():
    """
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    """

    pos_path = "./data/pos"
    neg_path = "./data/neg"

    pos_review = folder_list(pos_path, 1)
    neg_review = folder_list(neg_path, -1)

    review = pos_review + neg_review
    random.shuffle(review)

    train_data = review[:1500]
    test_data = review[1500:]

    train_output = open("train_data", "wb")
    pickle.dump(train_data, train_output)
    train_output.close()

    test_output = open("test_data", "wb")
    pickle.dump(test_data, test_output)
    test_output.close()




	
'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''
 

if __name__ == "__main__":
    shuffle_data()