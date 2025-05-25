import numpy as np
import math

def naive_bayes(X_train, y_train, x_test):
    occurance = np.bincount(y_train) #counts the number 0 as well so the first element on occurance will be 0
    occurance = np.delete(occurance, 0) # so we delete the first number 0
    total = len(y_train)
    prob_class = []
    #as the data in y_train is discrete, we find the probabilties using this method
    for i in range(len(occurance)):
            prob_class.append(occurance[i]/total)
    
    type_idx = {}
    for i in range(1, len(occurance) + 1):
          type_idx[i] = np.where(y_train == i)
    
    
    #we want a subset of rows where the glass is 1 - why because that is the class
    # for each class we want to add up the probabilities over all the columns

    prob = 1
    prob_arr = []
    for i in range(1,8):
        X_subset = X_train[y_train == i]
        for j in range(X_subset.shape[1]):
            prob = prob * conditional_prob(X_subset,j,x_test[j]) #multiplies each conditional probabilities according to Bayes theorem
        prob = prob *prob_class[i - 1]
        prob_arr.append(prob)
        prob = 1
    
    higest_prob = np.max(prob_arr)
    index = np.where(prob_arr == higest_prob)
    return index[0][0] + 1

#this function work out the conditional probabilties within the Bayes theorem
def conditional_prob(subset, index, x_test):
    arr = []
    for i in range(subset.shape[0]): #number of rows
        arr.append(subset.iloc[i, index])
    
    #calculation for the mean
    sum = 0
    for x in arr:
         sum = sum + x
    
    if len(arr) == 0:
         mean = 0
    else: 
        mean = sum / len(arr)

    #calculation for the variance
    variance = 0
    for x in arr:
         term = ((x-mean)**2)/len(arr)
         variance = variance + term
    
    #uses the probability density function as it the data from the features are continuous
    if math.sqrt(math.pi*2*variance) == 0:
         term_one = 0
    else:
        term_one = 1/(math.sqrt(math.pi*2*variance))
    term_two = (x_test-mean)**2
    term_three = 2*variance
    if term_three == 0:
         term_four = 0
    else:
         term_four = term_two/term_three
    term_five = math.exp(-term_four)
    prob = term_one*term_five
    return prob
