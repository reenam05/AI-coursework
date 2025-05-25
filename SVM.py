import numpy as np
#compares one class to all classes
def fit(X,y, num):
    X = X.to_numpy()
    y = y.to_numpy()
    y = np.where(y == num, -1, 1)
    w = np.zeros(X.shape[1])
    b = 0
    #hyperparameters than can be modified to affect the decision boundary to increase/decrease the accuracy of the model
    iteration = 1000
    lambd = 0.01
    rate = 0.01
    i = 0
    # contructs the equation for the decision boundary / finds the values w and b 
    for index in range(iteration):
        for row in X:
            dot_prod = np.dot(row, w) # as there is 9 features in one row of data we need to use the dot product 
            boundary = (y[i] * dot_prod) - b
            if boundary >= 1:# checks to see if the support vectors are within the margins
                w = w - rate*2*lambd*w
            else:
               dot_prod_two = np.dot(row, y[i])
               w = w - rate * (2*lambd* w - dot_prod_two)
               b = b - rate *y[i]

            i = i + 1
        i = 0
    return w, b # optimum values to construct the decision boundary for our data

def prediction(X, arr):
    sign_arr = []
    for x in arr: # find the prediction for the 7 sets of w and b
        pred = np.dot(X,x[0]) - x[1]
        pred_sign = np.sign(pred)
        sign_arr.append(pred_sign)
    sign_arr = np.array(sign_arr)##currently a list therefore converts to a numpy array
    pred_arr = np.where(sign_arr == -1)# finds what the model has predicted the value
    
    #as we used np.where the data is in a tuple and an array so we need to flatten it
    #some of the data is not predicted as anything, as the training dataset has no data for 4, we class this data as 4
    pred_arr = pred_arr[0]
    if pred_arr.size > 1:
        pred_arr = pred_arr.tolist() 
        pred_arr = pred_arr[1] + 1 #need to add 1 because the indexes 0-6 represent the classes 1-7
    elif pred_arr.size > 0:
        pred_arr = pred_arr.tolist()
        if len(pred_arr) > 0:
            pred_arr = pred_arr[0] + 1
    else:
        pred_arr = 4
    
    
    return pred_arr


