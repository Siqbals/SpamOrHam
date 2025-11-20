'''
creating a neural network as such 
1 - input layer 
2 - hidden layer 
3 - output layer 

purpose - email spam detection

layer 1:
take string input, split characters all via space (specific word combinations = spam)
organize matrix, m columns, each column is a string input 

layer 2:
pass input into computations 
Y = Wx + b (W = wieght, b = bias, x = input matrix)
ReLU - activation function 

layer 3:
again y = Wx + b
activation is softmax, to get the final output value 
'''

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import kagglehub 
import pandas as pd  
import re 
import os
import joblib 

path = kagglehub.dataset_download('venky73/spam-mails-dataset')

#text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)   #remove punctuation 
    return text.strip() 

#function one - creating the matrix 
#columns are each email 
def matrix_maker(path, max_features=5000):
    """
    Loads the spam_ham_dataset.csv, cleans the text, and vectorizes it into a TF-IDF matrix.
    
    Parameters:
        path (str): Path to the folder containing 'spam_ham_dataset.csv'.
        max_features (int): Maximum vocabulary size for TF-IDF.

    Returns:
        X (np.ndarray): TF-IDF matrix (num_emails x num_features)
        vectorizer (TfidfVectorizer): Fitted vectorizer (for later use)
    """
    #Load dataset
    csv_path = os.path.join(path, "spam_ham_dataset.csv")
    reader = pd.read_csv(csv_path)
    dataset = reader['text'].astype(str).values
    print(reader[['label', 'label_num']].head())

    corpus = [clean_text(email) for email in dataset] #organize into clean strings 

    #convert the words into mathematical values to pass into neural network 
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()

    # get true values 
    y = reader['label_num'].values.astype(int)
    num_classes = len(np.unique(y))
    ytrue = np.eye(num_classes)[y]  # converts [0,1,0,1,...] → [[1,0],[0,1],[1,0],[0,1],...]


    #return matrix and vectorizer (assigns wieghts to text to tell network which stuff is important)
    return X, vectorizer, ytrue

def softmax(X):
    # X: (num_samples, num_outputs)
    # subtract max per row for numerical stability
    X_shifted = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X_shifted)
    probs = exp_X / np.sum(exp_X, axis=1, keepdims=True)
    return probs

#neural network itself 
def neuralnet(X):
    #set up the neural network 
    featurecont = X.shape[1]  #feature count 

    #define variables
    hunits = 100 #hidden layer neuron count 
    rans = np.random.default_rng(seed=0) 

    '''
    layer 1:
    Y = Wx+b
    sigmoid activation
    '''
    #initialize wieghts and bias (randomize)
    W1 = rans.normal(0.0, np.sqrt(2.0/featurecont), size=(hunits, featurecont)).astype(np.float32)
    b1 = np.zeros(hunits, dtype=np.float32)
    Y1 = X @ W1.T + b1   #first pass 
    activated = 1 / (1 + np.exp(-Y1))   #activation 

    '''
    layer 2:
    Y = Wx+b
    softmax activation (classification)
    '''
    #initialize wieghts and bias (randomize)
    W2 = rans.normal(0.0, np.sqrt(2.0/featurecont), size=(2, hunits)).astype(np.float32)
    b2 = np.zeros(2, dtype=np.float32)
    Y2 = activated @ W2.T + b2   #first pass 
    classified = softmax(Y2)

    return classified

def initializenet(X):
    #define variables 
    featurecont = X.shape[1]  #feature count 
    hunits = 100 #hidden layer neuron count 
    rans = np.random.default_rng(seed=0)

    #initialize first wieghts and biases 
    W1 = rans.normal(0.0, np.sqrt(2.0/featurecont), size=(hunits, featurecont)).astype(np.float32)
    b1 = np.zeros(hunits, dtype=np.float32)

    #initalize second wieghts and biases
    W2 = rans.normal(0.0, np.sqrt(2.0/hunits), size=(2, hunits)).astype(np.float32)
    b2 = np.zeros(2, dtype=np.float32) 

    return W1,b1,W2,b2
     


'''
forwardpass - compute forward pass of neural network 
returns:
X - the input matrix 
Y1 - Y = Wx+b of input matrix
activated1 - activated neurons for layer 1 
Y2 - Y = Wx+b of activated neurons 
classified - activated neurons of layer 2 (classifications)
'''
#sigmoid derivative (used for back prop)
def sigder(x):
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)


def forwardpass(X, w1, b1, w2, b2):
    
    #first pass (Y = Wx+b -> sigmoid activation)
    Y1 = X @ w1.T + b1
    activated1 = 1 / (1 + np.exp(-Y1))

    #second pass (Y = Wx+b -> softmax activation)
    Y2 = activated1 @ w2.T + b2 
    classified = softmax(Y2)

    return X,Y1,activated1,Y2,classified


# loss function
def lossfunc(output, corvals):
    eps = 1e-9
    predicted = np.clip(output, eps, 1. - eps)
    return -np.sum(corvals * np.log(predicted)) / corvals.shape[0]

def accuracy(pred, y_true):
    """
    pred:     predicted probabilities (N, C)
    y_true:   one-hot true labels (N, C)
    """
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels)

'''
backward pass logic:
calculating a gradient:
1. gradient of Loss - (true value - predicted value)
2a. gradient of W2 - (activated neurons from prev layer * gradient value in front of it (loss gradient))
2b. gradient of b2 - gradient of value in front of it (gradient of loss)

3. calculate error propogation (will pass this value to the layer behind (hidden layer)) - (wieght matrix (transposed) * gradient of value in front of it (loss grad) * sigmoid derivative (Wx+b))

4a. gradient of w1 - (activated neurons from prev layer (original input) * gradient value in front of it (step 3))
4b. gradient of b1 - gradient value in front of it (step 3)

calculating update:
select learning rate n (e.g. n=0.1)
new W2 = oldW2 * n(W2 gradient)
new b2 = oldb2 * n(b2 gradient)

new W1 = oldW1 * n(W1 gradient)
new b1 = oldb1 * n(b1 gradient)
'''

def backprop(X, y_true, y_pred, Z1, A1, W2):
    """
    Perform backpropagation for a 2-layer neural network.
    Returns:
        dW1 : (D, H) gradient of loss w.r.t W1
        db1 : (1, H) gradient of loss w.r.t b1
        dW2 : (H, C) gradient of loss w.r.t W2
        db2 : (1, C) gradient of loss w.r.t b2
    """
    N = X.shape[0]  # batch size

    #Gradient at the output layer (softmax + cross-entropy)
    # This is dL/dZ2
    Lgrad = (y_pred - y_true) / N            

    #2. Gradients for W2 and b2
    # dL/dW2 = A1^T @ dZ2
    w2grad = Lgrad.T @ A1                      

    #dL/db2 = sum_i dZ2_i
    b2grad = Lgrad.sum(axis=0)   

    #3. Backpropagate to hidden layer
    # dL/dA1 = dZ2 @ W2^T
    errprop = (Lgrad @ W2) * sigder(Z1)                                      

    #4. Gradients for W1 and b1
    #dL/dW1 = X^T @ dZ1
    w1grad = errprop.T @ X                    

    #dL/db1 = sum_i dZ1_i
    b1grad = errprop.sum(axis=0, keepdims=True)   

    return w1grad, b1grad, w2grad, b2grad

def update(w1,b1,w2,b2,w1grad,b1grad,w2grad,b2grad,learningrate):
    #update w2
    neww2 = w2 - learningrate*w2grad
    
    #update b2 
    newb2 = b2 - learningrate*b2grad
    
    #update w1
    neww1 = w1 - learningrate*w1grad

    #update b1
    newb1 = b1 - learningrate*b1grad

    return neww2,newb2,neww1,newb1

#predictor - predict an actual email 
def predict_email(email_text, vectorizer, w1, b1, w2, b2):
    #Clean text
    cleaned = clean_text(email_text)

    #Vectorize - mathematical representations 
    X_email = vectorizer.transform([cleaned]).toarray()   # shape (1, D)

    #Forward pass (same logic as forwardpass, but for one sample)
    Y1 = X_email @ w1.T + b1          
    A1 = 1 / (1 + np.exp(-Y1))        
    Y2 = A1 @ w2.T + b2               
    probs = softmax(Y2)               

    #Get predicted class index
    pred_class = np.argmax(probs, axis=1)[0]

    return pred_class, probs[0]

#evaluate accuracy function 
def eval_on_subset(X_data, y_data, subset_size=128):
    n = X_data.shape[0]
    size = min(subset_size, n)
    idx = np.random.choice(n, size=size, replace=False)
    _, _, _, _, preds = forwardpass(X_data[idx], w1, b1, w2, b2)
    return accuracy(preds, y_data[idx])



if __name__ == "__main__":
    # loop repeat 
    path = kagglehub.dataset_download('venky73/spam-mails-dataset') # import dataset
    X, Vectorizer, truth = matrix_maker(path) # create the matrix from the dataset
    w1,b1,w2,b2 = initializenet(X) # initialize network

    #specify train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, truth, test_size=0.2, random_state=42, shuffle=True
    )

    #specify training parameters 
    num_epochs   = 100
    batch_size   = 128
    learningrate = 0.1
    N_train      = X_train.shape[0]

    indices = np.arange(N_train)

    for epoch in range(num_epochs):
        #Shuffle ONLY the training data each epoch
        indices = np.random.permutation(N_train)
        X_train = X_train[indices]
        y_train = y_train[indices]

        #Mini-batch training on training set
        for start in range(0, N_train, batch_size):
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            Xb, Y1, A1, Y2, classified = forwardpass(X_batch, w1, b1, w2, b2)
            w1grad, b1grad, w2grad, b2grad = backprop(X_batch, y_batch, classified, Y1, A1, w2)
            w2, b2, w1, b1 = update(w1, b1, w2, b2, w1grad, b1grad, w2grad, b2grad, learningrate)

        #Eval on small random batches from train and test
        train_acc = eval_on_subset(X_train, y_train, subset_size=128)
        test_acc  = eval_on_subset(X_test,  y_test,  subset_size=128)

        print(
            f"epoch {epoch+1:03d} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

    #save the neural network
    model = {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "vectorizer": Vectorizer
    }

    joblib.dump(model, "spam_classifier.pkl")










