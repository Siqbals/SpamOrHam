import joblib
from neuralnet import predict_email

#improt model 
model = joblib.load("spam_classifier.pkl")

#import the trained wieght and bias 
w1 = model["W1"]
b1 = model["b1"]
w2 = model["W2"]
b2 = model["b2"]
Vectorizer = model["vectorizer"]

#loop through to test if email is spam 
loopvar = True

while loopvar:
    spamcontents = input("enter your spam email! \n")
    pred_class, probs = predict_email(spamcontents, Vectorizer, w1, b1, w2, b2)

    if pred_class == 0:
        print("This is NOT spam!")
    else:
        print("This IS spam!")

