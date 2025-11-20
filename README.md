# SpamOrHam

SpamOrHam is a **from-scratch 2-layer neural network** for email spam detection.  
Instead of using high-level deep learning libraries like PyTorch or TensorFlow, the model is implemented **entirely with NumPy**, including:

- Manual forward pass
- Custom **sigmoid** and **softmax** activations
- **Cross-entropy loss**
- Full **backpropagation** implementation
- **Gradient descent** weight and bias updates

The only “helper” libraries are:
- `scikit-learn` for **TF-IDF text vectorization** and train/test splitting  
- `kagglehub` + `pandas` to download and load the spam dataset  
- `joblib` to save the trained model

The end result is an interactive CLI tool where you can type an email and the model predicts:

> **This IS spam** or **This is NOT spam**

---

## 🧠 Project Overview

### Architecture

The neural network is a simple **2-layer fully connected network**:

1. **Input layer**  
   - Emails are cleaned and converted into a **TF-IDF feature vector**  
   - Shape: `(num_features,)`

2. **Hidden layer**
   - 100 neurons (`hunits = 100`)
   - Linear transform: `Z1 = X @ W1.T + b1`
   - Activation: **sigmoid**  
   - Implemented manually with NumPy

3. **Output layer**
   - 2 neurons → `[not_spam, spam]`
   - Linear transform: `Z2 = A1 @ W2.T + b2`
   - Activation: **softmax** to get class probabilities  
   - Loss: **cross-entropy**

### From-Scratch Backpropagation

The gradients are computed by hand (not with autograd):

- Output layer gradient:  
  \[
  \nabla L = \frac{\hat{y} - y}{N}
  \]
- Weight and bias updates:
  - `dW2 = Lgrad.T @ A1`
  - `db2 = Lgrad.sum(axis=0)`
  - `errprop = (Lgrad @ W2) * sigmoid'(Z1)`
  - `dW1 = errprop.T @ X`
  - `db1 = errprop.sum(axis=0)`
- Parameters are updated with **gradient descent**:
  - `W_new = W_old - lr * dW`
  - `b_new = b_old - lr * db`

All of this logic lives in:
- `forwardpass(...)`
- `backprop(...)`
- `update(...)`

in **`neuralnet.py`**.

---

## 📂 Project Structure

```text
SpamOrHam/
├─ mainprog.py        # CLI interface to load trained model and classify new emails
├─ neuralnet.py       # Full from-scratch NN: TF-IDF, forward pass, backprop, training loop, saving model
└─ README.md
