import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


mnist = load_digits()
X = np.array(mnist.data)
y = np.array(mnist.target)
m, n = X.shape
#print(X.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X,y , test_size=0.2)
X_train = X_train.T
Y_train = Y_train.T
#print(X_train.shape)
#print(X_test[0])
def init_params():
    W1 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b1 = np.random.randn(10,1)
    W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forw_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X)+b1
    a1 = ReLU(Z1)
    Z2 = W2.dot(a1)+b2
    a2 = softmax(Z2)
    return Z1, a1, Z2, a2
def deriv_ReLU(z):
    return (z > 0).astype(float)

def one_hot(y):
    one_hot_y = np.zeros((y.size,y.max()+1))
    one_hot_y[np.arange(y.size),y] = 1
    one_hot_y =one_hot_y.T
    return one_hot_y

def back_prop(Z1,a1,a2,W2,y,X):
    m = y.size
    one_hot_y = one_hot(y)
    dZ2 = a2 - one_hot_y
    dW2 = 1/m * dZ2.dot(a1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return db1 ,dW1, db2, dW2
def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.mean(predictions == y)


def upd_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1, b1, W2, b2
def grad_desc(X, Y, iter, alpha):
    W1 ,b1, W2, b2 = init_params()
    acc= []
    for i in range(iter):
        Z1 ,a1, Z2, a2 = forw_prop(W1, b1, W2, b2, X)
        db1, dW1, db2, dW2 = back_prop(Z1,a1,a2,W2,Y,X)
        W1, b1, W2, b2 = upd_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i%50 == 0:
            print("Iteration: ", i )
            print("accuracy: ", get_accuracy(get_predictions(a2), Y))
            acc.append(get_accuracy(get_predictions(a2), Y))

    return W1, b1, W2, b2
grad_desc(
    X_train, Y_train, 4000, 0.01
)



