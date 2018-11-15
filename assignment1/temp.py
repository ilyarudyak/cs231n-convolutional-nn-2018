import random
import numpy as np
from cs231n.softmax_utils import get_CIFAR10_data
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.gradient_check import grad_check_sparse, eval_numerical_gradient

# get data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)

# perform check
N = 5
W = np.random.randn(N, 10) * 0.0001
loss, grad = softmax_loss_vectorized(W, X_dev[:, 0:N], y_dev, 0.0)
f = lambda w: softmax_loss_vectorized(w, X_dev[:, 0:N], y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, num_checks=10)

grad_numerical_full = eval_numerical_gradient(f, W, verbose=True, h=0.00001)

