import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    D, C = W.shape
    N, _ = X.shape

    for n in range(N):

        # forward pass
        S = X[n, :].dot(W)  # (1, C)
        S -= np.max(S)  # for numerical stability
        P = np.exp(S) / np.sum(np.exp(S))  # (1, C)

        # loss calculation
        loss -= np.log(P[y[n]])

        # backward pass
        for d in range(D):
            for c in range(C):
                if y[n] == c:
                    dW[d, c] += (P[c] - 1) * X[n, d]
                else:
                    dW[d, c] += P[c] * X[n, d]

    loss /= N
    dW /= N

    # regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # forward pass
    N, _ = X.shape
    S = X.dot(W)
    S -= np.max(S, axis=1, keepdims=True)
    S_exp = np.exp(S)
    P = S_exp / np.sum(S_exp, axis=1, keepdims=True)

    # loss calculation
    loss = -np.sum(np.log(P[np.arange(N), y]))
    loss /= N

    # backward pass
    P[np.arange(N), y] -= 1
    dW = X.T.dot(P)
    dW /= N

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
