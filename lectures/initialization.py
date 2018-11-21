import numpy as np
import matplotlib.pyplot as plt

D = np.random.randn(1000, 500)
hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh'] * len(hidden_layer_sizes)

act = {'tanh': lambda x: np.tanh((x))}
Hs = {}
for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i-1]
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in, fan_out) * 0.01

    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs[i] = H

layer_means = [f'{np.mean(H):.6f}' for H in Hs.values()]
layer_stds = [f'{np.std(H):.6f}' for H in Hs.values()]


if __name__ == '__main__':
    print(layer_means)
    print(layer_stds)
