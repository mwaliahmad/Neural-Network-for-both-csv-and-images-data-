import numpy as np
import pickle
import os
from PIL import Image


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def initialize_parameters(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = (
            np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        )
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)

    cache = (A_prev, W, b, activation_cache)
    return A, cache


def activation_backward(dA, cache, activation):
    A_prev, W, b, activation_cache = cache
    m = A_prev.shape[1]

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def load_parameters(file_path):
    with open(file_path, "rb") as f:
        parameters = pickle.load(f)
    return parameters


def save_parameters(parameters, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(parameters, f)
