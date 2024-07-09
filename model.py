import numpy as np
from utils import (
    initialize_parameters,
    activation_forward,
    activation_backward,
)


class Model:
    def __init__(
        self,
        X,
        Y,
        val_X,
        val_Y,
        layers_config,
        learning_rate,
        num_iterations,
        thread=None,
    ):
        self.thread = thread
        self.X = X
        self.Y = Y
        self.val_X = val_X
        self.val_Y = val_Y
        self.layers_config = layers_config
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.costs = []
        self.val_costs = []

    def train(self, progress_callback=None):
        np.random.seed(1)
        self.layers_config.insert(0, (self.X.shape[0], "relu"))
        parameters = initialize_parameters([layer[0] for layer in self.layers_config])

        for i in range(0, self.num_iterations + 1):
            AL, caches = self.forward_propagation(self.X, parameters)
            cost = self.calculate_cost(AL, self.Y)
            self.costs.append(cost)

            AL_val, _ = self.forward_propagation(self.val_X, parameters)
            val_cost = self.calculate_cost(AL_val, self.val_Y)
            self.val_costs.append(val_cost)

            grads = self.backward_propagation(AL, caches)
            parameters = self.update_parameters(parameters, grads)

            if self.thread is not None and i % 100 == 0:
                self.thread.progress_callback(i, cost, self.costs, self.val_costs)
                print(
                    f"Cost after iteration {i}: {cost} (Train) {val_cost} (Validation)"
                )

        return parameters

    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L + 1):
            A_prev = A
            A, cache = activation_forward(
                A_prev,
                parameters["W" + str(l)],
                parameters["b" + str(l)],
                activation=self.layers_config[l][1],
            )
            caches.append(cache)

        assert A.shape == (1, X.shape[1])
        return A, caches

    def backward_propagation(self, AL, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = self.Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(L)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = activation_backward(
                dAL if l == L - 1 else grads["dA" + str(l + 1)],
                current_cache,
                activation=self.layers_config[l + 1][1],
            )
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def calculate_cost(self, AL, Y):
        m = Y.shape[1]
        cost = (1.0 / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
        cost = np.squeeze(cost)
        assert cost.shape == ()
        return cost

    def update_parameters(self, parameters, grads):
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l + 1)] = (
                parameters["W" + str(l + 1)]
                - self.learning_rate * grads["dW" + str(l + 1)]
            )
            parameters["b" + str(l + 1)] = (
                parameters["b" + str(l + 1)]
                - self.learning_rate * grads["db" + str(l + 1)]
            )
        return parameters

    def predict(self, X, parameters):
        m = X.shape[1]
        p = np.zeros((1, m))

        probas, _ = self.forward_propagation(X, parameters)

        for i in range(0, probas.shape[1]):
            p[0, i] = 1 if probas[0, i] > 0.5 else 0

        return p
