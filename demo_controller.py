# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller
# (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np
from typing import List


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))


# implements controller structure for player
class PlayerController(Controller):
    def __init__(self, hidden_layer_sizes: List[int]):
        self.hidden_layer_sizes = hidden_layer_sizes

    def control(self, inputs, controller):
        """

        :param inputs: states of the env
        :param controller: kind of weight
        :return:
        """
        # Normalises the input using min-max scaling
        # TODO: replace with sklearn
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        # if self.n_hidden[0] > 0:
        # Preparing the weights and biases from the controller of layer 1

        # Biases for the n hidden neurons
        bias1 = controller[:self.hidden_layer_sizes[0]].reshape(1, self.hidden_layer_sizes[0])
        # Weights for the connections from the inputs to the hidden nodes
        weights1_slice = len(inputs) * self.hidden_layer_sizes[0] + self.hidden_layer_sizes[0]
        weights1 = controller[self.hidden_layer_sizes[0]:weights1_slice].reshape((len(inputs), self.hidden_layer_sizes[0]))

        # Outputs activation first layer.
        output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

        # Preparing the weights and biases from the controller of layer 2
        bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
        weights2 = controller[weights1_slice + 5:].reshape((self.hidden_layer_sizes[0], 5))

        # Outputting activated second layer. Each entry in the output is an action
        output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        # else:
        #     bias = controller[:5].reshape(1, 5)
        #     weights = controller[5:].reshape((len(inputs), 5))
        #
        #     output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        left = self.make_decision(output[0])
        right = self.make_decision(output[1])
        jump = self.make_decision(output[2])
        shoot = self.make_decision(output[3])
        release = self.make_decision(output[4])

        return [left, right, jump, shoot, release]

    def make_decision(self, output):
        if output > 0.5:
            action = 1
        else:
            action = 0
        return action


# implements controller structure for enemy
class enemy_controller(Controller):
    def __init__(self, _n_hidden):
        # Number of hidden neurons
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
        # Normalises the input using min-max scaling
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))

        if self.n_hidden[0] > 0:
            # Preparing the weights and biases from the controller of layer 1

            # Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            # Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs) * self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs), self.n_hidden[0]))

            # Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

            # Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1, 5)
            weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0], 5))

            # Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2) + bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))

            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # takes decisions about sprite actions
        if output[0] > 0.5:
            attack1 = 1
        else:
            attack1 = 0

        if output[1] > 0.5:
            attack2 = 1
        else:
            attack2 = 0

        if output[2] > 0.5:
            attack3 = 1
        else:
            attack3 = 0

        if output[3] > 0.5:
            attack4 = 1
        else:
            attack4 = 0

        return [attack1, attack2, attack3, attack4]
