### Neural Network - MLP ###
# Usage:
# - create with or without random weights (default True)
# # net = NeuralNetwork(weights_random = True)
#
# - add layers (default layer: 1 neuron, 1 bias neuron, sigmoid activation function)
# # net.add(Layer(neurons_count=5,activation_fun="linear",add_bias=True))
#
# - print neural network architecture
# # print(net)
#
# - set all weights to given as parameter
# # net.set_weights(weights)
# weights are a list of np.arrays same as in 'print(net)'
#
# - set random weights for a given layer
# # net.set_weights_randomized(after_layer=3)
# if after_layer=None then all layers will be randomized
#
# - predict (feed forward)
# # net.predict(x)
# here, x can be a list of numbers (predict 1 value) or a list of lists (predict many)
# returns a list or a list of lists

import numpy as np
import math

class Layer:
                
    def _activation_function_linear(self,x):
        return x

    def _activation_function_sigmoid(self,x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    def _activation_function_tahn(self,x):
        e2x = np.exp(2*x)
        t = (e2x-1) / (1 + e2x)
        return t

    def _activation_function_relu(self,x):
        return max(0,x)

    
    def __init__(self,neurons_count=1,activation_fun="sigmoid",add_bias=True):
        self.add_bias = add_bias

        if self.add_bias:
            self.neurons_count = neurons_count + 1
            self.neurons_vals = np.zeros(self.neurons_count)
            self.neurons_vals[0] = 1
        else:
            self.neurons_count = neurons_count
            self.neurons_vals = np.zeros(self.neurons_count)

        self.set_activation_function(activation_fun)
        

    def set_activation_function(self,fun_name):
        self.activation_function_name = fun_name + " function"
        if fun_name == "sigmoid":
            self.activation_fun = np.vectorize(self._activation_function_sigmoid)
        elif fun_name == "linear":
            self.activation_fun = np.vectorize(self._activation_function_linear)
        elif fun_name == "tahn":
            self.activation_fun = np.vectorize(self._activation_function_tahn)
        elif fun_name == "relu":
            self.activation_fun = np.vectorize(self._activation_function_relu)
        else:
            self.activation_function_name = None
            raise Exception(f"Unknown activation function selected: {fun_name}\nAvailable functions are: 'sigmoid' and 'linear'.")

    def __str__(self):
        txt = f"Layer has {self.neurons_count} neurons"
        txt += " (including 1 bias neuron)" if self.add_bias else " (with no bias neuron)"
        txt += f" and activation function is '{self.activation_function_name}'"
        return txt
        

class NeuralNetwork:
    def __init__(self, weights_random = True):
        self.layers = []
        self.neuron_values = []
        self.weights = []
        self._weights_random = weights_random
    
    def __str__(self):
        txt = "Neural network layers:\n" 
        for i in range(len(self.layers)):
            txt += f"\tLayer {i+1}: {str(self.layers[i])}\n"
        txt += "Neural network weights:\n" 
        for i in range(len(self.weights)):
            txt += f"\tWeights {i+1}: {self.weights[i].shape} (input, output)\n{str(self.weights[i])}\n"
        return txt

    def add(self,new_layer:Layer): # one by one
        if len(self.layers) == 0:
            new_layer.set_activation_function('linear') # no activation for input layer
        self.layers.append(new_layer)
        if self._weights_random and len(self.layers) > 1:
            self.set_weights_randomized(after_layer = len(self.layers)-1)
            
    def set_weights_randomized(self,after_layer = None):
        if after_layer is not None and after_layer >= 1:
            size_input = self.layers[after_layer-1].neurons_count
            size_output = self.layers[after_layer].neurons_count
            size_output = size_output - 1 if self.layers[after_layer].add_bias else size_output
            if len(self.weights) < len(self.layers) - 1:
                self.weights.append(None) # make list longer
            self.weights[after_layer - 1] = np.random.randn(size_input,size_output)
        elif after_layer is None:
            for i in range(len(self.layers)):
                self.set_weights_randomized(i+1)
        else:
            raise Exception(f"Error initializing weights - wrong layer number '{after_layer}'\nif None then for all")

    def set_weights(self,weights): # for all layers simultaneously
        ### format: a list of matrices
        # Each matrix has 'size_input' rows and 'size_output' columns, for example:
        # np.random.randn(size_input,size_output)
        self.weights = weights

    def _predict_single(self,input_raw):
        input_array = np.array(input_raw)
        self.layers[0].neurons_vals[-input_array.shape[0]:] = input_array
        for i in range(len(self.weights)):
            layer_in = self.layers[i]
            layer_out = self.layers[i+1]
            weights = self.weights[i]
            multiplied = layer_in.neurons_vals @ weights
            layer_out.neurons_vals[-multiplied.shape[0]:] = layer_out.activation_fun(multiplied)
            self.layers[i+1] = layer_out

        last_layer = self.layers[-1]
        return list(last_layer.neurons_vals) if not last_layer.add_bias else list(last_layer.neurons_vals[1:])

    def _predict_list(self,many_inputs):
        results = []
        for single_input in many_inputs:
            results.append(self._predict_single(single_input))
        return results

    def predict(self,input_raw):
        if isinstance(input_raw[0],list):
            return self._predict_list(input_raw)
        elif isinstance(input_raw[0],float) or isinstance(input_raw[0],int):
            return self._predict_single(input_raw)