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

    def _activation_function_tanh(self,x):
        e2x = np.exp(2*x)
        t = (e2x-1) / (1 + e2x)
        return t

    def _activation_function_relu(self,x):
        return x if x > 0 else 0

    def _activation_function_linear_grad(self,x):
        return 1

    def _activation_function_sigmoid_grad(self,x):
        sigmoid_val = self._activation_function_sigmoid(x)
        return sigmoid_val * (1 - sigmoid_val)

    def _activation_function_tanh_grad(self,x):
        tanh_val = self._activation_function_tanh(x)
        return 1 - tanh_val**2

    def _activation_function_relu_grad(self,x):
        return 1 if x > 0 else 0

    def __init__(self,neurons_count=1,activation_fun="sigmoid",add_bias=True):
        self.add_bias = add_bias
        self.neurons_count = neurons_count
        self.neurons_vals = np.zeros(self.neurons_count)
        if self.add_bias:
            self.neurons_count += 1
            self.neurons_vals = np.insert(self.neurons_vals,0,1)

        self.neurons_grad_vals = np.zeros(self.neurons_count)
        self.neurons_error_vals = np.zeros(self.neurons_count)
        self.set_activation_function(activation_fun)

    def set_activation_function(self,fun_name):
        if fun_name == "sigmoid":
            chosen_function = self._activation_function_sigmoid
            chosen_function_grad = self._activation_function_sigmoid_grad
        elif fun_name == "linear":
            chosen_function = self._activation_function_linear
            chosen_function_grad = self._activation_function_linear_grad
        elif fun_name == "tanh":
            chosen_function = self._activation_function_tanh
            chosen_function_grad = self._activation_function_tanh_grad
        elif fun_name == "relu":
            chosen_function = self._activation_function_relu
            chosen_function_grad = self._activation_function_relu_grad
        else:
            self.activation_function_name = None
            raise Exception(f"Unknown activation function selected: {fun_name}\nAvailable functions are: 'sigmoid', 'linear', 'tanh' and 'relu'.")
        self.activation_function_name = fun_name + " function"
        self.activation_fun = np.vectorize(chosen_function)
        self.activation_fun_grad = np.vectorize(chosen_function_grad)

    def __str__(self):
        txt = f"Layer has {self.neurons_count} neurons"
        txt += " (including 1 bias neuron)" if self.add_bias else " (with no bias neuron)"
        txt += f" and activation function is '{self.activation_function_name}'"
        return txt
        

class NeuralNetwork:
    def __init__(self, weights_random = True, weights_randomizer = "uniform"):
        self.layers = []
        self.neuron_values = []
        self.weights = []
        self._weights_random = weights_random
        self._weights_randomizer = weights_randomizer
    
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

    def _randomize_weights(self,size_input,size_output):
        if self._weights_randomizer == "uniform":
            return np.random.uniform(size=(size_input,size_output))
        elif self._weights_randomizer == "xavier":
            limit = math.sqrt(6 / (size_input + size_output))
            return np.random.uniform(low=-limit,high=limit,size=(size_input,size_output))
        elif self._weights_randomizer == "he":
            return np.random.normal(loc=0,scale=np.sqrt(2/size_input),size=(size_input,size_output))
        else:
            raise Exception(f"Unknown weights randomizer: {self._weights_randomizer}")
            
    def set_weights_randomized(self,after_layer = None):
        if after_layer is not None and after_layer >= 1:
            size_input = self.layers[after_layer-1].neurons_count
            size_output = self.layers[after_layer].neurons_count
            size_output = size_output - 1 if self.layers[after_layer].add_bias else size_output
            if len(self.weights) < len(self.layers) - 1:
                self.weights.append(None) # make list longer
            self.weights[after_layer - 1] = self._randomize_weights(size_input,size_output)
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
            layer_out.neurons_grad_vals[-multiplied.shape[0]:] = layer_out.activation_fun_grad(multiplied)
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


    def train(self, input_raw, expected_output_raw, learning_rate = 0.1, epochs = 1, verbose = False):
        input_array = np.array(input_raw)
        expected_output_array = np.array(expected_output_raw)
        if len(input_array) != len(expected_output_array):
            raise Exception("Input and expected output must be of the same length")
        N = len(input_array)
        out_length = len(expected_output_array[0])
        for epoch in range(epochs):
            weights_grad = [np.zeros(weights_i.shape) for weights_i in self.weights]            
            for i in range(N):
                self._predict_single(input_array[i])
                last_layer = self.layers[-1]
                last_layer.neurons_error_vals[-out_length:] = \
                    (last_layer.neurons_vals - expected_output_array[i]) * last_layer.neurons_grad_vals
                for j in range(len(self.weights)-1,0,-1): # over all layers, updating errors except last layer 
                    layer_in = self.layers[j]
                    layer_out = self.layers[j+1]
                    weights = self.weights[j]
                    f_prim = layer_in.neurons_grad_vals
                    errors_k_plus_one = layer_out.neurons_error_vals
                    weight_k_plus_one = weights
                    errors_k = f_prim * (weight_k_plus_one @ errors_k_plus_one).T

                    layer_in.neurons_error_vals[-errors_k.shape[0]:] = errors_k
                    self.layers[j] = layer_in

                # update weights after every example
                # for j in range(len(self.layers)-1):
                #     layer_in = self.layers[j]
                #     layer_out = self.layers[j+1]
                #     weights = self.weights[j]
                #     err = layer_out.neurons_error_vals
                #     l_in = layer_in.neurons_vals
                #     net_weights_j_grad = err * np.repeat(l_in[:,np.newaxis],len(err),axis=1)
                #     weights -= learning_rate * net_weights_j_grad
                #     self.weights[j] = weights

                # update grad, but weights after all samples
                for j in range(len(self.layers)-1):
                    layer_in = self.layers[j]
                    layer_out = self.layers[j+1]
                    weights = self.weights[j]
                    err = layer_out.neurons_error_vals
                    l_in = layer_in.neurons_vals
                    net_weights_j_grad = err * np.repeat(l_in[:,np.newaxis],len(err),axis=1)
                    weights_grad[j] += net_weights_j_grad

            # update weights after all samples
            for j in range(len(self.layers)-1):
                self.weights[j] -= learning_rate * weights_grad[j]
            if verbose:
                print(f"Epoch {epoch}, Weights: {self.weights}")
        if verbose:
            return self.weights
        else:
            return


