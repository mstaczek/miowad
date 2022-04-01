### Neural Network - MLP ###
# Usage:
# - create with or without random weights (default True)
# # net = NeuralNetwork(weights_random = True)
#
# - add layers (default layer: 1 neuron, 1 bias neuron, sigmoid activation function)
# # net.add(Layer(neurons_count=5,activation_fun=ActivationSigmoid(),add_bias=True))
#
# - print neural network architecture
# # print(net)
#
# - set all weights to given as parameter
# # net.set_weights(weights)
# weights are a list of np.arrays same as in 'print(net)'
#
# - predict (feed forward)
# # net.predict(x)
# here, x can be a list of numbers (predict 1 value) or a list of lists (predict many)
# returns a list or a list of lists

import numpy as np
import math
from itertools import chain
from matplotlib import pyplot as plt
import copy


class ActivationLinear:
    def __init__(self):
        pass
    def __str__(self):
        return "linear"
    def forw(self,x):
        return x
    def grad(self,x):
        result = np.empty(x.shape[0])
        result.fill(1)
        return result

class ActivationSigmoid:
    def __init__(self):
        pass
    def __str__(self):
        return "sigmoid"
    def forw(self,x):
        def _forw(x): # more numerically stable
            if x >= 0:
                z = math.exp(-x)
                return 1 / (1 + z)
            else:
                z = math.exp(x)
                return z / (1 + z)
        return np.vectorize(_forw)(x)

    def grad(self,x):
        forw = self.forw(x)
        return forw * (1 - forw)

class ActivationTanh:
    def __init__(self):
        pass
    def __str__(self):
        return "tanh"
    def forw(self,x):
        return np.tanh(x)
    def grad(self,x):
        return 1 - np.tanh(x)**2

class ActivationReLU:
    def __init__(self):
        pass
    def __str__(self):
        return "relu"
    def forw(self,x):
        return x if x > 0 else 0
    def grad(self,x):
        return 1 if x > 0 else 0     

class ActivationSoftmax:
    def __init__(self):
        pass
    def __str__(self):
        return "softmax"
    def forw(self,x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    def grad(self,x):
        return None # not used - softmax is implemented to be used only with crossentropy loss in last layer

class Layer:
    def __init__(self,neurons_count,activation_fun=ActivationLinear(),add_bias=True):
        self.add_bias = add_bias
        self.neurons_count = neurons_count
        self.neurons_vals = np.zeros(self.neurons_count)
        if self.add_bias:
            self.neurons_count += 1
            self.neurons_vals = np.insert(self.neurons_vals,0,1)

        self.neurons_grad_vals = np.zeros(self.neurons_count)
        self.neurons_error_vals = np.zeros(self.neurons_count)
        self.activation_fun = activation_fun

    def make_first_layer(self):
        self.activation_fun = ActivationLinear()

    def __str__(self):
        txt = f"Layer has {self.neurons_count} neurons"
        txt += " (including 1 bias neuron)" if self.add_bias else " (with no bias neuron)"
        txt += f" and activation function is '{str(self.activation_fun)}'" 
        return txt

class LossMSE:
    def __init__(self):
        pass
    def __str__(self):
        return "MSE"
    def loss(self,real, pred):      
        if type(real[0]) == list:
            real2 = list(chain.from_iterable(real))
        else:
            real2 = real
        if type(pred[0]) == list:
            pred2 = list(chain.from_iterable(pred))
        else:
            pred2 = pred
        return np.square(np.subtract(real2,pred2)).mean() 
    def get_errors_for_last_layer(self, y_pred, y_real, grads_last_layer):
        return (y_pred - y_real) * grads_last_layer


class LossCrossEntropy:
    def __init__(self):
        pass
    def __str__(self):
        return "CrossEntropy"
    def loss(self,y_pred,y_true):
        epsilon = 1e-12
        predictions = np.clip(np.array(y_pred), epsilon, 1. - epsilon)
        targets = np.array(y_true)
        N = predictions.shape[0]
        ce = -np.sum(targets*np.log(predictions))/N
        return ce
    def get_errors_for_last_layer(self, y_pred, y_real, grads_last_layer):
        return y_pred - y_real
        

class NeuralNetwork:
    def __init__(self, weights_random = True, weights_randomizer = "xavier"):
        self.layers = []
        self.neuron_values = []
        self.weights = []
        self._weights_random = weights_random
        self._weights_randomizer = weights_randomizer
        self.training_history = {"weights_history":[],"loss_train":[],"loss_test":[],\
                                "f1_macro_train":[],"f1_macro_test":[]}
    
    def __str__(self):
        txt = "Neural network layers:\n" 
        for i in range(len(self.layers)):
            txt += f"\tLayer {i}: {str(self.layers[i])}\n"
        txt += "Neural network weights:\n" 
        for i in range(len(self.weights)):
            txt += f"\tWeights {i+1}: shape is {self.weights[i].shape} as (input, output) "
            if self.layers[i].add_bias:
                txt += f"where first element in each column is bias" 
            txt += f"\n{str(self.weights[i])}\n"
        return txt

    def add(self,new_layer:Layer): # one by one
        if len(self.layers) == 0:
            new_layer.make_first_layer() # no activation for input layer
        self.layers.append(new_layer)
        if self._weights_random and len(self.layers) > 1:
            self.set_weights_randomized(after_layer = len(self.layers)-1)

    def _randomize_weights(self,size_input,size_output):
        size = (size_input,size_output)
        if self._weights_randomizer == "uniform":
            return np.random.uniform(size=size)
        elif self._weights_randomizer == "normal":
            return np.random.normal(size=size)
        elif self._weights_randomizer == "xavier":
            limit = math.sqrt(6 / (size_input + size_output))
            return np.random.uniform(low=-limit,high=limit,size=size)
        elif self._weights_randomizer == "he":
            return np.random.normal(loc=0,scale=np.sqrt(2/size_input),size=size)
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
        # Note: first value in each column is bias (if present)
        self.weights = weights

    def _predict_single(self,input_raw):
        input_array = np.array(input_raw)
        self.layers[0].neurons_vals[-input_array.shape[0]:] = input_array
        for i in range(len(self.weights)):
            layer_in = self.layers[i]
            layer_out = self.layers[i+1]
            weights = self.weights[i]
            multiplied = layer_in.neurons_vals @ weights
            layer_out.neurons_vals[-multiplied.shape[0]:] = layer_out.activation_fun.forw(multiplied)
            layer_out.neurons_grad_vals[-multiplied.shape[0]:] = layer_out.activation_fun.grad(multiplied)
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

    def _backprop_calculate_errors(self,expected_output, loss_function):
        last_layer = self.layers[-1]
        last_layer.neurons_error_vals = loss_function.get_errors_for_last_layer(y_pred = last_layer.neurons_vals,\
                                                                                y_real = expected_output,\
                                                                                grads_last_layer = last_layer.neurons_grad_vals) # for mse
        for j in range(len(self.weights)-1,0,-1): # over all layers, updating errors except last layer 
            layer_in = self.layers[j]
            layer_out = self.layers[j+1]
            weights = self.weights[j]
            f_prim = layer_in.neurons_grad_vals
            errors_k_plus_one = layer_out.neurons_error_vals[-weights.shape[1]:]
            weight_k_plus_one = weights

            w_times_e = errors_k_plus_one @ weight_k_plus_one.T
            errors_k = f_prim * w_times_e

            layer_in.neurons_error_vals = errors_k
            self.layers[j] = layer_in
        
    def _backprop_calculate_gradients(self,samples_count):
        net_weights_grad_new = []
        for j in range(len(self.layers)-1):
            layer_in = self.layers[j]
            layer_out = self.layers[j+1]
            weights = self.weights[j]
            err = layer_out.neurons_error_vals[-weights.shape[1]:]
            l_in = layer_in.neurons_vals
            l_in_rep_t = np.repeat(l_in[:,np.newaxis],len(err),axis=1)
            net_weights_j_grad = l_in_rep_t * err / samples_count
            net_weights_grad_new.append(net_weights_j_grad)
        return net_weights_grad_new

    def _backprop_update_weights(self,learning_rate,weights_grad):
        for j in range(len(self.weights)):
            self.weights[j] -= learning_rate * weights_grad[j]
    
    def _get_current_loss(self,samples_count,batch_size,in_raw,out_raw, loss_function):
        errors_iter = []
        for batch_part_no in list(range(0,min(math.ceil(samples_count/batch_size),samples_count))):
            batch_in = in_raw[batch_part_no*batch_size:(batch_part_no+1)*batch_size]
            batch_out = out_raw[batch_part_no*batch_size:(batch_part_no+1)*batch_size]
            y_hat = self.predict(batch_in)
            errors_iter += [loss_function.loss(batch_out,y_hat)]
        return np.mean(errors_iter)
    
    def _print_formatted_loss(self,epoch,epochs,loss_train,loss_test=None,with_test=False, loss_function=None):
        result = f"Epoch:{epoch+1:>5}/{epochs},  {str(loss_function)} loss train:{round(loss_train,3):>9}"
        if with_test:
            result += f",  loss test:{round(loss_test,3):>9}"
        print(result)    

    def _f1_per_class(self, y_true, y_pred):
        TP = np.sum(np.multiply([i==True for i in y_pred], y_true))
        TN = np.sum(np.multiply([i==False for i in y_pred], [not(j) for j in y_true]))
        FP = np.sum(np.multiply([i==True for i in y_pred], [not(j) for j in y_true]))
        FN = np.sum(np.multiply([i==False for i in y_pred], y_true))
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        return (2 * precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0

    def _get_f1_macro(self,in_raw,out_raw):
        y_true = np.array(out_raw).argmax(axis=1)
        y_pred = np.argmax(self.predict(in_raw), axis=1)
        macro = []
        for i in np.unique(y_true):
            modified_true = [i==j for j in y_true]
            modified_pred = [i==j for j in y_pred]
            score = self._f1_per_class(modified_true, modified_pred)
            macro.append(score)
        return np.mean(macro)

    def _print_formatted_f1_macro(self,f1_train,f1_test=None):
        result = f"\t\tF1 macro train:{round(f1_train,3):>9}"
        if f1_test is not None:
            result += f",  F1 macro test:{round(f1_test,3):>9}"
        print(result)
    
    def _update_training_history(self,weights,loss_train,loss_test,f1_train,f1_test):
        self.training_history = {"weights_history":weights,"loss_train":loss_train,"loss_test":loss_test,\
                                "f1_macro_train":f1_train,"f1_macro_test":f1_test}  

    def _after_epoch_prints_and_data_saving(self,epoch,epochs,weights,loss_train,loss_test,f1_train,f1_test,loss_function):
        self._print_formatted_loss(epoch,epochs,loss_train[-1],loss_test[-1] if len(loss_test)>0 else None,with_test=len(loss_test)>0,
                                            loss_function=loss_function)
        if len(f1_train)>0:
            self._print_formatted_f1_macro(f1_train[-1],f1_test[-1] if len(f1_test)>0 else None)
        self._update_training_history(weights,loss_train,loss_test,f1_train,f1_test)
                                
    def train(self, train_in, train_out, test_in=None, test_out=None, loss_function=LossMSE(),learning_rate=0.01, epochs=1, \
                batch_size=32, with_moment=False,moment_decay=0.9,with_rms_prop=False,\
                rms_prop_decay=0.5,with_f1_macro=False):
        train_input_array = np.array(train_in)
        train_output_array = np.array(train_out)
        N = len(train_input_array)
        batch_size = min(batch_size,N) if batch_size > 0 else N
## plots  
        for_plot_weights = self.training_history["weights_history"]
        for_plot_loss_train = self.training_history["loss_train"]
        for_plot_loss_test = self.training_history["loss_test"]
        for_plot_f1_macro_train = self.training_history["f1_macro_train"]
        for_plot_f1_macro_test = self.training_history["f1_macro_test"]
        if test_present:= (test_in is not None and test_out is not None):
            N_test = len(test_in)
            batch_size_test = min(batch_size,N_test)
## plots update
        for_plot_weights += [copy.deepcopy(self.weights)]
        for_plot_loss_train += [self._get_current_loss(N,batch_size,train_in,train_out,loss_function)]
        if with_f1_macro:
            for_plot_f1_macro_train += [self._get_f1_macro(train_in,train_out)]
        if test_present:
            for_plot_loss_test += [self._get_current_loss(N_test,batch_size_test,test_in,test_out,loss_function)]
            if with_f1_macro:
                for_plot_f1_macro_test += [self._get_f1_macro(test_in,test_out)]
## backprop
        if with_moment:
            moment_weights = [np.zeros(weights_i.shape) for weights_i in self.weights]
        if with_rms_prop:
            rms_prop_weights = [np.zeros(weights_i.shape) for weights_i in self.weights]            
        for epoch in range(epochs):
            for batch_part_no in list(range(0,min(math.ceil(N/batch_size),N))):
                batch_train_input_array = train_input_array[batch_part_no*batch_size:(batch_part_no+1)*batch_size]
                batch_train_output_array = train_output_array[batch_part_no*batch_size:(batch_part_no+1)*batch_size]
                weights_grad = [np.zeros(weights_i.shape) for weights_i in self.weights]            
                for i in range(len(batch_train_input_array)):
                    self._predict_single(batch_train_input_array[i])
                    self._backprop_calculate_errors(batch_train_output_array[i], loss_function)
                    weights_grad_new = self._backprop_calculate_gradients(samples_count=len(batch_train_input_array))
                    weights_grad = [weights_grad[j] + weights_grad_new[j] for j in range(len(self.weights))]
                if with_moment:
                    moment_weights = [moment_weights[j] * moment_decay + weights_grad[j] for j in range(len(self.weights))]
                    self._backprop_update_weights(learning_rate,moment_weights)
                elif with_rms_prop:
                    rms_prop_weights = [rms_prop_decay * rms_prop_weights[j] + (1 - rms_prop_decay) * weights_grad[j]**2 for j in range(len(self.weights))]
                    weights_grad = [weights_grad[j]/rms_prop_weights[j]**0.5 for j in range(len(self.weights))]
                    self._backprop_update_weights(learning_rate,weights_grad)
                else:
                    self._backprop_update_weights(learning_rate,weights_grad)
## plots update
            for_plot_weights += [copy.deepcopy(self.weights)]
            for_plot_loss_train += [self._get_current_loss(N,batch_size,train_in,train_out,loss_function)]
            if with_f1_macro:
                for_plot_f1_macro_train += [self._get_f1_macro(train_in,train_out)]
            if test_present:
                for_plot_loss_test += [self._get_current_loss(N_test,batch_size_test,test_in,test_out,loss_function)]
                if with_f1_macro:
                    for_plot_f1_macro_test += [self._get_f1_macro(test_in,test_out)]
## prints                
            if epoch % math.ceil(epochs/10) == 0:
                self._after_epoch_prints_and_data_saving(epoch,epochs,for_plot_weights,for_plot_loss_train,for_plot_loss_test,
                                                            for_plot_f1_macro_train,for_plot_f1_macro_test,loss_function)               
        self._after_epoch_prints_and_data_saving(epoch,epochs,for_plot_weights,for_plot_loss_train,for_plot_loss_test,
                                                    for_plot_f1_macro_train,for_plot_f1_macro_test,loss_function) 


    def get_training_history(self):
        return self.training_history

    def plot_training_history(self):
        if len(self.training_history["loss_train"]) == 0:
            raise Exception("Training history is empty")
        plt.figure()
        plt.plot(self.training_history["loss_train"],label="Loss train")
        if len(self.training_history["loss_test"]) > 0:
            plt.plot(self.training_history["loss_test"],label="Loss test")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss changes during training")
        plt.grid()
        plt.show()

        if len(self.training_history["f1_macro_train"]) > 0:
            plt.figure()
            plt.plot(self.training_history["f1_macro_train"],label="f1 macro train")
            if len(self.training_history["f1_macro_test"]) > 0:
                plt.plot(self.training_history["f1_macro_test"],label="f1 macro test")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("F1 macro")
            plt.title("F1 macro changes during training")
            plt.grid()
            plt.show()