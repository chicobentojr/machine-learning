import os
import logging
import math
import random
import time
import click
import click_log
import pandas as pd
import matrix as mt
import numpy

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
#logger.info('Testing forest with {} instances'.format(len(test)))

#-------------------------------------------------------
# Auxiliar functions

def gaussian(x):
    return 1/(1 + math.exp(-x))

def dot_product(array1, array2):
    return numpy.dot(array1, array2)

#-------------------------------------------------------

class TrainingExample:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DataSet:
    def __init__(self, data_set_filename, num_entries):
        df = pd.read_csv(data_set_filename, sep=";|,", header=None, engine='python')
        (numRows, numCols) = df.shape

        self.examples = []

        for r in range(0, numRows):
            values = [x for x in df.iloc[r].tolist() if str(x) != 'nan']
            attribute_values = values[0 : num_entries]
            expected_output_values = values[num_entries: ]
            self.examples.append(TrainingExample(attribute_values, expected_output_values))

    def print(self):
        i = 1
        print('\ndata set:')
        for example in self.examples:
            print('{}: attributes = {} -> output = {}'.format(i, example.x, example.y))
            i += 1


class Layer:
    def __init__(self, num_neurons):
        self.size = num_neurons
        self.neurons = numpy.zeros(num_neurons).tolist()
        self.weight_matrix = None
        self.gradient_matrix = None

    def sum_square_weights(self):
        return self.weight_matrix.sum_square_weights_without_bias()

    def init_gradient_matrix(self):
        self.gradient_matrix = mt.Matrix()
        self.gradient_matrix.set(
            self.weight_matrix.num_rows,
            self.weight_matrix.num_cols)


class Network:
    def __init__(self, network_filename, initial_weights_filename):
        
        # read network_filename
        dataframe = pd.read_csv(network_filename, sep=" ", header=None)
        values = dataframe[0].tolist()
        self.regularizationFactor = values[0]
        self.bias_added = False
        
        n = len(values)
        self.total_layers = n-1
        self.num_entries = int(values[1]) # layer 0 
        self.num_output = int(values[-1]) # layer n-1
        self.layers = []
        for value in values[1: ]:
            num_neurons = int(value)
            self.layers.append(Layer(num_neurons))

        # read initial_weights_filename    
        df = pd.read_csv(initial_weights_filename, sep=";|,", header=None, engine='python')
        (numRows, numCols) = df.shape
        
        for l in range(0, numRows):
            values = [x for x in df.iloc[l].tolist() if str(x) != 'nan']
            layer = self.layers[l]
            num_neurons = layer.size + 1 # +1 for bias
            bias_index = 0
            rows = []
            
            while bias_index < len(values):
                coefs = values[bias_index: bias_index + num_neurons]
                rows.append(coefs)
                bias_index += num_neurons

            layer.weight_matrix = mt.Matrix(rows)
            layer.init_gradient_matrix()


    def propagate_instance(self, x):
        self.layers[0].neurons = x  # set a1 = x
        if not self.bias_added:
            self.layers[0].neurons.insert(0,1) # add bias
        print('a_l=1 = {}'.format(self.layers[0].neurons))

        for l in range(1, self.total_layers):
            prev_layer = self.layers[l-1]
            theta = prev_layer.weight_matrix
            a = prev_layer.neurons
            if l > 1:
                a.insert(0,1) # add bias
            z = theta.multiply_by_vector(a)
            layer = self.layers[l]
            layer.neurons = list(map(gaussian, z))
            #print('z = {}'.format(z))
            print('a_l={} = {}'.format(l+1, self.layers[l].neurons))            

        self.bias_added = True

        return self.layers[-1].neurons


    def cost(self, xi, yi, f_xi):
        _yi = list(map(lambda y: -y, yi))
        log = list(map(lambda f: math.log(f), f_xi))
        Ji = dot_product(yi, log)

        n_yi  = list(map(lambda y: 1-y, yi))
        n_log = list(map(lambda f: math.log(1-f), f_xi))
        Ji -= dot_product(n_yi, n_log)
        return Ji

    def sum_square_weights(self):
        total = 0
        for l in range(0, self.total_layers-1):
            total += self.layers[l].sum_square_weights()
        return total
        

    def cost_regularizaded(self, data_set):
        n = len(data_set.examples)
        J = 0
        
        for example in data_set.examples:
            xi = example.x
            yi = example.y
            f_xi = self.propagate_instance(xi)
            Ji = self.cost(xi, yi, f_xi)
            J += numpy.sum(Ji)

        J /= n
        S = self.sum_square_weights()
        S *= self.regularizationFactor/(2*n)
        return J+S

    def regularize_cost(self, J, numExamples):
        S = self.sum_square_weights()
        S *= self.regularizationFactor/(2*numExamples)
        return J+S

    def __str__(self):
        output = ''
        
        for l in range(0, self.total_layers-1):
            layer = self.layers[l]
            output += ('\nl={}: {} neurons\n'.format(l+1, layer.size))
            output += ('a = {}\n'.format(layer.neurons))
            output += ('theta = \n{}\n'.format(layer.weight_matrix.matrix))

        l = self.total_layers-1
        layer = self.layers[l]
        output += ('\nl={}: {} neurons - output\n'.format(l+1, layer.size))
        output += ('a = {}\n'.format(layer.neurons))
        
        return output
    
    def print_all(self):
        print('num entries = {}'.format(self.num_entries))
        print('num output = {}'.format(self.num_output))
        print('total layers = {}'.format(self.total_layers))
        print('layers: \n{}'.format(self))

    
        

    