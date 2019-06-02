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

def gaussian(x):
    return 1/(1 + math.exp(-x))

class TrainingExample:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DataSet:
    def __init__(self, data_set_filename, num_entries):
        df = pd.read_csv(data_set_filename, sep=";|,", header=None)
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
        self.neurons = []
        for i in range(0, num_neurons):
            self.neurons.append(0)
        self.weight_matrix = mt.Matrix()

    def sum_square_weights(self):
        total = 0
        for row in self.weight_matrix.matrix:
            square = list(map(lambda x: x*x, row))
            total += numpy.sum(square) - square[0] # discount bias squared
        return total


class Network:
    def __init__(self, network_filename, initial_weights_filename):
        # read network_filename
        dataframe = pd.read_csv(network_filename, sep=" ", header=None)
        values = dataframe[0].tolist()
        self.regularizationFactor = values[0]
        
        n = len(values)
        self.total_layers = n-1
        self.num_entries = int(values[1]) # layer 0 
        self.num_output = int(values[-1]) # layer n-1
        self.layers = []
        for value in values[1: ]:
            num_neurons = int(value)
            self.layers.append(Layer(num_neurons))

        # read initial_weights_filename    
        df = pd.read_csv(initial_weights_filename, sep=";|,", header=None)
        (numRows, numCols) = df.shape
        
        for l in range(0, numRows):
            values = [x for x in df.iloc[l].tolist() if str(x) != 'nan']
            layer = self.layers[l]
            num_neurons = layer.size + 1 # +1 for bias
            bias_index = 0
            
            while bias_index < len(values):
                coefs = values[bias_index: bias_index + num_neurons]
                layer.weight_matrix.add_row(coefs)
                bias_index += num_neurons


    def print(self):
        print('num entries = %i' %self.num_entries)
        print('num output = %i' %self.num_output)
        print('total layers = %i' %self.total_layers)
        print('layers:')
        
        for l in range(0, self.total_layers-1):
            layer = self.layers[l]
            print('\nl={}: {} neuronios'.format(l+1, layer.size))
            layer.weight_matrix.print()

        l = self.total_layers-1
        layer = self.layers[l]
        print('\nl={}: {} neuronios - output\n'.format(l+1, layer.size))
        

    def propagate_instance(self, x):
        self.layers[0].neurons = x  # set a1 = x
        print('a_l=1 = {}'.format(self.layers[0].neurons))

        for l in range(1, self.total_layers):
            prev_layer = self.layers[l-1]
            theta = prev_layer.weight_matrix
            a = prev_layer.neurons.copy()
            a.insert(0,1) # bias
            z = theta.multiply_by_vector(a)
            layer = self.layers[l]
            layer.neurons = list(map(gaussian, z))
            print('a_l={} = {}'.format(l+1, self.layers[l].neurons))

        return self.layers[-1].neurons


    def cost(self, xi, yi, f_xi):
        _yi = list(map(lambda y: -y, yi))
        log = list(map(lambda f: math.log(f), f_xi))
        Ji = mt.dot_product(yi, log)

        n_yi  = list(map(lambda y: 1-y, yi))
        n_log = list(map(lambda f: math.log(1-f), f_xi))
        Ji -= mt.dot_product(n_yi, n_log)
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
    
