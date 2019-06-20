import os
import logging
import math
import random
import time
import pandas as pd
import click
import click_log
import numpy as np
import matrix as mt

logger = None
np.set_printoptions(precision=5)

#-------------------------------------------------------
# Auxiliar functions

def sigmoid(z):
    if z < 0:
        return 1 - 1/(1 + math.exp(z))
    return  1/(1 + math.exp(-z))

def format_list(array_list):
    return '{}'.format(np.array(array_list))

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

    def print(self, logger):
        i = 1
        logger.info('\nData set:')
        for example in self.examples:
            logger.info('#{}: x = {} => y = {}'.format(i, format_list(example.x), format_list(example.y)))
            i += 1


class Layer:
    def __init__(self, num_neurons):
        self.size = num_neurons
        self.neurons = np.zeros(num_neurons).tolist()
        self.weight_matrix = None
        self.gradient_matrix = None
        self.gradient_mean_matrix = None

    def sum_square_weights(self):
        return self.weight_matrix.sum_square_weights_without_bias()

    def init_gradient_matrix(self):
        self.gradient_matrix = mt.Matrix()
        self.gradient_matrix.set(
            self.weight_matrix.num_rows,
            self.weight_matrix.num_cols)

        self.gradient_mean_matrix = mt.Matrix()
        self.gradient_mean_matrix.set(
            self.weight_matrix.num_rows,
            self.weight_matrix.num_cols)


class Network:
    def __init__(self, network_filename, initial_weights_filename, logger_):

        global logger
        logger = logger_
        
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
        self.read_as_txt(initial_weights_filename)
    
        
    def read_as_txt(self, initial_weights_filename):
        fp = open(initial_weights_filename, 'r')
        line = fp.readline().replace(' ','')
        l=0
        
        while(line):
            rows = []
            for str_weights in line.split(';'):
                weights = [float(x) for x in str_weights.split(',')]
                rows.append(weights)
            
            layer = self.layers[l]
            layer.weight_matrix = mt.Matrix(rows)
            layer.init_gradient_matrix()
            
            l += 1
            line = fp.readline().replace(' ','')

        fp.close()
        
        
    # para o exemplo 2 desconsidera os 3 primeiros valores da linha (pq?)
    def read_with_pandas(initial_weights_filename):
        df = pd.read_csv(initial_weights_filename, sep=";|,", header=None, engine='python')
        (numRows, numCols) = df.shape
        
        for l in range(0, numRows):
            values = [x for x in df.iloc[l].tolist() if str(x) != 'nan']
            print('\nrow: {}'.format(df.iloc[l].tolist()))
            print('values = {}'.format(values))
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


    def propagate_instance(self, x, log_details=False):
        self.layers[0].neurons = x[:]  # set a1 = x

        for l in range(1, self.total_layers):
            prev_layer = self.layers[l-1]
            theta = prev_layer.weight_matrix
            a = prev_layer.neurons
            a.insert(0,1) # add bias
            z = theta.multiply_by_vector(a)
            layer = self.layers[l]
            layer.neurons = list(map(sigmoid, z))

            if log_details:
                logger.info('\t\ta{} = {}\n'.format(l, format_list(self.layers[l-1].neurons)))
                logger.info('\t\tz{} = {}'.format(l+1, format_list(z)))

        l = self.total_layers-1
        if log_details:
            logger.info('\t\ta{} = {}\n'.format(l+1, format_list(self.layers[l].neurons)))

        return self.layers[-1].neurons

    def propagate_instance_and_get_cost(self, xi, yi, log_details=False):
        f_xi = self.propagate_instance(xi, log_details)
        fi = f_xi.copy()
        
        func = np.vectorize(lambda f: -math.log(f))
        log_fi = func(fi)

        func = np.vectorize(lambda y: 1-y)
        _1_yi = func(yi.copy())

        func = np.vectorize(lambda f: math.log(1-f))
        log_1_fi = func(fi)

        arr = (yi * log_fi) - (_1_yi * log_1_fi)
        Ji = np.sum(arr)
        return (f_xi, Ji)


    def regularize_cost(self, J, numExamples):
        S = 0.0
        for l in range(0, self.total_layers-1):
            S += self.layers[l].sum_square_weights()
        
        S *= self.regularizationFactor/(2*numExamples)
        return J+S
       

    def __str__(self):
        output = ''
        
        for l in range(0, self.total_layers-1):
            layer = self.layers[l]
            output += ('\n l={}: {} neurons\n'.format(l+1, layer.size))
            output += ('\ta{} = {}\n'.format(l+1, layer.neurons))
            output += ('\tTheta{} = \n{}\n'.format(l+1, layer.weight_matrix.str_tabs(2)))

        l = self.total_layers-1
        layer = self.layers[l]
        output += ('\n l={}: {} neurons - output\n'.format(l+1, layer.size))
        output += ('\ta{} = {}\n'.format(l+1, layer.neurons))
        
        return output

    
    def print(self):
        logger.info('num entries = {}'.format(self.num_entries))
        logger.info('num output = {}'.format(self.num_output))
        logger.info('total layers = {}'.format(self.total_layers))
        logger.info('layers: \n{}'.format(self))

    def print_Thetas(self):
        for k in range(0,self.total_layers-1):
            layer_k = self.layers[k]
            theta_k = layer_k.weight_matrix
            print('\tTheta{} =\n {}'.format(k+1, theta_k.str_tabs(2)))
            
    
