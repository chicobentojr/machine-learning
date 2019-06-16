import os
import logging
import math
import random
import time
import pandas as pd
import click
import click_log
import numpy
import matrix as mt

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
numpy.set_printoptions(precision=5)

#-------------------------------------------------------
# Auxiliar functions

def gaussian(x):
    return 1/(1 + math.exp(-x))

def product_element_by_element(array1, array2):
    result = []
    for i in range (0, len(array1)):
        result.append(array1[i] * array2[i])
    return result

def format_list(list_array):
    return '{}'.format(numpy.array(list_array))

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
            print('#{}: x = {} => y = {}'.format(i, format_list(example.x), format_list(example.y)))
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
        self.layers[0].neurons = x[:]  # set a1 = x

        for l in range(1, self.total_layers):
            prev_layer = self.layers[l-1]
            theta = prev_layer.weight_matrix
            a = prev_layer.neurons
            a.insert(0,1) # add bias
            z = theta.multiply_by_vector(a)
            layer = self.layers[l]
            layer.neurons = list(map(gaussian, z))
            
            logger.info('\ta{} = {}\n'.format(l, format_list(self.layers[l-1].neurons)))
            logger.info('\tz{} = {}'.format(l+1, format_list(z)))

        l = self.total_layers-1
        logger.info('\ta{} = {}\n'.format(l+1, format_list(self.layers[l].neurons)))

        return self.layers[-1].neurons


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
    
    def print_all(self):
        print('num entries = {}'.format(self.num_entries))
        print('num output = {}'.format(self.num_output))
        print('total layers = {}'.format(self.total_layers))
        print('layers: \n{}'.format(self))

#--------------------- Classes End ----------------------------------


def cost(yi, f_xi):
    '''
    log = list(map(lambda f: -math.log10(f), f_xi))
    Ji = numpy.dot(yi, log)

    logger.info('dot1 = yi={} .* -log={} = {}'.format(yi, log, Ji))

    n_yi  = list(map(lambda y: 1-y, yi))
    n_log = list(map(lambda f: math.log10(1-f), f_xi))
    dot2 = numpy.dot(n_yi, n_log)

    logger.info('dot2 = 1-yi={} .* log={} = {}'.format(n_yi, n_log, dot2))
    
    return Ji - dot2
    '''
    J = 0.0
    for k in range(0,len(yi)):
        J += -yi[k] * math.log(f_xi[k]) - (1-yi[k]) * math.log(1-f_xi[k])
    return J

           
@click.group()
def main():
    pass


@main.command(name='backpropagation')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.5, help='Weights Update Rate, is used to smooth the gradient')
#@click.option('--epsilon', '-e', default=0.000001, help='Epsilon for gradient numeric verification')

def backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    alpha):
    
    network = Network(network_filename, initial_weights_filename)
    #network.print_all()
    
    data_set = DataSet(data_set_filename, network.num_entries)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers
    delta = [None] * numLayers

    less_acceptable_difference = 0.1 # how much?????
    last_regularized_cost = 1000000

    stop = False
    count = 0

    while not stop and count < 1:
        J = 0 # to define stop
        #logger.info('\nITERATION #{} **********************'.format(count+1))

        logger.info('\n 1. Percorrer exemplos (x,y):')
        
        for i in range(0, numExamples):
            logger.info('\n Exemplo #{} -------------------------------------------'.format(i+1))
            logger.info('\n 1.1. Propagacao pela rede:')

            example = data_set.examples[i]
            xi = example.x
            yi = example.y
            f_xi = network.propagate_instance(xi)
            Ji = cost(yi, f_xi)
            #Ji = network.regularize_cost(Ji, numExamples)
            J += numpy.sum(Ji)

            logger.info('\tSaida predita  para o exemplo {}: {}'.format(i+1, format_list(f_xi)))
            logger.info('\tSaida esperada para o exemplo {}: {}'.format(i+1, format_list(yi)))
            logger.info('\tJ do exemplo {}: {:.5f}'.format(i+1, Ji))

            delta[-1] = numpy.subtract(f_xi, yi)
            logger.info('\n 1.2. Delta da camada de saida\n\tdelta{} = {}'.format(numLayers, delta[-1]))
            logger.info('\n 1.3. Deltas das camadas ocultas')
            
            for k in range(numLayers-2, 1-1, -1):
                layer_k = network.layers[k]
                theta_k = layer_k.weight_matrix
                a_k = layer_k.neurons
                num_neurons = layer_k.size
                
                product = theta_k.transpose_and_multiply_by_vector(delta[k+1])
                delta[k] = numpy.zeros(num_neurons) # with no bias

                for j in range (0, num_neurons):
                    delta[k][j] = product[j+1] * a_k[j+1] * (1 - a_k[j+1])

                logger.info('\tdelta{} = {}'.format(k+1, delta[k]))


            logger.info('\n 1.4. Gradientes dos pesos de cada camada com base nos exemplo atual')

            for k in range(numLayers-2, 0-1, -1):
                layer_k = network.layers[k]
                a_k = layer_k.neurons
                gradient = numpy.matmul(
                    numpy.matrix(delta[k+1]).transpose(),
                    numpy.matrix(a_k))
                network.layers[k].gradient_matrix.matrix += gradient
                logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, mt.str_tabs(gradient,2)))

        # end for

        J /= numExamples
        J = network.regularize_cost(J, numExamples)
        logger.info('\n J total do dataset (com regularizacao): {:.5f}'.format(J))
        '''
        logger.info('\nCheck peformance to set stop:')
        diff = last_regularized_cost - J
        stop = (diff <= less_acceptable_difference)
        logger.info('diff = last_cost({}) - actual_cost({}) = {}'.format(last_regularized_cost, J, diff))
        last_regularized_cost = J
        '''
        count += 1

        logger.info('\n\n 2. Gradientes finais (regularizados) para os pesos de cada camada')

        for k in range(0, numLayers-1):
            layer_k = network.layers[k]
            theta_k = layer_k.weight_matrix
            P_k = theta_k.matrix * network.regularizationFactor
            D_k = layer_k.gradient_matrix.matrix
            network.layers[k].gradient_matrix.matrix =  (D_k + P_k) / numExamples
            logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, network.layers[k].gradient_matrix.str_tabs(2)))

        logger.info('\n 3. Atualizar pesos de cada camada com base nos gradientes')

        for k in range(0, numLayers-1):
            layer_k = network.layers[k]
            network.layers[k].weight_matrix.matrix -= (layer_k.gradient_matrix.matrix * alpha)
            logger.info('\n\tTheta{} = \n{}'.format(k+1, network.layers[k].weight_matrix.str_tabs(2)))
        
        logger.info('\n')



if __name__ == "__main__":
    main()
