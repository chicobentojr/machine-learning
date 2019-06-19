import os
import logging
import math
import random
import time
import click
import click_log
import numpy as np
import network_classes as net
import matrix as mt
import backpropagation as bp

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
np.set_printoptions(precision=5, suppress=True)


def numeric_gradient(network_filename, initial_weights_filename, data_set_filename, epsilon):
    
    network = net.Network(network_filename, initial_weights_filename, logger)    
    data_set = net.DataSet(data_set_filename, network.num_entries)
 
    numExamples = len(data_set.examples)
    numLayers = network.total_layers

    numericGradientMatrices = [None] * numLayers

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        theta_k = layer_k.weight_matrix
        theta_k_m = theta_k.matrix
        numericGradientMatrices[k] = np.zeros(shape=(theta_k.num_rows, theta_k.num_cols))

        for r in range(0, theta_k.num_rows):
            for c in range(0, theta_k.num_cols):
                original_value = theta_k_m[r,c]
                
                Jplus = 0.0
                Jminus = 0.0

                theta_k_m[r,c] += epsilon
                Jplus = bp.regularized_cost(network, data_set, numExamples)

                theta_k_m[r,c] = original_value - epsilon
                Jminus = bp.regularized_cost(network, data_set, numExamples)

                theta_k_m[r,c] = original_value

                numericGradientMatrices[k][r,c] = (Jplus - Jminus)/(2*epsilon)
        #end for r,c
                    
        logger.info('\n\tGradiente numerico de Theta{} = \n{}'.format(k+1, mt.str_tabs(numericGradientMatrices[k],2)))  
    # end for layers

    return numericGradientMatrices


@click.group()
def main():
    pass

@main.command(name='gradient_verification')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.1, help='Weights Update Rate, is used to smooth the gradient')
@click.option('--beta', '-b', default=0.9, help='Relevance of recent average direction (Method of Moment)')
@click.option('--epsilon', '-e', default=0.00001, help='Epsilon for gradient numeric verification')

def gradient_verification(network_filename, initial_weights_filename, data_set_filename,
                         alpha, beta, epsilon):

    bp.logger = logger
    backPropMaxIterations = 1
    (network, training_result) = bp.backpropagation(network_filename, initial_weights_filename, data_set_filename,
                                             backPropMaxIterations, alpha)

    logger.info('\n\n------------------------------------------------------')
    logger.info(' Rodando verificacao numerica de gradientes (epsilon={})'.format(epsilon))

    numericGradientMatrices = numeric_gradient(network_filename, initial_weights_filename, data_set_filename, epsilon)

    logger.info('\n\n------------------------------------------------------')
    
    numLayers = network.total_layers

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        Theta_k = layer_k.weight_matrix
        Theta_k_m = Theta_k.matrix
        
        numericGradientMatrix = numericGradientMatrices[k]

        total_error = 0.0

        for r in range(0, Theta_k.num_rows):
            for c in range(0, Theta_k.num_cols):
                theta_ji_back = Theta_k_m[r,c]
                theta_ji_num = numericGradientMatrix[r,c]
                diff = theta_ji_back - theta_ji_num
                total_error += diff * diff
        #end for r,c

        total_error /= (Theta_k.num_rows * Theta_k.num_cols)

        logger.info('\tErro entre gradiente via backprop e gradiente numerico para Theta{} = {}'.format(k+1, total_error))  
    # end

if __name__ == "__main__":
    main()

