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

    numericGradientMatrices = [None] * numLayers  #output

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        theta_k = layer_k.weight_matrix
        theta_k_m = theta_k.matrix
		rows = theta_k.num_rows
		cols = theta_k.num_cols
        numericGradientMatrices[k] = np.matrix(np.zeros(shape=(rows,cols)))

        for j in range(0, rows):
            for i in range(0, cols):
                original_value = theta_k_m[j,i] #save to recover

                theta_k_m[j,i] += epsilon
                Jplus = bp.regularized_cost(network, data_set, numExamples)

                theta_k_m[j,i] = original_value - epsilon
                Jminus = bp.regularized_cost(network, data_set, numExamples)

                theta_k_m[j,i] = original_value #recover

                numericGradientMatrices[k][j,i] = (Jplus - Jminus)/(2*epsilon)
        #end for j,i
                    
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
@click.option('--epsilon', '-e', default=0.000001, help='Epsilon for gradient numeric verification')

def gradient_verification(network_filename, initial_weights_filename, data_set_filename,
                         alpha, beta, epsilon):

    #bp.logger = logger
    (network, training_result) = bp.backpropagation(
			network_filename, initial_weights_filename, data_set_filename,
			max_iterations=1, alpha=alpha, beta=beta, logger=logger)

    logger.info('\n\n------------------------------------------------------')
    logger.info(' Rodando verificacao numerica de gradientes (epsilon={})'.format(epsilon))

    numericGradientMatrices = numeric_gradient(network_filename, initial_weights_filename, data_set_filename, epsilon)

    logger.info('\n\n------------------------------------------------------')
    logger.info('\tErro (medio quadratico) entre gradiente via backprop e gradiente numerico para:')
    
    numLayers = network.total_layers

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        D_k = layer_k.gradient_matrix
        D_k_m = D_k.matrix
        Diff = mt.numpy_to_Matrix(D_k_m - numericGradientMatrices[k]) #class Matrix
		numElems = D_k.num_rows * D_k.num_cols
        total_error = Diff.sum_square_elements()/numElems
        logger.info('\t\tTheta{} = {:.6f} ({})'.format(k+1, total_error, total_error))  

    # end

if __name__ == "__main__":
    main()

