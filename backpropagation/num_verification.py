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
#import backpropagation as bp

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
np.set_printoptions(precision=6, suppress=True)


@click.group()
def main():
    pass


@main.command(name='numeric_verification')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.5, help='Weights Update Rate, is used to smooth the gradient')
@click.option('--epsilon', '-e', default=0.000001, help='Epsilon for gradient numeric verification')

def numeric_verification(network_filename, initial_weights_filename, data_set_filename,
                         alpha, epsilon):
    
    network = net.Network(network_filename, initial_weights_filename, logger)
    #network.print()
    
    data_set = net.DataSet(data_set_filename, network.num_entries)
    #data_set.print(logger)
 
    #(network, data_set) = backpropagation(network_filename, initial_weights_filename, data_set_filename, alpha)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers

    numericGradientMatrices = [None] * numLayers

    logger.info('\n\nRodando verificacao numerica de gradientes (epsilon={})'.format(epsilon))

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        theta_k = layer_k.weight_matrix
        theta_k_m = theta_k.matrix
        numericGradientMatrices[k] = np.zeros(shape=(theta_k.num_rows, theta_k.num_cols))# theta_k_m.copy()

        print('Theta{} original =\n {}'.format(k+1, theta_k.str_tabs(2)))

        for r in range(0, theta_k.num_rows):
            for c in range(0, theta_k.num_cols):
                original_value = theta_k_m[r,c]
                Jplus = 0.0
                Jminus = 0.0

                conditionToPrint = (r==0 and c==1)

                if conditionToPrint:
                    print('Theta{} original =\n {}'.format(k+1, theta_k.str_tabs(2)))
                    print('\ntheta{}_({},{}) = {:.6f}'.format(k+1,r+1,c, original_value))
                    
                for i in range(0,numExamples):
                    example = data_set.examples[i]
                    xi = example.x
                    yi = example.y
                    #print(' #{}'.format(i))
                    
                    theta_k_m[r,c] += epsilon
                    (f_xi, J1) = network.propagate_instance_get_f_J(xi, yi)
                    Jplus += J1
                    
                    if conditionToPrint:
                        print('\n\t#{} - theta{}_({},{}) + eps = {} => J+ = {}'.format(i+1, k+1,r+1,c, theta_k_m[r,c], J1))
                        print('\tTheta{} =\n {}'.format(k+1, mt.str_tabs(theta_k_m, 2)))

                    theta_k_m[r,c] = original_value - epsilon
                    
                    (f_xi, J2) = network.propagate_instance_get_f_J(xi, yi)
                    Jminus += J2
                    
                    if conditionToPrint:
                        print('\n\t#{} - theta{}_({},{}) - eps = {} => J- = {}'.format(i+1, k+1,r+1,c, theta_k_m[r,c], J2))
                        print('\tTheta{} =\n {}'.format(k+1, mt.str_tabs(theta_k_m, 2)))

                    theta_k_m[r,c] = original_value
                # end examples

                gradient = ((Jplus - Jminus)/numExamples)/(2*epsilon)
                numericGradientMatrices[k][r,c] = gradient
                
                if conditionToPrint:
                    print('\ntheta{}_({},{}): \n\tJplus (media dos J+) = {} \n\tJminus (media dos J-) = {}'.format(k+1,r+1,c, Jplus/numExamples, Jminus/numExamples))
                    print('\tgradient theta{}_({},{}) = {:.6f}'.format(k+1,r+1,c,gradient))
        #end for r,c
                    
        logger.info('\n\tGradiente numerico de Theta{} = \n{}'.format(k+1, mt.str_tabs(numericGradientMatrices[k],2)))  
    # end


if __name__ == "__main__":
    main()

