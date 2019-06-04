import os
import logging
import math
import random
import time
import click
import click_log
import network_classes as ntc
import numpy
import matrix

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

           
@click.group()
def main():
    pass


@main.command(name='backpropagation')
@click_log.simple_verbosity_option(logger)
@click.argument('network_filename')
@click.argument('initial_weights_filename')
@click.argument('data_set_filename')
@click.option('--alpha', '-a', default=0.5, help='Weights Update Rate, is used to smooth the gradient')

def backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    alpha):
    
    network = ntc.Network(network_filename, initial_weights_filename)
    network.print_all()
    
    data_set = ntc.DataSet(data_set_filename, network.num_entries)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers
    delta = [None] * numLayers

    less_acceptable_difference = 0.1 # how much?????
    last_regularized_cost = 1000000

    stop = False
    count = 0

    while not stop and count < 100:
        J = 0 # to define stop
        logger.info('\ITERATION #{}'.format(count+1))
        
        for i in range(0, numExamples):
            logger.info('\nInstance #{}:'.format(i+1))
            logger.info('\n1. Network Propagation:')
            
            example = data_set.examples[i]
            f_xi = network.propagate_instance(example.x)
            delta[-1] = numpy.subtract(f_xi, example.y)

            Ji = network.cost(example.x, example.y, f_xi)
            J += numpy.sum(Ji)
            
            #logger.info('\n2. Delta for output layer: \ndelta_(l=L={}) = {}'.format(numLayers, delta[-1]))
            #logger.info('\n3. Deltas for the hidden layers:')
            
            for k in range(numLayers-2, 1-1, -1):
                layer_k = network.layers[k]
                theta_k = layer_k.weight_matrix
                a_k = layer_k.neurons
                _1_a_k = list(map(lambda x: 1-x, a_k))
                product = theta_k.transpose_and_multiply_by_vector(delta[k+1])
                delta[k] = numpy.dot(numpy.dot(product, a_k), _1_a_k)
                delta[k] = delta[k][1:len(delta[k])] # remove bias
                #logger.info('delta_(l={}) = {}'.format(k+1, delta[k]))

            #logger.info('\n4. Gradients of weights of each layer based on the current example:')

            for k in range(numLayers-2, 0-1, -1):
                layer_k = network.layers[k]
                a_k = layer_k.neurons
                network.layers[k].gradient_matrix.matrix += numpy.matmul(
                    numpy.matrix(delta[k+1]).transpose(),
                    numpy.matrix(a_k))
                
                #logger.info('\ngradient_(l={}) = \n{}'.format(k+1, layer_k.gradient_matrix.matrix))


        logger.info('\nCheck peformance to set stop:')
        #logger.info('Network state:\n{}'.format(network))

        J /= numExamples
        J = network.regularize_cost(J, numExamples)
        diff = last_regularized_cost - J
        stop = (diff <= less_acceptable_difference)
        logger.info('diff = last_cost({}) - actual_cost({}) = {}'.format(last_regularized_cost, J, diff))
        if stop:
            break
        
        last_regularized_cost = J
        count += 1

        #logger.info('\nFinal (regularized) gradients for the weights of each layer')

        for k in range(numLayers-2, 0-1, -1):
            layer_k = network.layers[k]
            theta_k = layer_k.weight_matrix
            P_k = theta_k.matrix * network.regularizationFactor
            D_k = layer_k.gradient_matrix.matrix
            network.layers[k].gradient_matrix.matrix =  (D_k + P_k) / numExamples
            #logger.info('\ngradient_(l={}): \n{}'.format(k+1, layer_k.gradient_matrix.matrix))

        #logger.info('\nUpdate weights of each layer based on gradients:')

        for k in range(numLayers-2, 0-1, -1):
            layer_k = network.layers[k]
            network.layers[k].weight_matrix.matrix -= (layer_k.gradient_matrix.matrix * alpha)
            #logger.info('\ntheta_(l={}) = \n{}'.format(k+1, theta_k))
        


if __name__ == "__main__":
    main()
