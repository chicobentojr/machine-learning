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

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
np.set_printoptions(precision=5)

def regularized_cost(network, data_set, numExamples, log_details=False):

    if log_details:
        logger.info('--------------------------------------------\nCalculando erro/custo J da rede');

    J = 0.0
    
    for i in range(0, numExamples):
        example = data_set.examples[i]
        xi = example.x
        yi = example.y

        if log_details:
            logger.info('\n\tProcessando exemplo #{}'.format(i+1))
            logger.info('\n\tPropagando entrada {}'.format(xi))
        
        (f_xi, Ji) = network.propagate_instance_and_get_cost(xi, yi, log_details)
        J += Ji

        if log_details:
            logger.info('\tSaida predita  para o exemplo {}: {}'.format(i+1, net.format_list(f_xi)))
            logger.info('\tSaida esperada para o exemplo {}: {}'.format(i+1, net.format_list(yi)))
            logger.info('\tJ do exemplo {}: {:.5f}'.format(i+1, Ji))

    J /= numExamples
    J = network.regularize_cost(J, numExamples)

    if log_details:
        logger.info('\n J total do dataset (com regularizacao): {:.5f}'.format(J))
        
    return J

       
def backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    max_iterations, alpha):

    network = net.Network(network_filename, initial_weights_filename, logger)
    network.print()
    
    data_set = net.DataSet(data_set_filename, network.num_entries)
    data_set.print(logger)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers
    delta = [None] * numLayers

    last_regularized_cost = regularized_cost(network, data_set, numExamples, True)
    less_acceptable_difference = 0.0001 # how much?????
    stop = False
    iteration = 0

    logger.info('\n\n--------------------------------------------\nRodando backpropagation')

    while not stop and iteration < max_iterations:

        logger.info('\n***********************************************************')
        logger.info(' ITERACAO #{}'.format(iteration+1))
        logger.info('\n 1. Percorrer exemplos (x,y):')
        
        for i in range(0, numExamples):
            logger.info('\n Calculando gradientes com base no exemplo #{}'.format(i+1))
            logger.info('\n 1.1. Propagacao pela rede')

            example = data_set.examples[i]
            xi = example.x
            yi = example.y
            f_xi = network.propagate_instance(xi)

            delta[-1] = np.subtract(f_xi, yi)
            logger.info('\n 1.2. Delta da camada de saida\n\tdelta{} = {}'.format(numLayers, delta[-1]))
            logger.info('\n 1.3. Deltas das camadas ocultas')
            
            for k in range(numLayers-2, 1-1, -1):
                layer_k = network.layers[k]
                theta_k = layer_k.weight_matrix
                a_k = layer_k.neurons
                num_neurons = layer_k.size
                
                product = theta_k.transpose_and_multiply_by_vector(delta[k+1])
                delta[k] = np.zeros(num_neurons) # with no bias

                for j in range (0, num_neurons):
                    delta[k][j] = product[j+1] * a_k[j+1] * (1 - a_k[j+1])

                logger.info('\tdelta{} = {}'.format(k+1, delta[k]))


            logger.info('\n 1.4. Gradientes dos pesos de cada camada com base nos exemplo atual')

            for k in range(numLayers-2, 0-1, -1):
                layer_k = network.layers[k]
                a_k = layer_k.neurons
                gradient = np.matmul(
                    np.matrix(delta[k+1]).transpose(),
                    np.matrix(a_k))
                network.layers[k].gradient_matrix.matrix += gradient
                logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, mt.str_tabs(gradient,2)))
                
        # end examples
        logger.info('\n Dataset completo processado.')

        logger.info('\n\n 2. Gradientes finais (regularizados) para os pesos de cada camada')

        for k in range(0, numLayers-1):
            layer_k = network.layers[k]
            theta_k = layer_k.weight_matrix
            P_k = theta_k.copy()
            P_k.regularize(network.regularizationFactor)
            D_k = layer_k.gradient_matrix
            network.layers[k].gradient_matrix.matrix =  (D_k.matrix + P_k.matrix) / numExamples
            logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, network.layers[k].gradient_matrix.str_tabs(2)))

        #logger.info('\n 3. Atualizar pesos de cada camada com base nos gradientes')

        for k in range(0, numLayers-1):
            layer_k = network.layers[k]
            network.layers[k].weight_matrix.matrix -= (layer_k.gradient_matrix.matrix * alpha)
            #logger.info('\n\tTheta{} = \n{}'.format(k+1, network.layers[k].weight_matrix.str_tabs(2)))
        
        #logger.info('\nCheck peformance to set stop:')

        J = regularized_cost(network, data_set, numExamples)
        diff = last_regularized_cost - J
        stop = (diff <= less_acceptable_difference)
        #logger.info('diff = last_cost({}) - actual_cost({}) = {}'.format(last_regularized_cost, J, diff))
        last_regularized_cost = J
        iteration += 1

    if max_iterations > 1 :
        logger.info('\nFim apos {} iteracoes.\n'.format(iteration))
        
    return (network, data_set)

