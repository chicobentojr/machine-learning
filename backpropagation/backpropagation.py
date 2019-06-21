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
import pandas as pd
import cross_validation_network as cvn

ALPHA = 0.1
BETA = 0.9

logger = logging.getLogger(__name__)
click_log.basic_config(logger)
np.set_printoptions(precision=5)


def regularized_cost(network, data_set, numExamples, log_details=False):

    if log_details:
        logger.info('--------------------------------------------\nCalculando erro/custo J da rede')

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
            logger.info('\tSaida esperada para o exemplo {}: {}'.format( i+1, net.format_list(yi)))
            logger.info('\tJ do exemplo {}: {:.5f}'.format(i+1, Ji))

    J /= numExamples
    J = network.regularize_cost(J, numExamples)

    if log_details:
        logger.info('\n J total do dataset (com regularizacao): {:.5f}'.format(J))

    return J


def backprop_iteration(network, data_set, numExamples, alpha=0.1, beta=0.9, momentum=True):
    logger.info('\n 1. Percorrer exemplos (x,y):')

    numLayers = network.total_layers
    delta = [None] * numLayers

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
            delta[k] = np.zeros(num_neurons)  # with no bias

            for j in range(0, num_neurons):
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
            logger.info('\n\tGradiente de Theta{} = \n{}'.format( k+1, mt.str_tabs(gradient, 2)))

    # end examples
    logger.info('\n Dataset completo processado.')

    logger.info('\n\n 2. Gradientes finais (regularizados) para os pesos de cada camada')

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        theta_k = layer_k.weight_matrix
        P_k = theta_k.copy()
        P_k.regularize(network.regularizationFactor)
        D_k = layer_k.gradient_matrix
        network.layers[k].gradient_matrix.matrix = (D_k.matrix + P_k.matrix) / numExamples
        logger.info('\n\tGradiente de Theta{} = \n{}'.format(k+1, network.layers[k].gradient_matrix.str_tabs(2)))

    #logger.info( '\n 4. Atualizar pesos de cada camada com base nos gradientes')

    for k in range(0, numLayers-1):
        layer_k = network.layers[k]
        D_k = layer_k.gradient_matrix.matrix

        if momentum:
            z_k = layer_k.gradient_mean_matrix
            z_k.matrix = (z_k.matrix * beta) + D_k
            layer_k.weight_matrix.matrix -= (z_k.matrix * alpha)
        else:
            layer_k.weight_matrix.matrix -= (D_k * alpha)
        #logger.info('\n\tTheta{} = \n{}'.format(k+1, network.layers[k].weight_matrix.str_tabs(2)))

    #end


def single_backpropagation(network_filename, initial_weights_filename, data_set_filename,
                           alpha=ALPHA, beta=BETA):
    
    network = net.Network(network_filename, initial_weights_filename, logger)
    data_set = net.DataSet(data_set_filename, network.num_entries)
    numExamples = len(data_set.examples)
    regularized_cost(network, data_set, numExamples, True)
    logger.info('\n\n--------------------------------------------\nRodando backpropagation')
    backprop_iteration(network, data_set, numExamples, alpha, beta)
    return network


def backpropagation(network_filename, initial_weights_filename, data_set_filename,
                    max_iterations=100, alpha=ALPHA, beta=BETA, less_acceptable_difference=0.0001,
                    momentum=True, validation_filename='', possible_labels=[], patience=50,
                    logger=logger):

    network = net.Network(network_filename, initial_weights_filename, logger)
    data_set = net.DataSet(data_set_filename, network.num_entries)
    validation_set = net.DataSet(validation_filename, network.num_entries)

    numExamples = len(data_set.examples)
    numLayers = network.total_layers

    last_regularized_cost = regularized_cost(network, data_set, numExamples, False)
    stop = False
    iteration = 0

    training_result = cvn.get_empty_result_dict(possible_labels)

    logger.info('\n\n--------------------------------------------\nRodando backpropagation')

    while not stop and iteration < max_iterations:

        logger.info('\n***********************************************************')
        logger.info(' ITERACAO #{}'.format(iteration+1))
        backprop_iteration(network, data_set, numExamples, alpha, beta)

        # Check peformance to set stop
        J = regularized_cost(network, data_set, numExamples)
        diff = last_regularized_cost - J

        if diff <= less_acceptable_difference:
            patience -= 1
            stop = (patience == 0)
        logger.info( 'diff = last_cost({}) - actual_cost({}) = {}'.format(last_regularized_cost, J, diff))

        last_regularized_cost = J

        # Getting training metrics
        training_result[cvn.LOSS].append(J)
        right_predictions = 0
        for i in range(numExamples):
            example = data_set.examples[i]
            xi = example.x
            yi = example.y
            f_xi = network.propagate_instance(xi)

            real_label = yi.index(1)
            predict_label = f_xi.index(max(f_xi))

            if real_label == predict_label:
                right_predictions += 1

        accuracy = right_predictions / numExamples
        training_result[cvn.ACCURACY].append(accuracy)

        if validation_filename:
            # Getting validation metrics
            # test epoch arch loss acc val_acc val_loss
            # prec_macro, preci_micro, rec_macro, rec_micro,
            # labels_M_vp,labels_M_fp,labels_M_fn,labels_M_precision,labels_M_recall,labels_M_f_measure
            validation_examples = len(validation_set.examples)
            val_J = regularized_cost(network, validation_set, validation_examples)

            val_right_predictions = 0

            matrix_dict = {}
            for x_label in possible_labels:
                for y_label in possible_labels:
                    matrix_dict['{}-{}'.format(x_label, y_label)] = 0

            for i in range(validation_examples):
                example = validation_set.examples[i]
                xi = example.x
                yi = example.y
                f_xi = network.propagate_instance(xi)

                real_label = possible_labels[yi.index(1)]
                predict_label = possible_labels[f_xi.index(max(f_xi))]

                matrix_dict['{}-{}'.format(real_label, predict_label)] += 1

                if real_label == predict_label:
                    val_right_predictions += 1

            val_acc = val_right_predictions / validation_examples

            training_result[cvn.VAL_LOSS].append(val_J)
            training_result[cvn.VAL_ACCURACY].append(val_acc)

            matrix = cvn.get_confusion_matrix_from_dict(matrix_dict)
            metrics = cvn.get_evaluation_metrics(matrix, possible_labels)

            cvn.add_metrics_to_dict(training_result, metrics)
        else:
            for k in training_result.keys():
                if k not in [cvn.ACCURACY, cvn.LOSS]:
                    training_result[k].append(None)

        logger.info('loss = {}\tacc = {}'.format(J, accuracy))
        if validation_filename:
            logger.info('val_loss = {}\tval_acc = {}'.format(val_J, val_acc))

        iteration += 1

    # end while

    if max_iterations > 1:
        logger.debug('\nFim apos {} iteracoes.\n'.format(iteration))
        # training_result_df = pd.DataFrame(training_result)
        # print(training_result_df)

    return (network, training_result)
