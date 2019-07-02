import os
import math
import copy
import sys
import random


def get_euclidean_distance(a, b):
    dist_sum = 0.0

    for dimension in range(len(a)):
        dist_sum += math.pow(float(a[dimension] - b[dimension]), 2)

    return math.sqrt(dist_sum)


def get_clusters_coord(points, k, initial_coord):

    original_coord = [[0, 0]] * k
    clusters = initial_coord

    while original_coord != clusters:
        # print('init ')
        # print(original_coord, clusters)
        original_coord = copy.deepcopy(clusters)
        distances = []

        for p in points:
            p_dists = []
            for cluster in clusters:
                dist_cluster_p = get_euclidean_distance(p, cluster)

                p_dists.append(dist_cluster_p)
                # print('dist', p, cluster, '{:.2f}'.format(
                # dist_cluster_p), end='\t\t')

            distances.append(p_dists)
            # print()

        # print('\ndistances')
        # print(distances)

        cluster_point_amount = [0] * len(clusters)
        cluster_points_sum = [[0, 0]] * len(clusters)

        # print('\nclssum', cluster_points_sum)

        for index, point in enumerate(points):
            closer_cluster_index = distances[index].index(
                min(distances[index]))
            # print('closer', closer_cluster_index + 1, 'point', point)

            old_sum = cluster_points_sum[closer_cluster_index]

            cluster_points_sum[closer_cluster_index] = [
                old_sum[0] + point[0], old_sum[1] + point[1]]

            cluster_point_amount[closer_cluster_index] += 1

            # print('clssum', cluster_points_sum)
            # print()

        for index, cluster_sum in enumerate(cluster_points_sum):
            for dimension in range(len(cluster_sum)):
                if cluster_point_amount[index] > 0:
                    clusters[index][dimension] = cluster_sum[dimension] / \
                        cluster_point_amount[index]

        # print()
        # print(cluster_points_sum)
        # print()
        # print(clusters)
        # print('check')
        # print(original_coord, '===', clusters)
    return clusters


if __name__ == "__main__":
    points = []

    filename = 'input/validation-class.txt'
    k = 2
    seed = None

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        k = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])

    with open(filename) as file:
        lines = file.readlines()

        for line in lines:
            values = line.strip().split(',')
            points.append([float(v) for v in values])

    # print(points)

    if seed:
        random.seed(seed)

    initial_coord = []

    for i in range(k):
        initial_coord.append(points[random.randrange(len(points))])

    # print(initial_coord)
    # print('\nclusters\n')
    # initial_coord = [
    #     [0, 0.29],
    #     [1, 0.71],
    # ]
    # initial_coord = [
    #     [1.0, 1.0],
    #     [5.0, 7.0],
    # ]

    clusters = get_clusters_coord(points, 2, initial_coord)
    # print()
    for c in clusters:
        print('{:.2f}, {:.2f}'.format(c[0], c[1]))
    # print(clusters)
