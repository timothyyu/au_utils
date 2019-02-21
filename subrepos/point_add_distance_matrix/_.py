import numpy as np
from scipy.spatial import distance

def add_points_to_distance_matrix(points, original_array, distance_matrix, metric='euclidean'):
    """
    There is an NxM array of points, a square matrix NxN with distances between points.
    This function adds new points to the distance matrix.
    We need to create a diagonal block holding distances between the new points
    and two identical blocks holding distances to original array.
    E.g.
    original_array = ([[1,2,3],[1,2,4],[1,2,5]])
    distance_matrix = distance.squareform(distance.pdist(original_array))
    >>> distance_matrix
    array([[0., 1., 2.],
           [1., 0., 1.],
           [2., 1., 0.]])
    >>> add_points_to_distance_matrix([[1,2,5],[1,2,5]], original_array,distance_matrix)
    array([[0., 1., 2.,  2., 2.],
           [1., 0., 1.,  1., 1.],
           [2., 1., 0.,  0., 0.],
           
           [2., 1., 0.,  0., 0.],
           [2., 1., 0.,  0., 0.]])
           
    """
    diagonal = distance.squareform(distance.pdist(points, metric=metric))
    twin_block = distance.cdist(original_array, points, metric=metric)
    return (np.vstack([np.hstack([distance_matrix,twin_block]), np.hstack([twin_block.T, diagonal])]))