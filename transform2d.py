# -*- coding: utf-8 -*-
"""transform2d

This module contains several funtions to perform 2D operations on vectors and matricies,
specifcally, but not limited to, common functions found in the study of robotics.

This module depends heavily on the numpy package.

"""

import numpy as np


def translate2d(vector, x, y):
    """Return 2D vector translated by given x and y values

    Args:
        vector (Iterable[int]): numpy ndarray of shape (2,)
        x (int): The x translation.
        y (int): The y translation.

    Returns:
        Iterable[Int]: numpy ndarray of the x- and y-translation applied to the vector.

    Raises:
        ValueError: If `vector` is not broadcastable into `(2,)`

    Examples:
        Translates x and y of origin vector:
        >>> translate2d(np.array([0, 0]), 1, 1)
        array([1, 1])

        Translates x and y arbitrary vector:
        >>> translate2d(np.array([1, 3]), 2, 2)
        array([3, 5])

        Boadcasts when `vector` argument is a scalar:
        >>> translate2d(1, 1, 1)
        array([2, 2])

        Boadcasts when translation is a boardast-able into vector shape:
        >>> translate2d(np.array([[1, 1],[1, 1]]), 1, 1)
        array([[2, 2],
               [2, 2]])

    """
    return np.array([x, y]) + vector


def rotate2d(vector, degrees):
    """Return 2D vector translated by given angle in degrees

    Args:
        vector (Iterable[int]): numpy ndarray of shape (2,)
        degrees (int): The angle of degrees to rotate

    Returns:
        Iterable[Int]: numpy ndarray of a rotation matrix applied to the vector.

    Raises:
        ValueError: If `vector` is not broadcastable into `(2,)`

    Examples:
        Rotating the by 0 degrees will return the original vector:
        >>> rotate2d(np.array([3, 8]), 0)
        array([3., 8.])

        Rotating the origin will return the origin:
        >>> rotate2d(np.array([0, 0]), 20)
        array([0., 0.])

        Rotates arbitrary vector:
        >>> rotate2d(np.array([1, 0]), 20)
        array([0.93969262, 0.34202014])

        Boadcasts when `vector` argument is a scalar:
        >>> rotate2d(1, 20)
        array([[ 0.93969262, -0.34202014],
               [ 0.34202014,  0.93969262]])

        Boadcasts when vector is a boardast-able into (2,) shape:
        >>> rotate2d(np.array([[1, 1],[1, 1]]), 20)
        array([[0.59767248, 0.59767248],
               [1.28171276, 1.28171276]])

    """
    theta = np.radians(degrees)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return rotation_matrix.dot(vector)
