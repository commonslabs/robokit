# -*- coding: utf-8 -*-
"""transform2d

This module contains several funtions to perform 2D operations on vectors and matricies,
specifcally, but not limited to, common functions found in the study of robotics.

This module depends heavily on the numpy package.

"""

import numpy as np


def homo_translate2d(x, y):
    """Returns homogenous translation matrix for given x and y

    Args:
        x (int): The x translation.
        y (int): The y translation.

    Returns:
        Iterable[Int]: numpy ndarray of the x- and y-translation.

    Examples:
        Returns a homogenous translation identity matrix for x and y:
        >>> homo_translate2d(0, 0)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

        Returns a homogenous translation matrix for x and y:
        >>> homo_translate2d(1, 1)
        array([[1, 0, 1],
               [0, 1, 1],
               [0, 0, 1]])

        Returns a homogenous translation matrix for x and y:
        >>> homo_translate2d(2, 2)
        array([[1, 0, 2],
               [0, 1, 2],
               [0, 0, 1]])

    """
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])


def rotate2d(degrees):
    """Return 2D rotation matrix based on the given angle

    Args:
        degrees (int): The angle of degrees to rotate

    Returns:
        Iterable[Int]: numpy ndarray of a rotation matrix

    Examples:
        Returns rotation identity matrix for x and y:
        >>> rotate2d(0)
        array([[ 1., -0.],
               [ 0.,  1.]])

        Returns a rotation matrix for the given angle in degrees:
        >>> rotate2d(20)
        array([[ 0.93969262, -0.34202014],
               [ 0.34202014,  0.93969262]])

        Returns a rotation matrix for the given angle in degrees:
        >>> rotate2d(45)
        array([[ 0.70710678, -0.70710678],
               [ 0.70710678,  0.70710678]])
    """
    theta = np.radians(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def homo_rotate2d(degrees):
    """Return a homogenous transformation of a rotation based on the given angle

    Args:
        degrees (int): The angle of degrees to rotate

    Returns:
        Iterable[Int]: numpy ndarray of a homogenous rotation matrix

    Examples:
        Returns a homogenous rotation identity matrix for the given angle in degrees:
        >>> homo_rotate2d(0)
        array([[ 1., -0.,  0.],
               [ 0.,  1.,  0.],
               [ 0.,  0.,  1.]])

        Returns a homogenous rotation matrix for the given angle in degrees:
        >>> homo_rotate2d(20)
        array([[ 0.93969262, -0.34202014,  0.        ],
               [ 0.34202014,  0.93969262,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])

        Returns a homogenous rotation matrix for the given angle in degrees:
        >>> homo_rotate2d(45)
        array([[ 0.70710678, -0.70710678,  0.        ],
               [ 0.70710678,  0.70710678,  0.        ],
               [ 0.        ,  0.        ,  1.        ]])
    """
    rotation_matrix = rotate2d(degrees)
    h = np.concatenate((rotation_matrix, np.array([[0], [0]])), axis=1)
    h = np.concatenate((h, np.array([[0, 0, 1]])), axis=0)
    return h
