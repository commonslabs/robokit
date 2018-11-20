# -*- coding: utf-8 -*-
"""transform2d

This module contains several funtions to perform linear algebraic operations
on vectors and matricies, specifcally, but not limited to, common functions found
in the study of robotics.

This module depends heavily on the numpy package.

"""

import numpy as np


def translate(vector, x, y):
    """Return 2D vector translated by given x and y values

    Parameters
    ----------
    vector : Iterable[int]
        numpy ndarray of shape (2,)
    x : int
        The x translation.
    y : int
        The y translation.

    Returns
    ----------
    Iterable[Int]
        numpy ndarray of the x- and y-translation applied to the vector.

    Raises
    ----------
    ValueError
        If `vector` is not of shape `(2,)`

    Examples
    ----------
    Translates x and y of origin vector:
    >>> translate(np.array([0, 0]), 1, 1)
    array([1, 1])

    Translates x and y arbitrary vector:
    >>> translate(np.array([1, 3]), 2, 2)
    array([3, 5])

    """
    return np.array([x, y]) + vector
