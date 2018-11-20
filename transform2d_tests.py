import unittest
from numpy.testing import assert_, assert_raises
import numpy as np
import transform2d


class TestTransform2d(unittest.TestCase):
    def test_translate2d_by_0_will_return_original_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 0, 0)
        expected = vector
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_1_in_x_will_return_translate2dd_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 1, 0)
        expected = np.array([1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_1_in_y_will_return_translate2dd_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 0, 1)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_1_x_and_1_y_will_return_translate2dd_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 1, 1)
        expected = np.array([1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_x_and_y_will_return_translate2dd_vector(self):
        vector = np.array([3, 1])
        result = transform2d.translate2d(vector, 5, 8)
        expected = np.array([8, 9])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_with_3d_vector_will_raise_exception(self):
        vector = np.array([0, 0, 0])
        self.assertRaises(ValueError, transform2d.translate2d, vector, 1, 1)


if __name__ == '__main__':
    unittest.main()
