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

    def test_translate2d_by_1_in_x_will_return_translate2d_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 1, 0)
        expected = np.array([1, 0])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_1_in_y_will_return_translate2d_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 0, 1)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_1_x_and_1_y_will_return_translate2d_vector(self):
        vector = np.array([0, 0])
        result = transform2d.translate2d(vector, 1, 1)
        expected = np.array([1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_by_x_and_y_will_return_translate2d_vector(self):
        vector = np.array([3, 1])
        result = transform2d.translate2d(vector, 5, 8)
        expected = np.array([8, 9])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_with_scalar_will_boardcast(self):
        scalar = 1
        result = transform2d.translate2d(scalar, 1, 1)
        expected = np.array([2, 2])
        np.testing.assert_array_equal(result, expected)

    def test_translate2d_with_broadcastable_vector_will_boardcast(self):
        boardcastable_vector = np.array([[1, 1], [1, 1]])
        result = transform2d.translate2d(boardcastable_vector, 1, 1)
        expected = np.array([[2, 2], [2, 2]])

    def test_translate2d_with_nonbroadcastable_vector_will_raise_exception(
            self):
        vector = np.array([0, 0, 0])
        self.assertRaises(ValueError, transform2d.translate2d, vector, 1, 1)

    def test_rotate2d_by_0_degrees_will_return_original_vector(self):
        vector = np.array([0, 0])
        result = transform2d.rotate2d(vector, 0)
        expected = np.array([0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_rotate2d_origin_vectors_rotated_will_return_an_origin_vector(
            self):
        vector = np.array([0, 0])
        result = transform2d.rotate2d(vector, 20)
        expected = np.array([0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_rotated2d_vector_rotated_by_20_degrees_will_return_rotated_vector(
            self):
        vector = np.array([1, 0])
        result = transform2d.rotate2d(vector, 20)
        expected = np.array([0.93969262, 0.34202014])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate2d_with_scalar_will_boardcast(self):
        scalar = 1
        result = transform2d.rotate2d(scalar, 20)
        expected = np.array([[0.93969262, -0.34202014],
                             [0.34202014, 0.93969262]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate2d_with_broadcastable_vector_will_boardcast(self):
        boardcastable_vector = np.array([[1, 1], [1, 1]])
        result = transform2d.rotate2d(boardcastable_vector, 20)
        expected = np.array([[0.59767248, 0.59767248],
                             [1.28171276, 1.28171276]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate2d_with_nonbroadcastable_vector_will_raise_exception(self):
        vector = np.array([0, 0, 0])
        self.assertRaises(ValueError, transform2d.rotate2d, vector, 20)


if __name__ == '__main__':
    unittest.main()
