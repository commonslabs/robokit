import unittest
from numpy.testing import assert_, assert_raises
import numpy as np
import transform2d


class TestTransform2d(unittest.TestCase):
    def test_homo_translate2d_returns_a_translation_identity_matrix(self):
        result = transform2d.homo_translate2d(0, 0)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_homo_translate2d_returns_a_translation_matrix_in_x(self):
        result = transform2d.homo_translate2d(1, 0)
        expected = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_homo_translate2d_returns_a_translation_matrix_in_y(self):
        result = transform2d.homo_translate2d(0, 1)
        expected = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_homo_translate2d_returns_a_translation_matrix_in_x_and_y(self):
        result = transform2d.homo_translate2d(1, 1)
        expected = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_homo_translate2d_returns_a_translation_matrix(self):
        result = transform2d.homo_translate2d(5, 8)
        expected = np.array([[1, 0, 5], [0, 1, 8], [0, 0, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_rotate2d_returns_a_rotation_identity_matrix(self):
        result = transform2d.rotate2d(0)
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate2d_returns_a_rotation_matrix(self):
        result = transform2d.rotate2d(20)
        expected = np.array([[0.93969262, -0.34202014],
                             [0.34202014, 0.93969262]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_homo_rotate2d_returns_a_homogenous_rotation_identity_matrix(self):
        result = transform2d.homo_rotate2d(0)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_homo_rotate2d_returns_a_homogenous_rotation_matrix(self):
        result = transform2d.homo_rotate2d(20)
        expected = np.array([[0.93969262, -0.34202014, 0],
                             [0.34202014, 0.93969262, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
