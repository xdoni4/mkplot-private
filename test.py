import unittest
import numpy as np
from gradient import Gradient


class TestGradient(unittest.TestCase):
    def test_simple(self):
        g = Gradient('F', 'F', 'F', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, np.ones((1, 5)))

        g = Gradient('F', 'x ** 2', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, np.array([[2], [4], [6], [8], [10]]))

        g = Gradient('F', 'k * x ** 10', 'x, k', 1)
        ans = g.compute(np.array([3]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 3 * 10 * np.array([[1], [2], [3], [4], [5]]) ** 9)

        g = Gradient('F', 'log(x)', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 1 / np.array([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'exp(x)', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, np.exp([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'sin(x)', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, np.cos([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'cos(x)', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, -np.sin([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'tan(x)', 'x', 1)
        ans = g.compute(np.array([]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 1 / np.cos([[1], [2], [3], [4], [5]]) ** 2)

        g = Gradient('F', 'x * y', 'x y', 2)
        ans = g.compute(np.array([]), np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))
        assert np.allclose(ans, np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))

        g = Gradient('F', 'x / y', 'x y', 2)
        ans = g.compute(np.array([]), np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))
        assert np.allclose(ans, np.array([[1, -1], [1 / 2, -1 / 2], [1 / 3, -1 / 3], [1 / 4, -1 / 4], [1 / 5, -1 / 5]]))

    def test_extreme_values(self):
        g = Gradient('F', 'k * exp(x)', 'x k', 1)
        ans = g.compute(np.array([1e-9]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 1e-9 * np.exp([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'k * exp(x)', 'x k', 1)
        ans = g.compute(np.array([1e-18]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 1e-18 * np.exp([[1], [2], [3], [4], [5]]))

        g = Gradient('F', 'k * exp(x)', 'x k', 1)
        ans = g.compute(np.array([1e-27]), np.array([[1], [2], [3], [4], [5]]))
        assert np.allclose(ans, 1e-27 * np.exp([[1], [2], [3], [4], [5]]))


if __name__ == '__main__':
    unittest.main()
