import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bacterial_growth_fitting_student import (
    V_model,
    W_model,
    fit_V_model,
    fit_W_model
)

class TestBacterialGrowthFitting(unittest.TestCase):
    def setUp(self):
        # 使用模拟数据进行测试
        self.t_test = np.linspace(0, 10, 100)
        self.V_data = V_model(self.t_test, 2.0) + np.random.normal(0, 0.01, 100)
        self.W_data = W_model(self.t_test, 1.0, 2.0) + np.random.normal(0, 0.01, 100)

    def test_V_model(self):
        """测试V(t)模型函数"""
        t_test = np.array([0, 1, 2, 3])
        tau = 2.0
        expected = np.array([0., 0.39346934, 0.63212056, 0.77686984])
        result = V_model(t_test, tau)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_W_model(self):
        """测试W(t)模型函数"""
        t_test = np.array([0, 1, 2, 3])
        A, tau = 1.0, 2.0
        expected = np.array([0., 0.10653066, 0.36787944, 0.72313016])
        result = W_model(t_test, A, tau)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_fit_V_model(self):
        """测试V(t)模型拟合"""
        popt_V, perr_V = fit_V_model(self.t_test, self.V_data)
        self.assertTrue(0.5 < popt_V[0] < 5.0)  # tau应该在合理范围内

    def test_fit_W_model(self):
        """测试W(t)模型拟合"""
        popt_W, perr_W = fit_W_model(self.t_test, self.W_data)
        self.assertTrue(0.5 < popt_W[0] < 2.0)  # A应该在合理范围内
        self.assertTrue(0.5 < popt_W[1] < 5.0)  # tau应该在合理范围内

if __name__ == "__main__":
    unittest.main()
