import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sampler import EigenSample

class TestEigenSample(unittest.TestCase):

	def setUp(self):
		self.data = load_iris().data
		self.target = load_iris().target
		self.model= RandomForestClassifier(random_state=42)

	def test_EigenSample(self):
		output = EigenSample(self.data, self.target, self.model).add_samples()
		self.assertIsInstance(output, dict) 
		self.assertIn('new_data', output)
		self.assertIn('new_target', output)
		self.assertIsInstance(output['new_data'], np.ndarray)
		self.assertIsInstance(output['new_target'], np.ndarray)
		self.assertEqual(output["new_data"].shape, self.data.shape)
		self.assertEqual(output["new_target"].shape, self.target.shape)


if __name__ == "__main__":
	unittest.main()
