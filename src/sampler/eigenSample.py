import numpy as np
from sklearn.cluster import KMeans


class EigenSample:
	"""
	EigenSample: Python package for generating synthetic samples in eigenspace to minimize distortion

	Attributes:
		data (ndarray): Sample data
		target (ndarray): Targer/labels for samples
		model (scikit-learn model): Classification or regression model from scikit-learn
	"""

	# Initialize objects
	def __init__(self, data, target, model):
		"""
		Initializes an EigenSample object
	
		Parameters:
			data (ndarray): Sample data
			target (ndarray): Targer/labels for samples
			model (scikit-learn model): Classification or regression model from scikit-learn
		"""

		assert isinstance(data, np.ndarray) == True , "Error: data object is not ndarray"
		assert isinstance(target, np.ndarray) == True , "Error: target object is not ndarray"
		self.data = data
		self.target = target
		self.model = model

	# Function for creating new samples
	def add_samples(self, mid_point = 0.5):
		"""
		Generate synthetic samples in eigenspace
	
		Parameters:
			mid_point (int): any value between 0 and 1
		"""

		# STEP-1: Mean center the dataset
		mean = np.mean(self.data, axis=0)
		centered_data = self.data - mean

		# STEP-2: Compute correlation matrix
		corr_matrix = np.cov(centered_data, rowvar = False)

		# STEP-3.1: Compute eigenvalues and eigenvectors
		evals, evecs = np.linalg.eig(corr_matrix)

		# STEP-3.2: Sort eigenvalues in descending order
		sorted_idx = np.argsort(evals)[::-1]
		sorted_evals = evals[sorted_idx]

		# STEP_3.3: Choose k based on 90% threshold
		threshold = 0.9 # threshold
		ratio = np.cumsum(sorted_evals**2)/(np.sum(sorted_evals**2)) # cumulative explained variance
		ratio = np.round(ratio, 2) # rounding values to 2 decimal points
		k = len(np.where(ratio<=threshold)[0]) # selecting k based on threshold
		k = 1 if k == 0 else k # Condition to make sure k is +vie intiger

		# STEP-4: Obtain the projection matrix for k eigenvalues
		proj_matrix = evecs[:, sorted_idx[:k]] # subset eigenvectors based on k

		# STEP-5: Compute projection data points
		proj_data = np.dot(centered_data, proj_matrix) # PCA Scores

		# STEP-6: Clustering
		kmeans = KMeans(n_clusters = len(set(self.target)), init = "k-means++", max_iter = 100, n_init = 10, random_state = 42)
		y_clusters = kmeans.fit_predict(proj_data) # clustering
		centers = kmeans.cluster_centers_ # cluster centers
		y_labels = kmeans.labels_ # cluster labels

		# STEP-7: Obtaining new datapoints
		# mid point should be >0 and <1
		new_data = (proj_data + centers[y_labels]) * mid_point

		# STEP-8: Obtaining target values for new datapoints by training a regressor/classifier
		self.model.fit(proj_data, self.target)
		new_target = self.model.predict(new_data)
		
		# Inverse PCA transformation
		Xhat = np.dot(new_data, proj_matrix.T) + mean
		
		# Return new_data and new_targets
		return {"new_data":Xhat, "new_target":new_target}


