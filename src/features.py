from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from itertools import combinations
import pandas as pd
import numpy as np



class FeatureTransformer(BaseEstimator):
	def __init__(self): 
		pass


	def get_feature_names(self):
		feature_names = []

		return np.array(feature_names)


	def fit(self, X, y=None):
		self.fit_transform(X, y)

		return self


	def fit_transform(self, X, y=None):
		self.kmeans = KMeans(n_clusters=2, n_jobs=-1, n_init=5)
		self.numerical_features = X.select_dtypes(exclude=['object']).columns
		
		addition_interaction_features = self.get_addition_interaction_features(X)
		multiplication_interaction_features = self.get_multiply_interaction_features(X)
		division_interaction_features = self.get_division_interaction_features(X)

		self.kmeans.fit(X)
		cluster_labels = self.kmeans.predict(X).reshape(-1, 1)

		features = []
		features.append(X[self.numerical_features])
		
		features.append(addition_interaction_features)
		features.append(multiplication_interaction_features)
		features.append(division_interaction_features)
		
		features.append(cluster_labels)
		features = np.hstack(features)

		features = features.astype(np.float)
		
		return features

	def get_addition_interaction_features(self, df):
		numerical_features = self.numerical_features[:-1]
		addition_interactions = []

		for f1, f2 in combinations(numerical_features, 2):
			addition_interactions.append(df[f1] + df[f2])

		return np.array(addition_interactions).T

	def get_multiply_interaction_features(self, df):
		numerical_features = self.numerical_features[:-1]
		multiply_interactions = []

		for f1, f2 in combinations(numerical_features, 2):
			multiply_interactions.append(df[f1] * df[f2])

		return np.array(multiply_interactions).T

	def get_division_interaction_features(self, df):
		numerical_features = self.numerical_features[:-1]
		division_interactions = []

		for f1, f2 in combinations(numerical_features, 2):
			division_interactions.append(df[f1] / df[f2])

		return np.array(division_interactions).T

	def transform(self, X):
		cluster_labels = self.kmeans.predict(X).reshape(-1, 1)
		
		addition_interaction_features = self.get_addition_interaction_features(X)
		multiplication_interaction_features = self.get_multiply_interaction_features(X)
		division_interaction_features = self.get_division_interaction_features(X)

		features = []
		features.append(X[self.numerical_features])
		
		features.append(addition_interaction_features)
		features.append(multiplication_interaction_features)
		features.append(division_interaction_features)
		
		features.append(cluster_labels)
		features = np.hstack(features)

		features = features.astype(np.float)

		return features
