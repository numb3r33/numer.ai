import numpy as np
import pandas as pd

class Munger():
	def __init__(self, train_df, validation_df, test_df):
		if train_df is None:
			raise ValueError('training dataframe is required')
		
		if test_df is None:
			raise ValueError('test dataframe is requied')

		if validation_df is None:
			raise ValueError('validation dataframe is required')

		self.train = train_df
		self.validation = validation_df
		self.test = test_df

		## separate out features from target
		features = train_df.columns[:-3]
		
		# training examples
		self.X = self.train[features].astype(np.float)
		self.y = self.train.target

		# validation examples
		self.X_validation = self.validation[features].astype(np.float)
		self.y_validation = self.validation.target

		# whole dataset
		self.X_full = pd.concat([self.X, self.X_validation])
		self.y_full = pd.concat([self.y, self.y_validation])

		# test examples
		self.test = test_df[test_df.columns[1:-1]].astype(np.float)

	def remove_correlated_features(self, correlated_features=[]):
		if len(correlated_features) == 0:
			correlated_features = [('f1', 'f8'), ('f2', 'f12'), ('f3', 'f9'),
			                       ('f4', 'f5'), ('f6', 'f13'), ('f10', 'f11'),
			                       ('f7', 'f14')]

		features_to_consider = [feat[0] for feat in correlated_features]

		self.X = self.train[features_to_consider]
		self.X_validation = self.validation[features_to_consider]
		self.X_full = self.X_full[features_to_consider]
		self.test = self.test[features_to_consider]

