from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

class Models():
	def __init__(self):
		pass


	def logistic_regression_model(self):
		"""
		Scored 0.5371 on the leaderboard
		Scored 0.5390 with reduced features
		"""
		scaler = StandardScaler()

		params = {'C': 0.1}
		clf = LogisticRegression(**params)

		pipeline = Pipeline([('scaler', scaler), ('clf', clf)])

		return pipeline

	def dummy_classifier(self):
		"""
		Scored 0.4909 on the leaderboard
		"""
		
		dummy = DummyClassifier()
		return dummy


	def random_forest_classifier(self):
		"""
		Scored 0.5271 on the leaderboard
		"""

		scaler = StandardScaler()
		params = {'n_estimators': 300, 'max_depth': 5, 'n_jobs': -1,
		          'min_samples_split': 5}

		clf = RandomForestClassifier(**params)
		pipeline = Pipeline([('scaler', scaler), ('clf', clf)])

		return pipeline

