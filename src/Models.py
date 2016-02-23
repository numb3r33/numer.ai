from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, chi2


from features import FeatureTransformer

from xgboost import XGBClassifier

class Models():
	def __init__(self):
		pass


	def logistic_regression_model(self):
		"""
		Scored 0.5371 on the leaderboard
		
		Scored 0.5390 with reduced features
		
		Scored 0.5290 with one hot encoded features
		
		Scored 0.5386 with label encoded features
		after calibration
		
		Scored 0.5271 with polynomial features
		after calibration

		Scored 0.5376 with all interaction features and
		then feature selection

		Scored 0.5392 with KMeans Clustering

		Scored 0.5403 with KMeans and other interaction
		features
		"""

		ft = FeatureTransformer()
		scaler = StandardScaler()
		select = SelectKBest(chi2, k=6)
		
		params = {'C': 0.08, 'penalty': 'l1',
		          'class_weight': 'auto'}

		clf = LogisticRegression(**params)

		pipeline = Pipeline([('ft', ft), ('select', select),
			                 ('scaler', scaler),('clf', clf)])

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
		
		Scored 0.5214 on the leaderboard with
		one hot encoded features and parameter
		settings as follows:
			
		params = {'n_estimators': 100, 'max_depth': 3, 
		          'n_jobs': -1}

		"""

		scaler = StandardScaler()
		params = {'n_estimators': 100, 'max_depth': 3, 
		          'n_jobs': -1}

		clf = RandomForestClassifier(**params)
		pipeline = Pipeline([('scaler', scaler), ('clf', clf)])

		return pipeline


	def extreme_gbm(self):
		"""
		Scored 0.529 on the leaderboard with all the
		interaction features and feature selection.
		"""

		ft = FeatureTransformer()
		scaler = StandardScaler()
		select = SelectKBest(chi2, k=7)

		params = {'n_estimators': 750, 'learning_rate': 0.03}

		clf = XGBClassifier(**params)
		pipeline = Pipeline([('ft', ft), ('select', select),
			                ('scaler', scaler), ('clf', clf)])

		return pipeline

