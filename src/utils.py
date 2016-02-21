import numpy as np

from sklearn.cross_validation import ShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


submission_dir = '../submissions/'

def prepare_submission(submission_df, predictions, filename=''):
	if submission_df is None:
		raise ValueError('Submission dataframe is required')

	if predictions is None:
		raise ValueError('Predictions are required')

	if filename is None:
		filename = 'default.csv'

	submission_df['probability'] = predictions
	submission_df.to_csv(submission_dir + filename, index=False)


def eval_models(models, X, y):
	cv = ShuffleSplit(len(y), n_iter=5, test_size=0.33)

	scores = []
	for train, test in cv:
		scores_combined = np.zeros(len(test))

		for clf in models:
			X_train, y_train = X.iloc[train], y.iloc[train]
			X_test, y_test = X.iloc[test], y.iloc[test]
			clf.fit(X_train, y_train)

			sig_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')		
			sig_clf.fit(X_test, y_test)
			
			auc = sig_clf.predict_proba(X_test)[:, 1]
			
			print("score: %f" % roc_auc_score(y_test, auc))
			scores_combined += auc

		scores_combined /= len(models) * 1.
		scores.append(roc_auc_score(y_test, scores_combined))
		print("combined score: %f" % scores[-1])

	return (np.mean(scores), np.std(scores))