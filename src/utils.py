import numpy as np

from sklearn.cross_validation import ShuffleSplit, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression


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

			# sig_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')		
			# sig_clf.fit(X_test, y_test)
			
			auc = clf.predict_proba(X_test)[:, 1]
			
			print("score: %f" % roc_auc_score(y_test, auc))
			scores_combined += auc

		scores_combined /= len(models) * 1.
		scores.append(roc_auc_score(y_test, scores_combined))
		print("combined score: %f" % scores[-1])

	return (np.mean(scores), np.std(scores))


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def transform_for_ranked(preds, index):
	ranks = []

	for i, pred in enumerate(preds):
		ranks.append((index[i], pred))

	return ranks

def stacked_blending(clfs, train, y, test):
	X = train
	y = y
	X_submission = test

	skf = list(StratifiedKFold(y, 3))

	print 'Creating train and test sets for blending.'

	dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
	dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

	for j, clf in enumerate(clfs):
		print j, clf
		dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
		for i, (train, test) in enumerate(skf):
			print "Fold", i
			
			X_train = X.iloc[train]
			y_train = y.iloc[train]
			X_test = X.iloc[test]
			y_test = y.iloc[test]
			clf.fit(X_train, y_train)
			y_submission = clf.predict_proba(X_test)[:,1]
			dataset_blend_train[test, j] = y_submission
			dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
		dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

	print
	print "Blending."
	clf = LogisticRegression(class_weight='auto')
	clf.fit(dataset_blend_train, y)
	y_submission = clf.predict_proba(dataset_blend_test)[:,1]

	# print "Linear stretch of predictions to [0,1]"
	# y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

	return y_submission

