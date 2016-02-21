
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
