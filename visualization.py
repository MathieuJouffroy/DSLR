import pandas as pd

def accuracy(y_true, y_pred):
	'''Calculate the accuracy score of our predictions'''
	accuracy = (y_pred == y_true).mean()
	return accuracy

def confusion_matrix(classes, y_true, y_pred):
	'''Generate the confusion matrix for our data'''
	# diagonal values for true positive 
	# false positive on the horizontal axis (predicted house when was not house)
	# false negative on the vertical axis (predicted not house when was house)
	cf_matrix = pd.DataFrame(0, columns=classes, index=classes)
	for i in range(len(y_pred)):
		for house in cf_matrix.columns:
			other = [x for x in classes if x != house]
			if y_pred[i] == house and y_true[i] == house:
				cf_matrix.loc[house][house] += 1

			for j in range(len(other)):
				if y_pred[i] == house and y_true[i] == other[j]:
					cf_matrix.loc[other[j]][house] += 1
	return cf_matrix
# 