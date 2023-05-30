import numpy as np

from eeghelpers import *

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.neural_network import MLPRegressor


if __name__ == '__main__':
	###FIRST YOU SHOULD RUN datasavingexample to create a training set.

	dm = DataManager("test")

	##get the test set
	# just get 1000 held back examples
	X_test, Y_test, finished = dm.get_testset_batch(1000)

	# number of elements extracted
	nepochs=1000

	for i in range(nepochs):

		done = False

		while not done:
			X, Y, done = dm.get_batch()
			X = X.reshape((-1, 1))
		#you need to always make sure your shape fits with the expection of the model.
		###NOW YOU CAN USE X and Y for a single training step.
		###X_test,Y_test are your test set and should be used to verify that learning has occured.

