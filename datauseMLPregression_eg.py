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
	###have a look at
	# 	https://blog.goodaudience.com/artificial-neural-networks-explained-436fcf36e75
	# 	https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110
	#	https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/#:~:text=Gradient%20descent%20is%20an%20iterative,function%20at%20the%20current%20point.
	# https://builtin.com/data-science/gradient-descent
	# https://medium.com/@daniel.hellwig.p/mathematical-representation-of-a-perceptron-layer-with-example-in-tensorflow-754a38833b44

	### from above note b=a-gamma*DEL(f(a))
	# b is next pos, a is current pos
	# gamma is step size
	# DEL(f(a)) is the direction to the steepest descent

	dm = DataManager("test")

	##get the testset
	# just get 1000 held back examples
	X_test, Y_test, finished = dm.get_testset_batch(1000)
	X_test = X_test.reshape((-1, 1))

	nepochs = 1000
	n_samples, n_features = 10, 5

	# stochastic gradient descent: the gradient of the loss is estimated each sample at a time
	# and the model is updated along the way with a decreasing strength schedule (aka learning rate)
	#reg = SGDRegressor(max_iter=1, tol=1e-3)

	# one hidden layer
	# reg= MLPRegressor(hidden_layer_sizes=[50],learning_rate ="adaptive",random_state=1, max_iter=1,shuffle=False)      #takes >65 epochs to get 0.99999

	# 2 hidden layers
	# reg= MLPRegressor(hidden_layer_sizes=[40,10],learning_rate ="adaptive",random_state=1, max_iter=1,shuffle=False)  #takes 19 epochs to get 0.99999

	# 3 hidden layers???
	reg = MLPRegressor(hidden_layer_sizes=[40, 5, 5], learning_rate="adaptive", random_state=1, max_iter=1,shuffle=False)  # takes 15 epochs to get 0.99999

	# break threshold for the function
	breakthresh = 0.99999

	fin = False
	print()

	best = -1

	for i in range(nepochs):

		done = False

		while not done:
			X, Y, done = dm.get_batch()

			if not done:
				# note - just for convenience won't do the last batch because it might be smaller than batch size
				X = X.reshape((-1, 1))

				# do one step/ one iteration
				# require (1, 2) shape
				# batch-wise process
				reg.partial_fit(X, Y)

			# get the R^2 value
			score = reg.score(X_test, Y_test)

			print(f"\r step: {dm.batchcounter} \t score:{score} ",end="")

			if score > breakthresh:

				# get the best fitting
				print(f"\n Finished")
				fin = True

				break

		if score > best:
			best = score
			isbest = True

		else:
			isbest = False

		print(f"\r------ epoch: {dm.epochcounter} \t {[' ','*'][isbest]}score:{score} ---------------------")
		if fin == True:
			break

	# check regression with the actual Y data
	test = reg.predict(X_test[:3])
	actual = Y_test[:3]
	print(f"Predict:{test} \n Actual:{actual}")
