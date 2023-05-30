import numpy as np
from eeghelpers import *
from dataManager import Manager as DataManager

if __name__ == '__main__':

	# cannot find what this datamanager thingy is!!!!
	dm = DataManager("test", testsplit=.2)
	folder = os.path.abspath("test")

	#now generate a datasets to save.
	# Often we have more data than we can fit into memory, so we have to do chunks of it.
	saveevery = 1000

	for i in range(1000):
		# create x and respective y values
		# choose 1000 random numbers between 0 and 100,000,000
		X = np.random.uniform(0.0, 1.0, 1000)
		Y_true = np.square(X)

		dm.saveData(X, Y_true, n_per_file=300)

	print(f"Done")