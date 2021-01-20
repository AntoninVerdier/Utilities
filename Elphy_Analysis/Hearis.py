""" This file groups all low and high level analysis function for behavioural data
"""
import numpy as np 
import elphy_reader as ertd


def get_P_lick(paths, params_multi=[4e3, 5e3, 6e3, 7e3, 9e3, 11e3, 13.189e3, 16e3], lick_treshold=4):
	""" Return the probability of lick for each stimulus presented

	Parameters
	----------
	filenames: iterable
		List of paths of Elphy DAT files

	Returns
	-------
	list
		Sorted list of parameters
	list
		Sorted list of probabilities of lick for each parameter

	"""
	# read the first file of the list of files
	tasks, licks = [], []
	for path in paths:
		recordings, vectors, xpar = ertd.read_behavior(path, verbose=False)
		# indicies = [i for i in vectors['TRECORD'] if i == 0]
		# del tasks[indicies]
		# del licks[indicies]

		tasks.append(vectors['TRECORD'])
		licks.append(vectors['LICKRECORD'] >= lick_treshold)
		#licks.append(vectors['correct'])

	tasks, licks = np.array(tasks).reshape(1, -1)[0], np.array(licks).reshape(1, -1)[0]

	P_lick = {key:sum(tasks*licks == i+1)/sum(tasks == i+1) for i, key in enumerate(list(set(tasks)))}

	sorted_P_licks = sorted(P_lick.items())
	frequencies, prob = zip(*sorted_P_licks)

	return np.array(params_multi), np.array(prob)
