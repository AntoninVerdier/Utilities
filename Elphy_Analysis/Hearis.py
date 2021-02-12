""" This file groups all low and high level analysis function for behavioural data
"""
import numpy as np
import elphy_reader as ertd


def get_P_lick(tasks, licks):
	""" Return the probability of lick for each stimulus presented
	"""

	tasks, licks = np.array(tasks).reshape(1, -1)[0], np.array(licks).reshape(1, -1)[0]

	P_lick = {key:sum(tasks*licks == i+1)/sum(tasks == i+1) for i, key in enumerate(list(set(tasks)))}

	sorted_P_licks = sorted(P_lick.items())
	frequencies, prob = zip(*sorted_P_licks)

	return np.array(prob)