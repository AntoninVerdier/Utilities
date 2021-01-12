# -*- coding: utf-8 -*-
"""
@author: Antonin
"""
import os as os
import numpy as np
import matplotlib.pyplot as plt
import elphy_reader as ertd

def get_mouse_data(folder, tag=None):
	all_files = [folder + file for file in os.listdir(folder)]
	if tag:
	 	all_files = [file for file in all_files if tag in file]

	return all_files


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

mouses = {}
for i, folder in enumerate(mouse_folder):
	mouses[folder]= get_mouse_data(folder, tag='PC')
	frequencies_M0, prob_M0 = get_P_lick(mouse_files)

folder = '/home/pouple/PhD/Code/DAT_Analysis/M0_All/M0_SP/'
all_files = get_mouse_data(folder, tag='PC')
frequencies_M0, prob_M0 = get_P_lick(all_files)


folder = '/home/pouple/PhD/Code/DAT_Analysis/M1_All/PC/'
all_files = [folder + file for file in os.listdir(folder)]
frequencies_M1, prob_M1 = get_P_lick(all_files)


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(frequencies_M0, prob_M0)
axs[0, 0].set_xscale('log')
axs[0, 0].set_xticks([4e3, 5e3, 6e3, 7e3, 9e3, 11e3, 13.189e3, 16e3])
axs[0, 1].plot(frequencies_M1, prob_M1)
axs[0, 1].set_xscale('log')

axs[1, 0].plot(frequencies_M0, (prob_M0+prob_M1)/2)

plt.show()






