import os
import scipy.io as sio
import numpy as np
import h5py

from rich import print
from rich.progress import track
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut


import settings as s
from Recording import Recording

paths = s.paths()
params = s.params()

rec = Recording(paths.Ksdir, paths.SoundInfo, name='testing')
rec.select_data_quality(quality='good')
rec.ttl_alignment(multi=False)
rec.get_timings_vectors(0, 500)



def compute_svm(X, y):
	clf = svm.SVC()
	scores = cross_val_score(clf, X, y, cv=5)
	return scores

def svm_preformance(rec):
	for i, t in enumerate([params.task1, params.task2]):
		scores = []
		for p in track(np.arange(50, 1000, 50), description='Compute SVM for each task ...'):
			pop_vectors = rec.get_timings_vectors(0, p)

			X = np.array([pop_vectors[stim][p] for stim in t for p in pop_vectors[stim]])
			if i < 2:
				y = np.array([0 if i < 8 else 1 for i, stim in enumerate(t) for p in pop_vectors[stim]])

			score = compute_svm(X, y)
			scores.append([np.mean(score), np.std(score)])

		scores = np.array(scores).reshape(-1, 2)

		plt.errorbar(np.arange(0.05, 1, 0.05), scores[:, 0], label='Task {}'.format(1 if not i else 2))

	plt.legend()
	plt.savefig('performance_svm.png')
	plt.show()

svm_preformance(rec)

# Still need to drax psycoM curves

	
		



# Compute global N x T matrix
# dict_timings = {}
# for clu in track(usable_clusters, description='Computing individual cluster ...'):
# 	dict_timings[clu] = np.where(spikes_cluster == clu)[0]


# dict_stim_timings = {}
# for stim in track(np.unique(stim_vector), description='Generating individual timings ...'):
# 	curr_stim = np.where(stim_vector == stim)[0]
# 	dict_stim_timings[stim] = {}
# 	for i, pres in enumerate(curr_stim):
# 		ttl = ttl_indices[pres]
# 		dict_stim_timings[stim][i] = [(v[np.logical_and(v >= ttl - params.pad_before, v <= ttl + params.pad_after)] - ttl)/params.fs*1000 for v in list(dict_timings.values())]

# for stim in dict_stim_timings:
# 	curr_stim = [[sum(list(dict_stim_timings[stim][i][j])) for i in dict_stim_timings[stim]] for j in range(len(usable_clusters))]
# 	dict_stim_timings[stim] = [a for l in curr_stim for a in l if a]
# 	dict_stim_timings[stim].sort()
# # 	dict_stim_timings[stim] = [l for l in curr_stim if l]

# print([len(dict_stim_timings[stim]) for stim in dict_stim_timings])
	







# all_psth = {}
# for i in track(range(max(stim_vector)), description='Generating PSTHs...'):
# 	psths = [] 
# 	curr_stim = np.where(stim_vector == i)[0]
# 	for stim in curr_stim:
# 		curr_ttl = list(ttl_indices)[0::2][stim]

# 		curr_n_spikes = spikes_times[spikes_times > curr_ttl - params.pad_before]
# 		curr_n_spikes = np.array(curr_n_spikes[curr_n_spikes < curr_ttl + params.pad_after], dtype=np.int64)
# 		psths.append(np.array(curr_n_spikes - curr_ttl)/params.fs*1000)
	
# 	psth = [i for p in psths for i in p]
# 	all_psth[sound_names[i]] = psth



# for sound in track(all_psth, description='Drawing Figures...'):
# 	sns.set_theme(style='ticks')

# 	f, ax = plt.subplots(figsize=(7, 5))
# 	sns.despine(f)

# 	sns.histplot(data=all_psth[sound], palette='light:m_r', 
# 				 edgecolor='.3', linewidth=.5, bins=50)
# 	plt.axvline(0, color='red')
# 	ax.set_xlabel('Time (ms)')
# 	ax.set_ylabel('# of spikes')
# 	ax.set_title('{}'.format(sound[:-4]))
# 	plt.savefig(os.path.join(paths.Output, '{}.png'.format(sound[:-4])), dpi=150)
# 	plt.close()


# May be useful to get correlation matrices between histograms
