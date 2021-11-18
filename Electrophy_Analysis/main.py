import os
import rich
import scipy.io as sio
import numpy as np
import h5py

from rich import print
from rich.progress import track
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

from sklearn import svm


import settings as s

paths = s.paths()
params = s.params()

# Load files from KS directory
spikes_times = np.load(os.path.join(paths.Ksdir,'spike_times.npy'))
spikes_cluster = np.load(os.path.join(paths.Ksdir,'spike_clusters.npy'))
coords = np.load(os.path.join(paths.Ksdir,'channel_positions.npy'))
cluster_group = pd.read_csv(os.path.join(paths.Ksdir,'cluster_group.tsv'), delimiter='\t')
cluster_group = cluster_group['group'].to_numpy()

spikes_times = np.array([i[0] for i in spikes_times])



# Not necessary to load for PSTHs
# spikes_templates = np.load(os.path.join(paths.Ksdir,'spike_templates.npy'))
# templates = np.load(os.path.join(paths.Ksdir,'templates.npy'))
# spikes_amplitude = np.load(os.path.join(paths.Ksdir, 'amplitudes.npy'))
# winv = np.load(os.path.join(paths.Ksdir,'whitening_mat_inv.npy'))


sound_info = sio.loadmat(paths.SoundInfo)

# Be careful with 1-indexing of matlab so remove 1
stim_vector = np.squeeze(sound_info['StimsVector']) - 1
sound_names = np.array([n[0] for n in np.squeeze(sound_info['SoundNames'])])

print(sound_names[params.task1])

# Get cluster numbers that are not noise
usable_clusters = [i for i, clu in enumerate(cluster_group) if clu == 'good']

# Clean spike timings to keep only good clusters
usable_spikes = np.isin(spikes_cluster, usable_clusters)
spikes_times = spikes_times[usable_spikes]
spikes_cluster = spikes_cluster[usable_spikes]


# Load ttl infos
f = open(paths.digitalin, 'rb') 
sound_array = np.fromfile(f, np.int16)
f.close()
ttl_indices = np.where(np.insert(np.diff(sound_array), 0, 0) == 1)[0]

idxs_clu = []
for j, clu in enumerate(np.unique(spikes_cluster)):
	idxs_clu.append(np.where(spikes_cluster == clu)[0])

# Structure of array is dict[stim_number][presentation_number] containg a number_of_cluster * spikes matrix
d_stims = {}
for stim in track(np.unique(stim_vector), description='Generating individual timings ...'):
	stim_ttls = ttl_indices[np.where(stim_vector == stim)[0]]
	d_stims[stim] = {}
	for i, ttl in enumerate(stim_ttls):
		spikes = spikes_times[spikes_times > ttl - params.pad_before]
		spikes = np.array(spikes[spikes < ttl + params.pad_after], dtype=np.int64)

		d_stims[stim][i] = []
		for idx_clu in idxs_clu:
			spikes_per_clu = (np.intersect1d(spikes_times[idx_clu], spikes) - ttl)/(params.fs*1000)
			d_stims[stim][i].append(len(spikes_per_clu))

pop_vectors = [d_stims[k] for k in params.task1]
X = np.array([pres[k] for pres in pop_vectors for k in pres])
y = np.tile(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]), 15)

X_train = X[:-16]
y_train = y[:-16]

print(X_train.shape, y_train.shape)

X_test = X[-16:]
y_test = y[-16:]

clf = svm.SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)




	
		







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
