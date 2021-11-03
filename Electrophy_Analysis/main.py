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

import settings as s

paths = s.paths()
params = s.params()

# Load files from KS directory

spikes_times = np.load(os.path.join(paths.Ksdir,'spike_times.npy'))
spikes_cluster = np.load(os.path.join(paths.Ksdir,'spike_clusters.npy'))

# Not necessary to load for PSTHs
# spikes_templates = np.load(os.path.join(paths.Ksdir,'spike_templates.npy'))
# templates = np.load(os.path.join(paths.Ksdir,'templates.npy'))
# spikes_amplitude = np.load(os.path.join(paths.Ksdir, 'amplitudes.npy'))
# winv = np.load(os.path.join(paths.Ksdir,'whitening_mat_inv.npy'))
# coords = np.load(os.path.join(paths.Ksdir,'channel_positions.npy'))

cluster_group = pd.read_csv(os.path.join(paths.Ksdir,'cluster_group.tsv'), delimiter='\t')
cluster_group = cluster_group['group'].to_numpy()


sound_info = sio.loadmat(paths.SoundInfo)

# Be careful with 1-indexing of matlab so remove 1
stim_vector = np.squeeze(sound_info['StimsVector']) - 1
sound_names = [n[0] for n in np.squeeze(sound_info['SoundNames'])]

# Get cluster numbers that are not noise
usable_clusters = [i for i, clu in enumerate(cluster_group) if not clu == 'noise']

usable_spikes = np.isin(spikes_cluster, usable_clusters)
spikes_times = spikes_times[usable_spikes]

f = open(paths.digitalin, 'rb') 
sound_array = np.fromfile(f, np.int16)
ttl_indices = np.where(np.diff(sound_array))[0]

all_psth = {}
for i in track(range(max(stim_vector)), description='Generating PSTHs...'):
	psths = [] 
	curr_stim = np.where(stim_vector == i)[0]
	for stim in curr_stim:
		curr_ttl = list(ttl_indices)[0::2][stim]

		curr_n_spikes = spikes_times[spikes_times > curr_ttl - params.pad_before]
		curr_n_spikes = np.array(curr_n_spikes[curr_n_spikes < curr_ttl + params.pad_after], dtype=np.int64)
		psths.append(np.array(curr_n_spikes - curr_ttl)/params.fs*1000)
	
	psth = [i for p in psths for i in p]
	all_psth[sound_names[i]] = psth

for sound in track(all_psth, description='Drawing Figures...'):
	sns.set_theme(style='ticks')

	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)

	sns.histplot(data=all_psth[sound], palette='light:m_r', 
				 edgecolor='.3', linewidth=.5, bins=50)
	plt.axvline(0, color='red')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('# of spikes')
	ax.set_title('{}'.format(sound[:-4]))
	plt.savefig('Output/{}.png'.format(sound[:-4]), dpi=150)
	plt.close()