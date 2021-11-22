import os
import numpy as np 
import pandas as pd
import scipy.io as sio

class Recording():
	"""docstring for Recording"""
	def __init__(self, ksdir, sound_info):
		self.ksdir = ksdir
		self.sound_path = sound_info
		self.__load_data()

	def __load_data(self):
		# Load files from KS directory
		sp_times = np.load(os.path.join(self.ksdir,'spike_times.npy'))
		self.sp_times = np.array([i[0] for i in sp_times])

		self.sp_clu = np.load(os.path.join(self.ksdir,'spike_clusters.npy'))
		
		cluster_group = pd.read_csv(os.path.join(self.ksdir,'cluster_group.tsv'), delimiter='\t')
		self.clu_gp = cluster_group['group'].to_numpy()

		# Be careful with 1-indexing of matlab so remove 1

		sound_info = sio.loadmat(self.sound_path)
		self.s_vector = np.squeeze(sound_info['StimsVector']) - 1
		self.sd_names = np.array([n[0] for n in np.squeeze(sound_info['SoundNames'])])

		f = open(os.path.join(self.ksdir, 'digitalin.dat'), 'rb') 
		sd_array = np.fromfile(f, np.int16)
		f.close()
		self.ttl_idxs = np.where(np.insert(np.diff(sd_array), 0, 0) == 1)[0]

		# Not necessary to load for PSTHs
		# spikes_templates = np.load(os.path.join(self.ksdir,'spike_templates.npy'))
		# templates = np.load(os.path.join(self.ksdir,'templates.npy'))
		# spikes_amplitude = np.load(os.path.join(self.ksdir, 'amplitudes.npy'))
		# winv = np.load(os.path.join(self.ksdir,'whitening_mat_inv.npy'))
		# coords = np.load(os.path.join(self.ksdir,'channel_positions.npy'))
	
	def clean_data(self, quality='good'):
		""" Extract soundnames and remove unwanted cluster categories"""

		# Get cluster numbers that are not noise
		if quality == 'good':
			usable_clusters = [i for i, clu in enumerate(self.clu_gp) if clu == 'good']
		elif quality =='mua':
			usable_clusters = [i for i, clu in enumerate(self.clu_gp) if not clu == 'noise']
		elif quality =='noise':
			usable_clusters = list(range(len(self.clu_gp)))

		# Clean spike timings to keep only good clusters
		self.usable_sp = np.isin(self.sp_clu, usable_clusters)
		self.sp_times = self.sp_times[self.usable_sp]
		self.sp_clu = self.sp_clu[self.usable_sp]

	def get_pop_vectors(spikes_cluster, spikes_times, stim_vector, ttl_indices, pad_after=params.pad_after, task=params.task1):
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
				spikes = np.array(spikes[spikes < ttl + pad_after], dtype=np.int64)

				d_stims[stim][i] = [len(np.intersect1d(spikes_times[idx_clu], spikes) - ttl)/(params.fs*1000) for idx_clu in idxs_clu]


		pop_vectors = [d_stims[k] for k in task]

		return pop_vectors
