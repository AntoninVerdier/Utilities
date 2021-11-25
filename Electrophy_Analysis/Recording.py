import os
import time
import numpy as np 
import pandas as pd
import scipy.io as sio
import settings as s

from rich.progress import track

params = s.params()
class Recording():
	"""docstring for Recording"""
	def __init__(self, ksdir, sound_info):
		self.ksdir = ksdir
		self.sound_path = sound_info
		self.date = 0 # To define
		self.__load_data()

	def __load_data(self):
		# Load files from KS directory
		sp_times = np.load(os.path.join(self.ksdir,'spike_times.npy'))
		self.sp_times = np.array([i[0] for i in sp_times])

		self.sp_clu = np.load(os.path.join(self.ksdir,'spike_clusters.npy'))
		
		clu_gp = pd.read_csv(os.path.join(self.ksdir,'cluster_group.tsv'), delimiter='\t')
		self.clu_gp = clu_gp['group'].to_numpy()

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
	
	def select_data_quality(self, quality='good'):
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

	def new_ttl_alignment(self):
		idxs_clu = []
		for clu in np.unique(self.sp_clu):
			idxs_clu.append(np.where(self.sp_clu == clu)[0])
		d_stim = {k:{i: [ for idx in idxs_clu] 
				  for i, ttl in enumerate(self.ttl_idxs[np.where(self.s_vector == k)[0]])} 
				  for k in np.unique(self.s_vector)}
		print(d_stim)

	def ttl_alignmnent(self, pad_before=params.pad_before, pad_after=params.pad_after):
		idxs_clu = []
		for clu in np.unique(self.sp_clu):
			idxs_clu.append(np.where(self.sp_clu == clu)[0])

		# Structure of array is dict[stim_number][presentation_number] containg a number_of_cluster * spikes matrix
		self.d_stims = {}
		for stim in track(np.unique(self.s_vector), description='Aligning neural activity to ttls ...'):
			stim_ttls = self.ttl_idxs[np.where(self.s_vector == stim)[0]]
			d_stims[stim] = {}
			for i, ttl in enumerate(stim_ttls):
				spikes = np.array(self.sp_times[np.logical_and(self.sp_times > ttl - pad_before, self.sp_times < ttl + pad_after)], dtype=np.int64)


				self.d_stims[stim][i] = [len(np.intersect1d(self.sp_times[idx_clu], spikes) - ttl)/(params.fs*1000) for idx_clu in idxs_clu]



	# def get_pop_vectors(self, pad_before=params.pad_before, pad_after=params.pad_after, task=params.task1):
	# 	idxs_clu = []
	# 	for j, clu in enumerate(np.unique(self.sp_clu)):
	# 		idxs_clu.append(np.where(self.sp_clu == clu)[0])

	# 	# Structure of array is dict[stim_number][presentation_number] containg a number_of_cluster * spikes matrix
	# 	d_stims = {}
	# 	for stim in track(np.unique(self.s_vector), description='Generating individual timings ...'):
	# 		stim_ttls = self.ttl_idxs[np.where(self.s_vector == stim)[0]]
	# 		d_stims[stim] = {}
	# 		for i, ttl in enumerate(stim_ttls):
	# 			spikes = self.sp_times[self.sp_times > ttl - params.pad_before]
	# 			spikes = np.array(spikes[spikes < ttl + pad_after], dtype=np.int64)
	# 			print(len(idxs_clu[i]))
	# 			print(self.sp_times[idxs_clu[i]], spikes)

	# 			d_stims[stim][i] = [len(np.intersect1d(self.sp_times[idx_clu], spikes) - ttl)/(params.fs*1000) for idx_clu in idxs_clu]


	# 	self.pop_vectors = [d_stims[k] for k in task]