import os
import time
import numpy as np 
import pandas as pd
import pickle as pkl
import scipy.io as sio
import multiprocessing as m
import settings as s

from rich.progress import track

import matplotlib.pyplot as plt

params = s.params()
class Recording():
	"""docstring for Recording"""
	def __init__(self, ksdir, sound_info, name=None, fs=20000):
		self.fs = fs
		self.name = name
		self.ksdir = ksdir
		self.sound_path = sound_info
		self.date = 0 # To define
		self.idxs_clu = None
		self.output = os.path.join(self.ksdir, 'output')
		if not os.path.exists(self.output):
			os.makedirs(self.output)
		self.__load_data()

	def __add__(self, other, n_pres=15):
		selfkeys = [k for k in self.d_stims]
		otherkeys = [k for k in other.d_stims]
		common_keys = np.intersect1d(selfkeys, otherkeys)

		for sound in common_keys:
			for p in range(n_pres):
				self.d_stims[sound][p] += other.d_stims[sound][p]

		return self

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

	# def new_ttl_alignment(self):
	# 	idxs_clu = []
	# 	for clu in np.unique(self.sp_clu):
	# 		idxs_clu.append(np.where(self.sp_clu == clu)[0])
	# 	d_stim = {k:{i: [ for idx in idxs_clu]
	# 			  for i, ttl in enumerate(self.ttl_idxs[np.where(self.s_vector == k)[0]])}
	# 			  for k in np.unique(self.s_vector)}
	# 	print(d_stim)

	def __get_idxs_clu(self):
		self.idxs_clu = []
		for clu in np.unique(self.sp_clu):
			self.idxs_clu.append(np.where(self.sp_clu == clu)[0])
		self.n_clu = len(self.idxs_clu)


	def __ttl_alignment_single_cpu(self, pad_before, pad_after):
		if not self.idxs_clu:
			self.__get_idxs_clu()

		# Structure of array is dict[stim_number][presentation_number] containg a number_of_cluster * spikes matrix
		self.d_stims = {}
		for stim in track(np.unique(self.s_vector), description='Aligning neural activity to ttls ...'):
			stim_ttls = self.ttl_idxs[np.where(self.s_vector == stim)[0]]
			self.d_stims[stim] = {}
			for i, ttl in enumerate(stim_ttls):
				spikes = self.sp_times[self.sp_times > ttl + pad_before]
				spikes = np.array(spikes[spikes < ttl + pad_after], dtype=np.int64)

				self.d_stims[stim][i] = [(np.intersect1d(self.sp_times[idx_clu], spikes) - ttl)*1000/params.fs for idx_clu in self.idxs_clu]

	def ttl_alignment_multi(self, args):
		""" in dev
		"""
		if not self.idxs_clu:
			self.__get_idxs_clu()
		# Structure of array is dict[stim_number][presentation_number] containg a number_of_cluster * spikes matrix
		stim_ttls = self.ttl_idxs[np.where(self.s_vector == args[0])[0]]
		self.d_stims[args[0]] = {}
		for i, ttl in enumerate(stim_ttls):
			spikes = self.sp_times[self.sp_times > ttl + args[1]]
			spikes = np.array(spikes[spikes < ttl + args[2]], dtype=np.int64)
			self.d_stims[args[0]][i] = [(np.intersect1d(self.sp_times[idx_clu], spikes) - ttl)*1000/params.fs for idx_clu in self.idxs_clu]

	def ttl_alignment(self, pad_before=0, pad_after=20000, multi=True):
		if os.path.isfile(os.path.join(self.output, '{}_aligned_spikes.pkl'.format(self.name))):
			self.d_stims = pkl.load(open(os.path.join(self.output, '{}_aligned_spikes.pkl'.format(self.name)), 'rb'))
		else:
			self.d_stims = {}
			if multi:
				args = ([stim, pad_before, pad_after] for stim in track(np.unique(self.s_vector)))
				with m.Pool() as pool:
					pool.map(self.ttl_alignment_multi, args)
			else:
				self.__ttl_alignment_single_cpu(pad_before, pad_after)

			pkl.dump(self.d_stims, open(os.path.join(self.output, '{}_aligned_spikes.pkl'.format(self.name)), 'wb'))


	def get_population_vectors(self, pad_before, pad_after):
		pop_vectors = {}
		for stim in self.d_stims:
			pop_vectors[stim] = {}
			for pres in self.d_stims[stim]:
				pop_vector = []
				for i, clu in enumerate(self.d_stims[stim][pres]):
					sp = clu[clu > pad_before]
					sp = np.array(sp[sp < pad_after], dtype=np.int64)
					pop_vector.append(sp.shape[0])
				pop_vectors[stim][pres] = pop_vector

		self.pop_vectors = pop_vectors

		return pop_vectors

	def get_timings_vectors(self, pad_before, pad_after):
		time_vectors = {}
		bins = list(np.arange(pad_before, pad_after + 10, 10))
		for stim in self.d_stims:
			time_vectors[stim] = {}
			for pres in self.d_stims[stim]:
				time_vector = np.sort([s for clu in self.d_stims[stim][pres] for s in clu])
				time_vector = time_vector[time_vector > - pad_before]
				time_vector = np.array(time_vector[time_vector < pad_after], dtype=np.int64)
				time = np.histogram(time_vector, bins)[0]

				time_vectors[stim][pres] = time

		self.time_vectors = time_vectors

		return time_vectors

	def complete_vectors(self, pad_before, pad_after):
		bins = np.arange(0, 1000, 10)
		pop_vectors = {}
		for stim in self.d_stims:
			pop_vectors[stim] = {}
			for pres in self.d_stims[stim]:
				pop_vector = []
				for i, clu in enumerate(self.d_stims[stim][pres]):
					clu = clu[clu > - pad_before]
					clu = np.array(clu[clu < pad_after], dtype=np.int64)
					pop_vector.append(np.histogram(clu, bins)) 
				pop_vectors[stim][pres] = np.array(np.flatten(pop_vector))

		self.pop_vectors = pop_vectors

		return pop_vectors
	def __compute_svm(kernel='linear'):
		clf = svm.SVC()
		########## Need to make sure that data is shuffled
		scores = cross_val_score(clf, X, y, cv=5)
		return scores

	def svm_preformance(rec, task=params.task1, scale=np.arange(50, 1000, 50)):
		"""define all y_task
		"""
		scores = []
		for p in track(scale, description='Compute SVM ...'):
			pop_vectors = rec.get_population_vectors(0, p)

			X = np.array([pop_vectors[stim][p] for stim in task for p in pop_vectors[stim]])
			
			if i < 2:
				y = np.array([0 if i < 8 else 1 for i, stim in enumerate(task) for p in pop_vectors[stim]])
			elif i == 2:
				y = params.y_task3
			elif i == 3:
				y = params.y_task4

			score = compute_svm(X, y)
			scores.append([np.mean(score), np.std(score)])

		scores = np.array(scores).reshape(-1, 2)

		plt.errorbar(scale, scores[:, 0], label='Task {}'.format(i + 1))

		plt.legend()
		plt.savefig('performance_svm_timings.png')
		plt.show()

	def raster_plot(self):
		activity = {}
		bins = np.arange(0, 1000, 10)
		for stim in self.d_stims:
			activity[stim] = [np.sort(np.concatenate([self.d_stims[stim][pres][i] for pres in self.d_stims[stim]])) for i in range(len(self.d_stims[stim][0]))]

		for stim in activity:
			raster = [np.histogram(clu, bins)[0] for clu in activity[stim]]
			plt.imshow(raster)
			plt.savefig('Output/Raster/raster_{}.png'.format(params.sound_names[stim]))
			plt.close()

		self.activity = activity

		return activity









'''
Extend tasks to tasks 3 and 4
Concatenate across dimensions to get large array
Make proper figures from it (svm performance model, save for each task
make an __add__ method to merge two recording together
add metadat to recording object ? So I can save it for later
'''

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
