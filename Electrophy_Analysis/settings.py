import os

class paths():
	def __init__(self):
		self.Ksdir = '/home/user/Documents/Antonin/DataEphys/Antonin_08112021_2/Antonin_08112021__211108_183744/'
		self.SoundInfo = '/home/user/Documents/Antonin/DataEphys/20211108T18_37_57/SoundInfo.mat'
		self.Output = os.path.join(self.Ksdir, 'Output')

		if not os.path.exists(self.Output):
			os.makedirs(self.Output)

class params():
	def __init__(self):
		self.fs = 20000
		self.pad_before = 0.0 * self.fs
		self.pad_after = 0.2 * self.fs	

		self.task1 = [52, 53, 54, 56, 57, 59, 60, 62, 31, 32, 34, 35, 37, 38, 40, 41]
		self.task2 = [44, 45, 46, 47, 48, 49, 50, 51, 55, 58, 61, 33, 36, 39, 42, 43]
		self.task3 = [8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
		self.task4 = [29, 26, 27, 28, 30, 25]
