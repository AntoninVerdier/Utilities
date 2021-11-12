import os

class paths():
	def __init__(self):
		self.Ksdir = '/home/user/Documents/Antonin/DataEphys/Antonin_08112021_2/Antonin_08112021__211108_183744/'
		self.SoundInfo = '/home/user/Documents/Antonin/DataEphys/20211108T18_37_57/SoundInfo.mat'
		self.digitalin = '/home/user/Documents/Antonin/DataEphys/Antonin_08112021_2/Antonin_08112021__211108_183744/digitalin.dat'
		self.Output = os.path.join(self.Ksdir, 'Output')

		if not os.path.exists(self.Output):
			os.makedirs(self.Output)

class params():
	def __init__(self):
		self.fs = 20000
		self.pad_before = 0.5 * self.fs
		self.pad_after = 1.0 * self.fs	
