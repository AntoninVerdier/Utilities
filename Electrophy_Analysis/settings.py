import os

class paths():
	def __init__(self):
		self.Ksdir = '/home/user/Documents/Antonin/DataEphys/Antonin_21102021_211021_191144/raw_amp'
		self.SoundInfo = '/home/user/Documents/Antonin/DataEphys/MouseAntonin_Rec1_20211021T19_12_11/SoundInfo.mat'
		self.digitalin = '/home/user/Documents/Antonin/DataEphys/Antonin_21102021_211021_191144/digitalin.dat'

class params():
	def __init__(self):
		self.fs = 20000
		self.pad_before = 0.5 * self.fs
		self.pad_after = 0.5 * self.fs	