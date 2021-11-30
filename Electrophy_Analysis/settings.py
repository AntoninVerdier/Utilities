import os

class paths():
	def __init__(self):
		self.Ksdir = '/home/user/Documents/Antonin/DataEphys/To_analyze/08_11_2021_1'
		self.SoundInfo = '/home/user/Documents/Antonin/DataEphys/To_analyze/08_11_2021_1/SoundInfo.mat'
		self.Output = os.path.join(self.Ksdir, 'Output')

		if not os.path.exists(self.Output):
			os.makedirs(self.Output)

class params():
	def __init__(self):
		self.fs = 20000
		# self.pad_before = -0.5 * self.fs
		# self.pad_after = 1.0 * self.fs	

		self.task1 = [52, 53, 54, 56, 57, 59, 60, 62, 31, 32, 34, 35, 37, 38, 40, 41]
		self.task2 = [44, 45, 46, 47, 48, 49, 50, 51, 55, 58, 61, 33, 36, 39, 42, 43]
		self.task3 = [8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]

		self.task4 = [29, 26, 27, 28, 30, 25]

		self.y_task1 = []
		self.y_task3 = [0]*3*4*15 + [1]*3*4*15
		self.y_task4 = [0]*15 + [1]*5*15

		self.sound_names = ['10kHz_126Hz_45dB.wav', '10kHz_126Hz_50dB.wav', '10kHz_126Hz_55dB.wav',
							'10kHz_126Hz_60dB.wav', '10kHz_200Hz_45dB.wav', '10kHz_200Hz_50dB.wav',
							'10kHz_200Hz_55dB.wav', '10kHz_200Hz_60dB.wav', '10kHz_20Hz_45dB.wav',
							'10kHz_20Hz_50dB.wav', '10kHz_20Hz_55dB.wav', '10kHz_20Hz_60dB.wav',
							'10kHz_31Hz_45dB.wav', '10kHz_31Hz_50dB.wav', '10kHz_31Hz_55dB.wav',
							'10kHz_31Hz_60dB.wav', '10kHz_50Hz_45dB.wav', '10kHz_50Hz_50dB.wav',
							'10kHz_50Hz_55dB.wav', '10kHz_50Hz_60dB.wav', '10kHz_79Hz_45dB.wav',
							'10kHz_79Hz_50dB.wav', '10kHz_79Hz_55dB.wav', '10kHz_79Hz_60dB.wav',
							'500ms_6kHz.wav', '6kto16k_10step_70dB.wav', '6kto16k_2step_70dB.wav',
							'6kto16k_4step_70dB.wav', '6kto16k_6step_70dB.wav', '6kto16k_70dB.wav',
							'6kto16k_8step_70dB.wav', 'PT_10123Hz_500ms_70dB.wav',
							'PT_10807Hz_500ms_70dB.wav', 'PT_108Hz_500ms_70dB.wav',
							'PT_11537Hz_500ms_70dB.wav', 'PT_12317Hz_500ms_70dB.wav',
							'PT_126Hz_500ms_70dB.wav', 'PT_13150Hz_500ms_70dB.wav',
							'PT_14038Hz_500ms_70dB.wav', 'PT_147Hz_500ms_70dB.wav',
							'PT_14987Hz_500ms_70dB.wav', 'PT_16000Hz_500ms_70dB.wav',
							'PT_171Hz_500ms_70dB.wav', 'PT_200Hz_500ms_70dB.wav',
							'PT_20Hz_500ms_70dB.wav', 'PT_23Hz_500ms_70dB.wav',
							'PT_27Hz_500ms_70dB.wav', 'PT_31Hz_500ms_70dB.wav',
							'PT_36Hz_500ms_70dB.wav', 'PT_43Hz_500ms_70dB.wav',
							'PT_50Hz_500ms_70dB.wav', 'PT_58Hz_500ms_70dB.wav',
							'PT_6000Hz_500ms_70dB.wav', 'PT_6405Hz_500ms_70dB.wav',
							'PT_6838Hz_500ms_70dB.wav', 'PT_68Hz_500ms_70dB.wav',
							'PT_7300Hz_500ms_70dB.wav', 'PT_7793Hz_500ms_70dB.wav',
							'PT_79Hz_500ms_70dB.wav', 'PT_8320Hz_500ms_70dB.wav',
							'PT_8882Hz_500ms_70dB.wav', 'PT_92Hz_500ms_70dB.wav',
							'PT_9482Hz_500ms_70dB.wav']
		self.colors_1 = []

