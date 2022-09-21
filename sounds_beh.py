import os
import QuickSound.Sound as Sound
import numpy as np

samplerate = 192000
amplitude = 70
path = 'Sounds_Hearlight_extended'
duration = 500

if not os.path.exists(path):
	os.makedirs(path)

# Generate sounds for frequency discrimnation
# for f in np.geomspace(6e3, 16e3, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.pure_tone(f, duration=duration)
# 	pure.save_wav(name='PT_{0}Hz_{1}ms_{2}dB'.format(int(f), duration, amplitude), path=path)

# Generate sounds for amplitude modulation
# for a in np.geomspace(20, 200, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.amplitude_modulation(10e3, a, duration=duration)
# 	pure.save_wav(name='AM_{0}Hz_{1}ms_{2}dB'.format(int(a), duration, amplitude), path=path)

# Generates sounds for amplitude modulation and noise
# for na in [45, 50, 55, 60]:
# 	for a in np.geomspace(20, 200, 6):
# 		pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 		pure.amplitude_modulation(10e3, a, duration=duration)
# 		noise = Sound(samplerate=samplerate, amplitude=na)
# 		noise.noise(duration=duration)
# 		final = pure * noise
# 		final.save_wav(name='AMN_{0}Hz_{1}ms_{2}dB_noise_{3}dB'.format(int(a), duration, amplitude, na), path=path)

# Generates a chirp
# chirp = Sound(samplerate=samplerate, amplitude=amplitude)
# chirp.freq_modulation(6e3, 16e3, duration=duration)
# chirp.save_wav(name='Chirp_6000Hz_16000Hz_{}ms_70dB'.format(duration), path=path)

# Generates steps
# for s in [2, 4, 6, 8, 10]:
# 	step = Sound(samplerate=samplerate, amplitude=amplitude)
# 	step.steps(6e3, 16e3, s, duration=duration)
# 	step.save_wav(name='Steps_{}_6000Hz_16000Hz_{}ms_70dB'.format(s, duration), path=path)

# Generate harmonic 
struc = [(4e3,), (4e3, 20e3), (4e3, 12e3, 20e3), (4e3, 8e3, 12e3, 20e3), (4e3, 8e3, 12e3, 16e3, 20e3), (4e3, 12e3, 16e3, 20e3)]
names = ['1', '15', '135', '1235', '12345', '1345']
for i, h in enumerate(struc):
	fmul = Sound(samplerate=samplerate, amplitude=65)
	fmul.multi_freqs(h, duration=duration)
	fmul.save_wav(name='Harmonics_{}_4kHz_65dB'.format(names[i]), path=path, bit16=True)
	
# # Generate harmonic psychophysic
volumes = np.linspace(15, 65, 16)
for v in volumes:
	fmul = Sound(samplerate=samplerate, amplitude=amplitude)
	fmul.multi_freqs((4e3, 12e3, 16e3, 20e3), duration=duration)
	fmulvol = Sound(samplerate=samplerate, amplitude=v)
	fmulvol.pure_tone(8e3, duration=duration)
	final = fmul * fmulvol
	final.save_wav(name='Harmonics_1345_4kHz_65dB_{}'.format(int(v*100)), path=path, bit16=True)

# Generate toeplitz reg 
# path = 'toeplitz6-16'
# for f in np.geomspace(6e3, 16e3, 25):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.pure_tone(f, duration=duration)
# 	pure.save_wav(name='{}'.format(int(f), duration, amplitude), path=path)

# Colored noise
# for f in np.linspace(1e3, 16e3, 16):
# 	color = Sound(samplerate=samplerate, amplitude=amplitude)
# 	color.colored_noise(duration=duration, type='pink', freqs=[f - 0.1*f, f+0.1*f])
# 	#color.noise(duration=duration)
# 	color.save_wav(name='PN_{}'.format(int(f)), path=path)



