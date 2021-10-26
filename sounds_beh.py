import QuickSound.Sound as Sound
import numpy as np

samplerate = 192000
amplitude = 70
path = ''

# Generate sounds for frequency discrimnation
# for f in np.geomspace(6e3, 16e3, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.pure_tone(f, duration=4000)
# 	pure.save_wav(name='PT_{0}Hz_{1}ms_{2}dB'.format(int(f), 4000, amplitude), path=path)

# Generate sounds for amplitude modulation
# for a in np.geomspace(20, 200, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.amplitude_modulation(10e3, a, duration=4000)
# 	pure.save_wav(name='AM_{0}Hz_{1}ms_{2}dB'.format(int(a), 4000, amplitude), path=path)

# Generates sounds for amplitude modulation and noise
# for na in [45, 50, 55, 60]:
# 	for a in np.geomspace(20, 200, 6):
# 		pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 		pure.amplitude_modulation(10e3, a, duration=4000)
# 		noise = Sound(samplerate=samplerate, amplitude=na)
# 		noise.noise(duration=4000)
# 		final = pure * noise
# 		final.save_wav(name='AMN_{0}Hz_{1}ms_{2}dB_noise_{3}dB'.format(int(a), 4000, amplitude, na), path=path)

# Generates a chirp
# chirp = Sound(samplerate=samplerate, amplitude=amplitude)
# chirp.freq_modulation(6e3, 16e3, duration=4000)
# chirp.save_wav(name='Chirp_6000Hz_16000Hz_4000ms_70dB', path=path)

# Generates steps
# for s in [2, 4, 6, 8, 10]:
# 	step = Sound(samplerate=samplerate, amplitude=amplitude)
# 	step.steps(6e3, 16e3, s, duration=4000)
# 	step.save_wav(name='Steps_{}_6000Hz_16000Hz_4000ms_70dB'.format(s), path=path)
