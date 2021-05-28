""" This file groups all low and high level analysis function for behavioural data
"""
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Custom import Mouse
import settings

batch = settings.Batch()

def psycho_week():
	pass

def sigmoid(x, L ,x0, k, b):
	y = L / (1 + np.exp(-k*(x-x0)))+b
	return y

def d_sigmoid(x, L, x0, k, b):

	y = (L*k*np.exp(-k*(x-x0)))/((1+np.exp(-k*(x-x0)))**2)
	return y

def fit_sigmoid(f, p):
	p0 = [1, np.mean(f), 0.005, 0] # this is an mandatory initial guess
	popt, pcov = curve_fit(sigmoid, f, p, p0, method='lm')

	x = np.geomspace(6e3, 16e3, 1000)
	y = sigmoid(x, *popt)

	d_y = d_sigmoid(popt[1], *popt)

	x1 = popt[1]
	y1 = sigmoid(popt[1], *popt)

	return x, y, d_y, x1, y1


def mean_psycoacoustic(mice, tag='PC', stim_freqs=np.geomspace(6e3, 16e3, 16), date=None, threshold=None, plot=True):
	"""

	"""
	psykos = []
	for mouse in mice:
		_, probs = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, plot=True, date=date, threshold=threshold)
		psykos.append(probs)
	psykos = np.array(psykos)
	psykos = np.mean(psykos, axis=0)
	mouse.psychoacoustic_fig(_, psykos, stim_freqs)

def all_weights(mice, plot=False):

	fig, axs = plt.subplots(4, 2, figsize=(20, 20))
	fig.suptitle('Weight\'s follow up')

	weights = []
	for i, mouse in enumerate(mice):
		w, d = mouse.weight(plot=plot)

		axs[i%4, i//4].plot(d, w, 'ro-')
		axs[i%4, i//4].grid(c='gainsboro')

		mults = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
		cs = ['chartreuse', 'chartreuse', 'gold', 'gold', 'firebrick', 'firebrick']

		for mult, c in zip(mults, cs):
			axs[i%4, i//4].axhline(y=w[0]+w[0]*mult, c=c, ls='--', linewidth=1)

		axs[i%4, i//4].set_ylim([w[0]-5, w[0]+5])
		axs[i%4, i//4].set_title(label='Weight evolution of {}'.format(mouse.ID),
					 fontsize=10,
					 fontstyle='italic')

	plt.savefig(os.path.join(mouse.output, 'weights_followup.svg'))

def all_perfs(mice, tag=['PC'], plot=False):

	fig, axs = plt.subplots(4, 2, figsize=(20, 20))
	fig.suptitle('Perfs\' follow up - AVEC GAP REMOVAL')

	for i, mouse in enumerate(mice):
		corr, d = mouse.perf(tag=tag, plot=plot)

		axs[i%4, i//4].yaxis.grid(c='gainsboro', ls='--')
		axs[i%4, i//4].plot(d, corr, 'o-')
		axs[i%4, i//4].set_ylim(0, 100)
		axs[i%4, i//4].set_yticks(np.linspace(0, 100, 11))
		axs[i%4, i//4].set_title(label='Perf evolution of {}'.format(mouse.ID),
								fontsize=10,
								fontstyle='italic')
	plt.savefig(os.path.join(mouse.output, 'perf_followup_PC_gaps.svg'))
	plt.show()

def all_psycho(mice, tag=['PC'], stim_freqs=np.geomspace(6e3, 16e3, 16), threshold=None):
	fig, axs = plt.subplots(2, 4, figsize=(10, 5), dpi=120)

	dys = []
	shift_errors = []
	for i, mouse in enumerate(mice):
		f, p = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, threshold=threshold, plot=False)

		x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, p)

		shift_errors.append(x1 - np.median(stim_freqs))

		axs[i//4, i%4].plot(x, y, label='fit', c='darkorange')
		#axs[i//4, i%4].plot(x, d_y*x + (y1 - d_y * x1))

		dys.append(d_y)

		axs[i//4, i%4].set_xscale('log')
		axs[i//4, i%4].set_ylim(0, 1)
		axs[i//4, i%4].plot(stim_freqs, p, 'o-', markersize=3, c='royalblue', label='PC_6_16')
		axs[i//4, i%4].axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='black', ls='-', linewidth=1)
		axs[i//4, i%4].set_title(label='{}'.format(mouse.ID),
								fontsize=10,
								fontstyle='italic')
		
		axs[i//4, i%4].set_xlabel('Sound Freq (kHz)', fontsize=11)
		axs[i//4, i%4].set_ylabel('Go prob', fontsize=11)
		axs[i//4, i%4].spines["top"].set_visible(False)
		axs[i//4, i%4].spines["right"].set_visible(False)
		axs[i//4, i%4].legend(fontsize=7)

	dys = np.array(dys) * (stim_freqs[8] - stim_freqs[7])
	shift_errors = np.array(shift_errors) / (stim_freqs[8] - stim_freqs[7])

	axs[1, 3].bar(['Slope', 'Shift'], [np.mean(np.abs(dys)), np.mean(np.abs(shift_errors))], 
				  yerr=[np.std(np.abs(dys)), np.std(np.abs(dys))], 
				  align='center', alpha=0.5, ecolor=['blue', 'darkorange'], capsize=10)
	axs[1, 3].set_ylabel('Stimulus.', fontsize=11)


	plt.tight_layout()
	plt.savefig(os.path.join(mouse.output, 'psychoacoustic.svg'))
	plt.show()

def noise_psycho(mice, tag=['PCAMN45'], stim_freqs=np.geomspace(20, 200, 6), threshold=85):
	fig, axs = plt.subplots(4, 2, figsize=(10, 20))

	slopes = {}
	for i, mouse in enumerate(mice):
		slopes[mouse.ID] = []
		f, p = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, threshold=threshold, plot=False)

		slopes[mouse.ID].append(np.abs((p[-1]-p[0])/(16e3 - 6e3)))
		axs[i%4, i//4].set_xscale('log')
		axs[i%4, i//4].plot(stim_freqs, p[:6], 'o-', markersize=2, label='10kHz')
		axs[i%4, i//4].axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
		# axs[i%4, i//4].set_title(label='Psycho curve of {}'.format(mouse.ID),
		# 						fontsize=10,
		# 						fontstyle='italic')
		axs[i%4, i//4].set_xlabel('Hz')
		axs[i%4, i//4].xaxis.set_label_coords(1.05, -0.05)


		axs[i%4, i//4].set_ylabel('Lick Prob.')
		axs[i%4, i//4].spines["top"].set_visible(False)
		axs[i%4, i//4].spines["right"].set_visible(False)

		for j, noise in enumerate(tag):
			f, p = mouse.psychoacoustic(tag=[noise], stim_freqs=stim_freqs, threshold=threshold, plot=False)
			slopes[mouse.ID].append(np.abs((p[-1]-p[0])/(16e3 - 6e3)))
			axs[i%4, i//4].plot(stim_freqs, p[6:], 'o-', markersize=2, label='10kHz+WN_{}dB'.format(noise[-3:]))

			axs[i%4, i//4].legend()

	for m in slopes:
		axs[3, 1].plot(['0', '45', '50', '55', '60'], slopes[m], c='gray', alpha=.5)

	all_slopes = [np.mean([slopes[m][i] for m in slopes]) for i in range(len(tag)+1)]
	axs[3, 1].plot(['0', '45', '50', '55', '60'], all_slopes, c='red')

	# axs[3, 1].set_title(label='Slope btw extremes'.format(mouse.ID),
	# 							fontsize=10,
	# 							fontstyle='italic')

	plt.tight_layout()
	plt.savefig(os.path.join(mouse.output, 'psycho_curves_85.svg'))
	plt.show()



#all_weights(mice)
# all_psycho(mice, tag=['PCAMN45'], threshold=40, stim_freqs=np.geomspace(20, 200, 6))
# all_psycho(mice, tag=['PCAMN50'], threshold=40, stim_freqs=np.geomspace(20, 200, 6))
# all_psycho(mice, tag=['PCAMN60'], threshold=40, stim_freqs=np.geomspace(20, 200, 6))
# all_perfs(mice,tag=['PCAMN45'], plot=False)
# all_perfs(mice,tag=['PCAMN50'], plot=False)
# all_perfs(mice,tag=['PCAMN60'], plot=False)

#all_psycho(mice, tag=['PCAM'], threshold=80)

#noise_psycho(mice, tag=['PCAMN45_', 'PCAMN50_', 'PCAMN55_', 'PCAMN60_'], threshold=60)
#mice_id = ['268', '269']
#mice_id = ['459','462', '269']
#mice_id = ['463', '268']

mice_id = batch.id_first_collab
mice = [Mouse(path='/home/user/share/gaia/Data/Behavior/Antonin/{}'.format(i), tag=['DISOA'], collab=True) for i in mice_id]
#all_psycho(mice, tag=['DISCS46810'], threshold=0, stim_freqs=np.linspace(0, 5))

all_perfs(mice)

#all_psycho(mice, tag=['PC'], threshold=80)
#all_psycho(mice, tag=['DISCS4'], stim_freqs=np.arange(1, 6), threshold=50)

#mean_psycoacoustic(mice)

# psycho = {}
# for mouse in mice:
# 	f, p = mouse.psychoacoustic(tag=['PC_'], threshold=80, stim_freqs=np.geomspace(6e3, 16e3, 16))
# 	psycho[mouse.ID] = p

# pickle.dump(psycho, open('frequency_discrimination_data.pkl', 'wb'))


