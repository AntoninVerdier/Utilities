""" This file groups all low and high level analysis function for behavioural data
"""
import os
import numpy as np

import matplotlib.pyplot as plt

from Custom import Mouse

mice_id = ['459', '461', '462', '463', '267', '268', '269']
#mice_id = ['459', '462']
mice = [Mouse('/home/pouple/PhD/Data/660{}'.format(i)) for i in mice_id]

def psycho_week():
	pass

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

def all_psycho(mice, tag=['PC'], stim_freqs=np.geomspace(6e3, 16e3, 16), threshold=80):
	fig, axs = plt.subplots(4, 2, figsize=(10, 20))
	fig.suptitle('Psycho curves - AVEC GAP REMOVAL - {} %'.format(threshold))

	for i, mouse in enumerate(mice):
		f, p = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, threshold=threshold, plot=False)

		axs[i%4, i//4].set_xscale('log')
		axs[i%4, i//4].plot(stim_freqs, p, 'o-', markersize=2)
		axs[i%4, i//4].axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
		axs[i%4, i//4].set_title(label='Psycho curve of {}'.format(mouse.ID),
								fontsize=10,
								fontstyle='italic')

	plt.savefig(os.path.join(mouse.output, 'psycho_curves_gaps.svg'))
	plt.show()
all_perfs(mice)
all_psycho(mice)


