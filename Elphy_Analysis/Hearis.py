""" This file groups all low and high level analysis function for behavioural data
"""
import os
import numpy as np

from Custom import Mouse

mice_id = ['461', '462', '463', '268']#, '267', '268', '269']
mice = [Mouse('/home/user/share/gaia/Data/Behavior/Antonin/660{}'.format(i)) for i in mice_id]

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
	print(psykos)
	psykos = np.mean(psykos, axis=0)
	print(psykos)
	mouse.psychoacoustic_fig(_, psykos, stim_freqs)
	


mean_psycoacoustic(mice, tag='PC', threshold=80)