import os
import numpy as np 
import matplotlib.pyplot as plt

from Custom import Mouse

mice_id = ['459', '461', '462', '463', '267', '268', '269']
mice = [Mouse('/home/user/share/gaia/Data/Behavior/Antonin/660{}'.format(i) for i in mice_id)]


for m in mice:
    mouse = Mouse(path='/home/user/share/gaia/Data/Behavior/Antonin/660{}'.format(m))
    mouse.summary(tag=['PC'], stim_freqs=np.geomspace(6e3, 16e3, 16), threshold=80, last=False)


