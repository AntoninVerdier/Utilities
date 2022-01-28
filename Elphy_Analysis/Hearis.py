""" This file groups all low and high level analysis function for behavioural data
"""
import os
import pickle as pkl 
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
    p0 = [-1, np.mean(f), 0.01, 1] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, f, p, p0, method='lm', maxfev=1000000)
    perr = np.sqrt(np.diag(pcov))

    x = f
    y = sigmoid(x, *popt)

    d_y = np.max(np.abs(d_sigmoid(popt[1], *popt)))

    x1 = popt[1]
    y1 = sigmoid(popt[1], *popt)

    return x, y, d_y, x1, y1


def mean_psycoacoustic(mice, tag=['PC'], stim_freqs=np.geomspace(6e3, 16e3, 16), date=None, threshold=None, plot=False):
    """

    """
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    psykos = []

    data_to_save = {}
    for mouse in mice:
        _, probs = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, plot=False, date=date, threshold=threshold)
        if mouse.reversed:
            probs = [1 - p for p in probs]

        ax.plot(stim_freqs, probs, linewidth=2, c='gray', alpha=0.4)
        psykos.append(probs)

        data_to_save[mouse.ID] = probs
    
    psykos = np.array(psykos)
    np.save('psykos_panel_a.npy', psykos)
    mean_psykos = np.mean(psykos, axis=0)

    data_to_save['mean_psykos'] = mean_psykos
    data_to_save['stim_freqs'] = stim_freqs


    x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, mean_psykos)
    ax.plot(x, y, color='red')
    print(d_y)


    ax.errorbar(stim_freqs, mean_psykos, yerr=None, color='forestgreen', linewidth=2, markersize=6, marker='o')
    plt.tight_layout()

    plt.savefig('../Output/Psychoacoustic.svg')
    plt.show()

    return data_to_save
def mean_psycoacoustic_noise(mice, tag='PC', stim_freqs=np.geomspace(6e3, 16e3, 16), date=None, threshold=None, plot=False):
    """

    """
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    no_noise_p = []
    colors = ['#faa307', '#f48c06', '#e85d04', '#dc2f02']
    labels = ['45dB', '50dB', '55dB', '60dB']

    dys=[]
    errs=[]
    for i, noise_level in enumerate(tag):
        print(noise_level)
        psykos = []
        curr_d = []
        for mouse in mice:
            _, probs = mouse.psychoacoustic(tag=noise_level, stim_freqs=stim_freqs, plot=False, date=date, threshold=threshold)
            if mouse.reversed:
                probs = [1 - p for p in probs]

            psykos.append(probs)

            #x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, probs[6:])
            #ax[0].plot(x, y, color='black', alpha=0.4, linewidth=4, markersize=6, marker='o', label=labels[i])

            #curr_d.append(d_y)


        # dys.append(np.mean(curr_d))
        # errs.append(np.std(curr_d))

        psykos = np.array(psykos)
        mean_psykos = np.mean(psykos, axis=0)
        no_noise_p.append(mean_psykos)


        ax.plot(stim_freqs, mean_psykos[6:], color=colors[i], linewidth=2, markersize=5, marker='o', label=labels[i])
        
        #x, y, d_y, x1, y1, perr = fit_sigmoid(stim_freqs, no_noise_p[:6])

           
    no_noise_p = np.mean(np.array(no_noise_p), axis=0)
    
    ax.plot(stim_freqs, no_noise_p[:6], linewidth=2, c='gray', alpha=0.4)    
    #ax[0].plot(x, y, label='fit', c='blue')
    ax.legend()
    errs = [0] + errs
    dys = [0] + dys
    

    #ax[1].bar(x=['0dB', '45dB', '50dB', '55dB', '60dB'] , height=dys, color='royalblue', yerr=errs)

    plt.tight_layout()
    plt.savefig('../Output/Psychoacoustic.svg')
    plt.show()

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

def all_perfs(mice, plot=False, blank=False):

    fig, axs = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle('Perfs\' follow up - AVEC GAP REMOVAL')

    for i, mouse in enumerate(mice):
        corr, d = mouse.perf(plot=plot, blank=blank)
        axs[i%4, i//4].yaxis.grid(c='gainsboro', ls='--')
        axs[i%4, i//4].plot(d, corr, 'o-')
        axs[i%4, i//4].set_ylim(0, 100)
        axs[i%4, i//4].set_yticks(np.linspace(0, 100, 11))
        axs[i%4, i//4].set_title(label='Perf evolution of {}'.format(mouse.ID),
                                fontsize=10,
                                fontstyle='italic')
    plt.savefig(os.path.join(mouse.output, 'perf_followup_PC_gaps.svg'))
    plt.show()

def all_perfs_once(mice):

    corrs, dates, tags = [], [], []
    means = {}
    tagss = {}

    all_d = []
    for i, mouse in enumerate(mice):
        corr, d = mouse.perf(rmcons=True)
        all_d += list(d)
    all_d = np.sort(list(set(all_d)))

    for d in all_d:
        means[d] = []
        tagss[d] = []

    data_to_save = []
    for i, mouse in enumerate(mice):
        corr, d = mouse.perf(rmcons=True)
        dates.append(d)
        corrs.append(corr)
        tags.append(t)

        for i, j in enumerate(d):
            means[j].append(corr[i])
            tagss[j].append(t[i])

        colors = ['red' if t[i] == 'DIS_' else 'lightsteelblue' for i in range(len(t))]
        colors = [j if not ((corr[i] >= 80) and (t[i] == 'PC_')) else '#0000ffff' for i, j in enumerate(colors)]

        plt.grid(c='gainsboro', ls='--', axis='y')
        plt.plot(d, corr, color='slategrey', alpha=1)
        plt.scatter(d, corr, c=colors)

        data_to_save.append([d, corr, colors])

    average_perf = []
    average_dates = []
    
    for k in means:
        if tagss[k].count(tagss[k][0]) == len(tagss[k]):
            day_mean = np.mean(means[k])
            average_perf.append(day_mean)
            average_dates.append(k)


    plt.plot(average_dates, average_perf, color='red')


    plt.ylim(0, 100)
    plt.yticks(np.linspace(0, 100, 11))
    plt.savefig(os.path.join(mouse.output, 'perf_followup_PC_gaps.svg'))
    plt.show()

    return average_dates, average_perf, data_to_save 

def all_psycho(mice, tag=['PC'], stim_freqs=np.geomspace(6e3, 16e3, 16), threshold=None):
    fig, axs = plt.subplots(2, 4, figsize=(10, 5), dpi=120)

    dys = []
    shift_errors = []
    for i, mouse in enumerate(mice):
        f, p = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, threshold=threshold, plot=False)

        # x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, p)

        # shift_errors.append(x1 - np.median(stim_freqs))

        #axs[i//4, i%4].plot(x, y, label='fit', c='darkorange')
        #axs[i//4, i%4].plot(x, d_y*x + (y1 - d_y * x1))

        # dys.append(d_y)

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

    # dys = np.array(dys) * (stim_freqs[8] - stim_freqs[7])
    # shift_errors = np.array(shift_errors) / (stim_freqs[8] - stim_freqs[7])

    # axs[1, 3].bar(['Slope', 'Shift'], [np.mean(np.abs(dys)), np.mean(np.abs(shift_errors))], 
    #             yerr=[np.std(np.abs(dys)), np.std(np.abs(dys))], 
    #             align='center', alpha=0.5, ecolor=['blue', 'darkorange'], capsize=10)
    # axs[1, 3].set_ylabel('Stimulus.', fontsize=11)


    plt.tight_layout()
    plt.savefig(os.path.join(mouse.output, 'psychoacoustic.svg'))
    plt.show()

def noise_psycho(mice, tag=['PCAMN45'], stim_freqs=np.geomspace(20, 200, 6), threshold=85):
    fig, axs = plt.subplots(4, 2, figsize=(10, 20))

    slopes = {}
    for i, mouse in enumerate(mice):
        slopes[mouse.ID] = []
        f, p = mouse.psychoacoustic(tag=tag, stim_freqs=stim_freqs, threshold=threshold, plot=False)

        axs[i%4, i//4].set_xscale('log')
        axs[i%4, i//4].plot(stim_freqs, p[:6], 'o-', markersize=2, label='10kHz')
        axs[i%4, i//4].axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
        # axs[i%4, i//4].set_title(label='Psycho curve of {}'.format(mouse.ID),
        #                       fontsize=10,
        #                       fontstyle='italic')
        axs[i%4, i//4].set_xlabel('Hz')
        axs[i%4, i//4].xaxis.set_label_coords(1.05, -0.05)


        axs[i%4, i//4].set_ylabel('Lick Prob.')
        axs[i%4, i//4].spines["top"].set_visible(False)
        axs[i%4, i//4].spines["right"].set_visible(False)

        for j, noise in enumerate(tag):
            f, p = mouse.psychoacoustic(tag=[noise], stim_freqs=stim_freqs, threshold=threshold, plot=False)
            slopes[mouse.ID].append(np.abs((p[-1]-p[0])/(16e3 - 6e3)))
            axs[i%4, i//4].plot(stim_freqs, p[6:], 'o-', markersize=2, label='10kHz+WN_{}dB'.format(noise[-3:]))

            #axs[i%4, i//4].legend()

    # for m in slopes:
    #     axs[3, 1].plot(['0', '45', '50', '55', '60'], slopes[m], c='gray', alpha=.5)

    # all_slopes = [np.mean([slopes[m][i] for m in slopes]) for i in range(len(tag)+1)]
    # axs[3, 1].plot(['0', '45', '50', '55', '60'], all_slopes, c='red')

    # axs[3, 1].set_title(label='Slope btw extremes'.format(mouse.ID),
    #                           fontsize=10,
    #                           fontstyle='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(mouse.output, 'psycho_curves_85.svg'))
    plt.show()

def all_score_by_task(mice, names=None):
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))

    for i, mouse in enumerate(mice):
        score = mouse.score_by_task(names=names, plot=False)
        axs[i%3, i//3].bar(x=names, height=score)
        axs[i%3, i//3].set_title('{}'.format(mouse.ID))
        axs[i%3, i//3].set_axisbelow(True)
        axs[i%3, i//3].grid(c='gainsboro', linestyle='dashed', axis='y')


    plt.suptitle('% Correct depending on stimulus')
    plt.savefig(os.path.join(mouse.output, 'score_by_task_02062021.svg'))

    plt.show()

def histogram_slopes(mice, tag='PC', stim_freqs=np.geomspace(6e3, 16e3, 16), date=None, threshold=None, plot=False):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111)

    labels = ['45dB', '50dB', '55dB', '60dB']
    dys = {}    
    for noise in tag:
        dys[noise] = []
    
    for i, noise_level in enumerate(tag):
        for mouse in mice:
            _, probs = mouse.psychoacoustic(tag=noise_level, stim_freqs=stim_freqs, plot=False, date=date, threshold=threshold)
            if mouse.reversed:
                probs = [1 - p for p in probs]


            x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, probs[6:])
            dys[noise_level].append(d_y)
            
            #ax.set_xscale('log')
            #ax.plot(stim_freqs, probs[6:], color='red')
            #ax.plot(x, y, color='gray')

        #ax.bar(['45db'], height=[np.mean(dys)], yerr=np.std(dys))

    dmeans = []
    xs = []
    for d in dys:
        dmeans.append(np.mean(dys[d]))
        xs.append(d)



    ax.bar(x=xs, height=dmeans, yerr=np.std(dmeans))
    plt.tight_layout()
    plt.savefig('../Output/Psychoacoustic.svg')
    plt.show()

def histogram_slopes_PC(mice, tag='PC', stim_freqs=np.geomspace(6e3, 16e3, 16), date=None, threshold=None, plot=False):
    fig = plt.figure(figsize=(3, 4))
    ax = plt.subplot(111)

    labels = ['Task 1', 'Task 2']
    dys  = {'Task 1': [], 'Task 2': []}

    
    for i, task in enumerate(tag):
        all_probes = []
        if not i:
            threshold = 80
            stim_freqs=np.geomspace(6e3, 16e3, 16)
        else:
            threshold = 65
            stim_freqs=np.geomspace(20, 200, 16)

        for mouse in mice:
            _, probs = mouse.psychoacoustic(tag=task, stim_freqs=stim_freqs, plot=False, date=date, threshold=threshold)
            if mouse.reversed:
                probs = [1 - p for p in probs]

            all_probes.append(probs)

            if not i:
                x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, probs)
                dys[labels[i]].append(d_y)
                # ax.plot(stim_freqs, probs, color='orange')
                # ax.plot(x, y, color='black')
            else:
                x, y, d_y, x1, y1 = fit_sigmoid(stim_freqs, probs)
                dys[labels[i]].append(d_y) # to account for difference in scales
                # ax.plot(stim_freqs, probs, color='red')
                # ax.plot(x, y, color='blue')
            
            #ax.set_xscale('log')

        #ax.bar(['45db'], height=[np.mean(dys)], yerr=np.std(dys))
    dmeans = [np.mean(dys[k]) for k in dys]

    edgecolors = ['#0000ffff', '#228b22ff']
    colors = ['#0000ff19', '#228b2219']



    ax.bar(x=labels, height=dmeans, color=colors, edgecolor=edgecolors)

    coord = [0, 1]
    for i, c in enumerate(coord):
        if not i:
            ax.scatter([c]*7, dys[labels[i]], color='#0000ff23')
        else:
            ax.scatter([c]*7, dys[labels[i]], color='#228b2223')



    plt.tight_layout()
    plt.savefig('../Output/Psychoacoustic.svg')
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

# mice_id = batch.id_second_batch
# mice = [Mouse(path='/home/user/share/gaia/Data/Behavior/Antonin/{}'.format(i), tag=['DISP'], collab=False) for i in mice_id]
# #all_psycho(mice, tag=['DISCS46810'], threshold=0, stim_freqs=np.linspace(0, 5))
# all_perfs(mice)

# Collab perf
if __name__ == '__main__':
    
    mice_id = batch.id_first_dlp
    mice = [Mouse(path='/home/anverdie/share/gaia/Data/Optogenetic/DLP/{}'.format(i), tag=['SPA'], dlp=True, collab=False, rmgaps='Antonin', verbose=True, linkday=True) for i in mice_id]

    all_weights(mice)
    all_perfs(mice, blank=True)
    all_psycho(mice, tag=['PC'], stim_freqs=np.geomspace(4e3, 16e3, 16), threshold=50)
# for mouse in mice:p
#     files = mouse.elphy
#     files = [f for f in files]
#     tasks = np.array([item for f in files for item in f.tr_type])
#     corr = np.array([item for f in files for item in f.tr_corr])
#     types = np.array([item for f in files for item in f.tr_corr])
#     licks = [not c if 1 <= tasks[i] <= 8 else c for i, c in enumerate(corr)]
#     #licks = [not c if 1 <= tasks[i] <= 8 else c for i, c in enumerate(corr)]

    
#     w = 50
#     tasks = [4 if t == 1 else t for t in tasks ]
#     tasks = [1 if t == 3 else t for t in tasks ]
#     rewards = [0 if t == 2 else t for t in tasks]


#     rewarded_trials = np.convolve([c for i, c in enumerate(corr) if rewards[i] == 1], np.ones(w))
#     non_rewarded_trials = np.convolve([c for i, c in enumerate(corr) if rewards[i] == 0], np.ones(w))
#     blank_trials = np.convolve([c for i, c in enumerate(corr) if rewards[i] == 4], np.ones(w))
    
#     plt.plot(rewarded_trials, c='blue')
#     plt.plot(non_rewarded_trials, c='red')
#     plt.plot(blank_trials, c='green')
#     plt.show()
# pkl.dump(mice, open('Panel_D_mice.pkl', 'wb'))
# data_to_save = mean_psycoacoustic(mice, 'PC', stim_freqs=np.geomspace(20, 200, 16), threshold=65)
# pkl.dump(data_to_save, open('Panel_D_data.pkl', 'wb'))

#mean_psycoacoustic_noise(mice, tag=['PCAMN45', 'PCAMN50', 'PCAMN55', 'PCAMN60'], stim_freqs=np.geomspace(20, 200, 6), threshold=65)
#all_perfs(mice, tag=['PC'])
#all_perfs_once(mice)
# all_score_by_task(mice, names=['Blank', 'NOGO_50ms', 'GO_150ms', 'L_Blank', 'NOGOL_50ms', 'GOL_150ms'])




