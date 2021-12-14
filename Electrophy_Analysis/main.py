import os
import umap
import scipy.io as sio
import numpy as np
import h5py

import matplotlib
from rich import print
from rich.progress import track
from rich.traceback import install 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, RepeatedKFold, cross_val_predict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle



import settings as s
from Recording import Recording

paths = s.paths()
params = s.params()

def compute_svm(X, y):

    scores = []
    X, y = shuffle(X, y)
    clf = svm.SVC(kernel='linear')
    scores = cross_val_score(clf, X, y, cv=5)


    ########## Need to make sure that data is shuffled
    #scores = cross_val_score(clf, X, y, cv=5)
    return scores

def svm_preformance(rec):
    for i, t in enumerate([params.task1, params.task2, params.task3, params.task4]):
        scores = []
        for p in track(np.arange(0, 1000, 50), description='Compute SVM for each task ...'):
            pop_vectors = rec.complete_vectors(0, p)

            X = np.array([pop_vectors[stim][p] for stim in t for p in pop_vectors[stim]])
            if i < 2:
                y = np.array([0 if i < 8 else 1 for i, stim in enumerate(t) for p in pop_vectors[stim]])
            elif i == 2:
                y = params.y_task3
            elif i == 3:
                y = params.y_task4

            score = compute_svm(X, y)
            scores.append([np.mean(score), np.std(score)])

        scores = np.array(scores).reshape(-1, 2)

        plt.errorbar(np.arange(0, 1000, 50), scores[:, 0], label='Task {}'.format(i + 1))

    plt.legend()
    plt.savefig('performance_svm_population.png')
    plt.show()

def psychocurve(rec, p=1000, timebin=10):
    for i, t in enumerate([params.task1, params.task2]):
        pop_vectors = rec.get_complete_vectors(0, p, timebin=timebin)

        X = np.array([pop_vectors[stim][p] for stim in t for p in pop_vectors[stim]])
        y = np.array([0 if i < 8 else 1 for i, stim in enumerate(t) for p in pop_vectors[stim]])
        true_classes = np.array([i for i, stim in enumerate(t) for p in pop_vectors[stim]]) + 1

        #X, y, true_classes = shuffle(X, y, true_classes)
        psycos = []
        for train_index, test_index in RepeatedKFold(n_splits=5, n_repeats=20).split(X):
            clf = svm.SVC(kernel='linear')
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tc_train, tc_test = true_classes[train_index], true_classes[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            correct = np.logical_not(np.logical_xor(y_pred, y_test))
            counting_vec = list(correct * tc_test)

            for i in np.unique(true_classes):
                try:
                    if i < 9:
                        psycos.append(counting_vec.count(i)/list(tc_test).count(i))
                    else:
                        psycos.append(1 - counting_vec.count(i)/list(tc_test).count(i))
                except ZeroDivisionError:
                    psycos.append(0)

        psycos = np.array(psycos).reshape(100, 16)
        psycos = np.mean(psycos, axis=0)

        plt.plot(np.geomspace(20, 200, 16), psycos, color='forestgreen', linewidth=2, markersize=6, marker='o')
        plt.xscale('log')
        plt.savefig('Output/TimeAverage/Task{}_finer_{}_{}ms'.format(i, p, timebin), dpi=300)
        
        plt.close()


recs = []
main_folder = '/home/anverdie/Documents/Electrophy/To_analyze'
for folder in os.listdir(main_folder):
    cp = os.path.join(main_folder, folder)
    print('Analyzing {} ...'.format(folder))
    rec = Recording(cp, os.path.join(cp, 'SoundInfo.mat'), name=folder)
    rec.select_data_quality(quality='good')
    rec.ttl_alignment(multi=False)
    recs.append(rec)


rec = np.sum(recs)
#rec.raster_plot()
#rec.raster_plot()

#svm_preformance(rec)
# for j in track(range(40, 80, 5)):
#     for i in np.arange(10, j + 5, 5):
psychocurve(rec)


pop_vectors = rec.get_population_vectors(0, 1000)

cmap = matplotlib.cm.get_cmap('hsv')
colors = []
for stim in [s for s in np.unique(rec.s_vector) for p in pop_vectors[s]]:
    if stim in params.task1:
        colors.append(cmap(0.1)) # Orange
    elif stim in params.task2:
        colors.append(cmap(0.3)) # vert
    elif stim in params.task3:
        colors.append(cmap(0.5)) # Blue
    elif stim in params.task4:
        colors.append(cmap(0.7))  #ble dark
    else:
        colors.append(cmap(0.9)) # Pink



# stims = [s for s in np.unique(rec.s_vector) for p in pop_vectors[s]]
# colors, values = [], []
# for stim in stims:
#   if stim in params.task1:
#       cmap = matplotlib.cm.get_cmap('Blues')
#       value = np.where(stim == params.task1)[0][0]/len(params.task1)
#   elif stim in params.task2:
#       cmap = matplotlib.cm.get_cmap('Reds')
#       value = np.where(stim == params.task2)[0][0]/len(params.task2)
#   elif stim in params.task3:
#       cmap = matplotlib.cm.get_cmap('Greens')
#       value = np.where(stim == params.task3)[0][0]/len(params.task3)
#   elif stim in params.task4:
#       cmap = matplotlib.cm.get_cmap('Greys')
#       value = np.where(stim == params.task4)[0][0]/len(params.task4)
#   else:
#       value = (1, 1, 1, 1)

#   colors.append(cmap(value))
#   print(len(colors))


for time in track(range(50, 1050, 50)):
    pop_vectors = rec.get_population_vectors(0, time)
    X = np.array([pop_vectors[stim][p] for stim in np.unique(rec.s_vector) for p in pop_vectors[stim]])

    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(X)

    plt.scatter(Y[:, 0], Y[:, 1], c=colors)


    plt.savefig('Output/TSNE/tsne_time_{}.png'.format(time))
    plt.close()

X = np.array([pop_vectors[stim][p] for stim in np.unique(rec.s_vector) for p in pop_vectors[stim]])

umap = umap.UMAP(n_neighbors=5)
Y = umap.fit_transform(X)

print(X.shape, Y.shape)
plt.scatter(Y[green, 0], Y[green, 1], c="g")
plt.scatter(Y[red, 0], Y[red, 1], c="r")
plt.scatter(Y[blue, 0], Y[blue, 1], c="b")
plt.scatter(Y[yellow, 0], Y[yellow, 1], c="y")
plt.scatter(Y[gray, 0], Y[gray, 1], c="gray")



plt.show()
plt.close()

# total_rec.get_population_vectors(0, 500)

# rec = Recording(paths.Ksdir, paths.SoundInfo, name='testing')
# rec.select_data_quality(quality='good')
# rec.ttl_alignment(multi=False)




# Still need to drax psycoM curves





# Compute global N x T matrix
# dict_timings = {}
# for clu in track(usable_clusters, description='Computing individual cluster ...'):
#   dict_timings[clu] = np.where(spikes_cluster == clu)[0]


# dict_stim_timings = {}
# for stim in track(np.unique(stim_vector), description='Generating individual timings ...'):
#   curr_stim = np.where(stim_vector == stim)[0]
#   dict_stim_timings[stim] = {}
#   for i, pres in enumerate(curr_stim):
#       ttl = ttl_indices[pres]
#       dict_stim_timings[stim][i] = [(v[np.logical_and(v >= ttl - params.pad_before, v <= ttl + params.pad_after)] - ttl)/params.fs*1000 for v in list(dict_timings.values())]

# for stim in dict_stim_timings:
#   curr_stim = [[sum(list(dict_stim_timings[stim][i][j])) for i in dict_stim_timings[stim]] for j in range(len(usable_clusters))]
#   dict_stim_timings[stim] = [a for l in curr_stim for a in l if a]
#   dict_stim_timings[stim].sort()
# #     dict_stim_timings[stim] = [l for l in curr_stim if l]

# print([len(dict_stim_timings[stim]) for stim in dict_stim_timings])








# all_psth = {}
# for i in track(range(max(stim_vector)), description='Generating PSTHs...'):
#   psths = []
#   curr_stim = np.where(stim_vector == i)[0]
#   for stim in curr_stim:
#       curr_ttl = list(ttl_indices)[0::2][stim]

#       curr_n_spikes = spikes_times[spikes_times > curr_ttl - params.pad_before]
#       curr_n_spikes = np.array(curr_n_spikes[curr_n_spikes < curr_ttl + params.pad_after], dtype=np.int64)
#       psths.append(np.array(curr_n_spikes - curr_ttl)/params.fs*1000)

#   psth = [i for p in psths for i in p]
#   all_psth[sound_names[i]] = psth



# for sound in track(all_psth, description='Drawing Figures...'):
#   sns.set_theme(style='ticks')

#   f, ax = plt.subplots(figsize=(7, 5))
#   sns.despine(f)

#   sns.histplot(data=all_psth[sound], palette='light:m_r',
#                edgecolor='.3', linewidth=.5, bins=50)
#   plt.axvline(0, color='red')
#   ax.set_xlabel('Time (ms)')
#   ax.set_ylabel('# of spikes')
#   ax.set_title('{}'.format(sound[:-4]))
#   plt.savefig(os.path.join(paths.Output, '{}.png'.format(sound[:-4])), dpi=150)
#   plt.close()


# May be useful to get correlation matrices between histograms
