# Author : Antonin Verdier

import os
import re
import pickle
import datetime
import numpy as np
import pandas as pd
import elphy_reader as ertd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow,Flow
from google.auth.transport.requests import Request

from collections import Counter

import settings

batch = settings.Batch()



class Mouse(object):
    """ Use to define a Mouse object encapsulating Elphy and Behavioural data and to perform analysis on it.

    Attributes:
    -----------
    ID : int
        Mouse identification number in the mouse house
    path : str
        path to folder containing all the elphy files
    output : str
        path for figure and data storage
    
    Constructors
    ------------
    __init__(self, path=None, ID=None, output='../Output', rmgaps=False, elphy_only=False, tag=None, date=None, collab=False)
        Initialize and define which files to load

    Methods
    -------
    weight(plot=False)
        Return weights of the mouse and dates ordered
    perf(tag=['DIS', 'PC'], plot=False, dateformat='%d%m%Y')
        Return the total score of the mouse for each session
    psychoacoustic(self, tag=['PC'], last=False, stim_freqs=None, plot=True, date=None, threshold=None)
        Return the probabilities of lick for each task
    summary(self, tag=['PC'], stim_freqs=None, last=False, name='summary', show=False, threshold=None)
        Provide a global figure with weight, performance and psycoacoustic curve of a mice
    correct_graph(date)
        Return the output correct or incorrect for each trial
    get_session_info(date)
        Return lick nbumber and timing of a particular session
    score_by_task(names=None, plot=True)
        Return the score of the mice for each type of task in the session 
    lick_number_by_task(names=None, plot=True)
        Return the number of licks for each type of task in the session 
   """
    def __init__(self, path=None, ID=None, output='../Output', rmgaps=False, elphy_only=False, tag=None, date=None, dlp=False, collab=False, verbose=True, rmblocks=None, linkday=False, blank=False, gsheet=True):
        self.ID = ID
        self.path = path
        self.output = output
        self.rmgaps = rmgaps
        self.verbose = verbose
        self.rmblocks = rmblocks
        self.linkday = linkday
        self.blank = blank

        if self.ID:
            self.df_beh = self.__get_data_from_gsheet()
        # else:
        #     print('Please provide an ID to retrieve data from Google Sheets')
        # if elphy_only:
        #     self.elphy = self.__process_elphy_at_file(path)
        elif path and not elphy_only:
            self.ID = os.path.basename(os.path.normpath(path))
            #self.elphy = self.__process_elphy_at_file(path)
            self.elphy = self.__process_elphy_file_by_tag(path, tag, date)
            if self.linkday:
                to_link = [i for i, f in enumerate(self.elphy[:-1]) if f.date == self.elphy[i + 1].date]
                if to_link:
                    for t in reversed(to_link):
                        self.elphy[t].tr_type = np.append(self.elphy[t].tr_type , self.elphy[t + 1].tr_type)
                        self.elphy[t].tr_corr = np.append(self.elphy[t].tr_corr , self.elphy[t + 1].tr_corr)
                        self.elphy[t].ta_type = np.append(self.elphy[t].ta_type , self.elphy[t + 1].ta_type)
                        self.elphy[t].tr_licks = np.append(self.elphy[t].tr_licks , self.elphy[t + 1].tr_licks)

                        self.elphy = np.delete(self.elphy, t+1)

            if gsheet: 
                print('Retrieve data from Gsheet')
                self.df_beh = self.__get_data_from_gsheet(collab=collab, dlp=dlp)
            else:
                self.reversed = True

        else:
            print('Please provide a path to retrieve data from elphy dat files')

    def __get_data_from_gsheet(self, collab=False, dlp=False):
        """ Retrieve behavioural etadat from Google Sheet"""
        SAMPLE_SPREADSHEET_ID_input = '1PNvkKMTGbVxGGG-2eyWFEtG9dcv3ZVb9m9zVixjRlfc'
        if collab: SAMPLE_SPREADSHEET_ID_input = '1utsDBiSvcNIuYyOS4LiIG0wRCn7fdxf7-EbAjwbXntE'
        if dlp: SAMPLE_SPREADSHEET_ID_input = '1LaetwSVk1E4hXb5_eDdARpfZD25l2njMKIXBHeZxqHo'
        SAMPLE_RANGE_NAME = 'A1:BA1000'

        creds = self.__google_credentials()
        service = build('sheets', 'v4', credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result_input = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
                                    range=SAMPLE_RANGE_NAME, majorDimension='COLUMNS').execute()
        all_data = result_input.get('values', [], )

        for i, col in enumerate(all_data):
            all_data[i] = ["x" if x == '' else x for x in col]

        # Read basic infos given the self_id of the mouse
        mice_ids = [col[0] for col in all_data]
        assert self.ID, 'A mouse ID need to be specified'
        mouse_idx = mice_ids.index(self.ID)

        # Get basic infos about the mouse and its surgery
        self.strain = all_data[mouse_idx][1]
        self.sex = all_data[mouse_idx][2]
        self.experimenter = all_data[mouse_idx][3]
        self.date_surgery = all_data[mouse_idx][4]
        self.surgeon = all_data[mouse_idx + 1][4]
        self.surgery_type = all_data[mouse_idx][5]
        self.weight_at_surgery = all_data[mouse_idx][6]
        self.postop_obs = all_data[mouse_idx][7]
        self.date_waterd = all_data[mouse_idx][8]
        self.hour_waterd = all_data[mouse_idx + 1][8]
        self.weight_befored = all_data[mouse_idx][9]
        self.health_atd = all_data[mouse_idx][10]
        self.reversed = int(all_data[mouse_idx][11])

        # Extract behavioural data on a daily basis and return a pd dataframe
        dates, weights, water_profile, health, protocol, estimated_perf = [], [], [], [], [], []

        for row in range(12, len(all_data[mouse_idx]), 6):
            dates.append(all_data[mouse_idx][row])
            weights.append(np.float(all_data[mouse_idx][row + 1]))
            water_profile.append(all_data[mouse_idx][row + 2])
            health.append(all_data[mouse_idx][row + 3])
            protocol.append(all_data[mouse_idx][row + 4])
            estimated_perf.append(all_data[mouse_idx][row + 5])

        dict_beh = {'date': dates,
                    'weight': weights,
                    'water_profile': water_profile,
                    'health': health,
                    'protocol': protocol,
                    'estimated_perf': estimated_perf}

        df_beh = pd.DataFrame(data=dict_beh)
        df_beh['date'] = pd.to_datetime(df_beh['date'], format='%d/%m/%Y')


        return df_beh

    def __google_credentials(self):
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES) # here enter the name of your downloaded JSON file
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return creds

    def __process_elphy_at_file(self, folder, tag=False):
        """ Order rax elphy data into an usable dictionary
        """
        files = []
        for file in os.listdir(folder):
            if tag:
                if file.split('_')[0] in tag:
                    current_file = self.File(os.path.join(folder, file), self.rmgaps, self.rmblocks)
                    if len(current_file.tr_corr) != 0:
                        files.append(current_file)
            else:
                current_file = self.File(os.path.join(folder, file), self.rmgaps, self.rmblocks)
                if len(current_file.tr_corr) != 0:
                    files.append(current_file)

        sorted_dates = np.argsort([datetime.datetime.strptime(f.date, '%d%m%Y') for f in files])

        files = [files[i] for i in sorted_dates]

        return files
    def __process_elphy_file_by_tag(self, folder, tag, date=None):
        files = []
        if self.verbose: print('Processing files for mice {}...'.format(self.ID))
        for file in os.listdir(folder):
            for t in tag:
                if t + '_' == file.split('_')[0] + '_':
                    if self.verbose: print(file)
                    current_file = self.File(os.path.join(folder, file), self.rmgaps, self.rmblocks)
                    if len(current_file.tr_corr) != 0:
                        files.append(current_file)

        sorted_dates = np.argsort([datetime.datetime.strptime(f.date, '%d%m%Y') for f in files])
        files = [files[i] for i in sorted_dates]

        if date: files = [f for f in files if f.date==date]

        return files


    def save(self):
        pass

    def select_best_trials(self, n_trials):
        files = self.elphy
        ps, _ = self.perf()
        bests = np.argsort(ps)
        self.elphy = list(np.array(files)[bests[-n_trials:]])
        print('Best scores are :', np.array(ps)[bests[-n_trials:]])

    def __weight_fig(self, weights, dates):
        plt.figure(figsize=(12, 9))

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(c='gainsboro')
        ax.plot(dates, weights, 'ro-')

        mults = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
        cs = ['chartreuse', 'chartreuse', 'gold', 'gold', 'firebrick', 'firebrick']

        for mult, c in zip(mults, cs):
            ax.axhline(y=weights[0]+weights[0]*mult, c=c, ls='--', linewidth=1)

        plt.ylim([weights[0]-5, weights[0]+5])
        plt.title(label='Weight evolution of {}'.format(self.ID),
                  fontsize=15,
                  fontstyle='italic')

        # For further improvment, add a legend for dashed lines
        plt.show()

    def weight(self, plot=False):
        weights = self.df_beh['weight']
        dates = self.df_beh['date']

        if plot: self.__weight_fig(weights, dates)

        return weights, dates

    def __perf_fig(self, dates, correct_tr):
        """ Plot evolution of mouse's performance following the task's type"""

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(c='gainsboro', ls='--')


        ax.plot(dates, correct_tr, 'o-')
        ax.set_ylim(0, 100)
        ax.set_yticks(np.linspace(0, 100, 11))
        plt.show()

    def perf(self, tag=None, plot=False, dateformat='%d%m%Y', blank=False, limit_n=False):
        """ Compute evolution of mouse's performance following the task's type"""
        files = self.elphy

        if tag:
            print(tag)
            files = [file for file in self.elphy if tag in file.tag]

        if self.blank or blank:
            fs = []
            for f in files:
                idx_blank = np.where(np.array(f.tr_type) == 1)[0]
                fs.append([c for i, c in enumerate(f.tr_corr) if i not in idx_blank])
            
            correct_tr = [100*sum(f)/len(f) for f in fs]
        else:     
            correct_tr = [100*sum(f.tr_corr)/len(f.tr_corr) for f in files]
        
        dates = [f.date for f in files]
        tags = [f.tag for f in files]

        dates = pd.to_datetime(dates, format=dateformat)


        correct_tr = [correct_tr[i] for i in np.argsort(dates)]
        tags = [tags[i] for i in np.argsort(dates)]
        dates = np.sort(dates)

        if len(files) > 1 and plot:
            print('fig')
            self.__perf_fig(dates, correct_tr)
        
        if limit_n:
            valid_trials = [0 if len(f.tr_corr) < 0.2 * f.xpar['fix']['MaxTrialNumber'] else 1 for f in files]
            print(valid_trials)
            return correct_tr, dates, valid_trials

        else:
            return correct_tr, dates

    def psychoacoustic_fig(self, frequencies, prob, stim_freqs):

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)

        if stim_freqs is not None:
            ax.plot(stim_freqs, prob)
            ax.axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
        else:
            ax.plot(frequencies, prob)

        plt.show()

    def psychoacoustic(self, tag=None, last=False, stim_freqs=None, plot=True, date=None, threshold=None):
        files = self.elphy

        if tag:
            files = [file for file in self.elphy if tag in file.tag]

        for i, f in enumerate(files):
            if len(f.tr_type) != len(f.tr_corr):
                files[i].tr_type = files[i].tr_type[:len(f.tr_corr)]

        if date:
            files = [file for file in files if file.date in date]

        # else:
        #     files = [file for file in self.elphy if file.tag in tag]

        if last:
            files = [files[-1]]

        if threshold:
            ps, _ = self.perf(tag=tag)
            files = [f for f, p in zip(files, ps) if p > threshold]

        tasks = np.array([item for f in files for item in f.tr_type])
        corr = np.array([item for f in files for item in f.tr_corr])
        ta_type = np.array([item for f in files for item in f.ta_type])

        print(len(corr))
        
        if self.reversed:
            licks = [not c if ta_type[i] == 1 else c for i, c in enumerate(corr)]
        else:
            licks = [not c if ta_type[i] == 2 else c for i, c in enumerate(corr)]

        
        P_lick = {key:sum(tasks*licks == key)/sum(tasks == key) for key in list(set(tasks))}

        sorted_P_licks = sorted(P_lick.items())
        frequencies, prob = zip(*sorted_P_licks)

        if plot: self.psychoacoustic_fig(frequencies, prob, stim_freqs)

        return frequencies, prob

    def summary(self, tag=['PC'], stim_freqs=None, last=False, name='summary', show=False, threshold=None):
        """ Display general infos about the current mice (average weight, sex, strain, etc., maybe age ??)
        """
        weights, dates_w = self.weight(plot=False)
        correct_tr, dates_p = self.perf(tag=tag, plot=False)
        frequencies, prob = self.psychoacoustic(tag=tag, plot=False, last=last, threshold=threshold)

        fig = plt.figure(constrained_layout=True, figsize=(12, 9))
        gs = fig.add_gridspec(4, 4)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[2:4, :])


        ax1.grid(c='gainsboro')
        ax1.plot(dates_w, weights, 'ro-')
        ax1.set_ylim([weights[0]-5, weights[0]+5])
        ax1.set_title(label='Weight evolution',
                      fontsize=13,
                      fontstyle='italic')

        mults = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
        cs = ['chartreuse', 'chartreuse', 'gold', 'gold', 'firebrick', 'firebrick']

        for mult, c in zip(mults, cs):
            ax1.axhline(y=weights[0]+weights[0]*mult, c=c, ls='--', linewidth=1)

        ax2.set_xscale('log')
        if stim_freqs is not None:
            ax2.plot(stim_freqs, prob)
            ax2.axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
        else:
            ax2.plot(frequencies, prob)

        ax2.set_title(label='Psychoacoustic, above {}%'.format(threshold),
                      fontsize=13,
                      fontstyle='italic')

        ax3.yaxis.grid(c='gainsboro', ls='--')
        ax3.set_ylim(0, 100)
        ax3.set_yticks(np.linspace(0, 100, 11))
        ax3.plot(dates_p, correct_tr, 'o-')

        ax3.set_title(label='Psychophysic',
                      fontsize=13,
                      fontstyle='italic')

        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        ax3.xaxis.set_major_locator(locator)
        ax3.xaxis.set_major_formatter(formatter)

        plt.suptitle('{}'.format(self.ID), fontsize=18)

        plt.savefig(os.path.join(self.output, '{}_{}.svg'.format(self.ID, name)))
        if show:
            plt.show()

    def correct_graph(self, date):
        file = [file for file in self.elphy if file.date == date][0]

        fig = plt.figure(figsize=(30, 2)) # Maybe best if it is a heatmap ?
        plt.subplot(211)
        plt.plot(file.tr_licks)
        for i in range(0, 17):
            print('{}: '.format(i), np.sum((file.tr_licks == i))*100/len(file.tr_licks))
        plt.subplot(212)
        plt.plot(file.tr_corr)
        plt.show()

    def get_session_info(self, date):
        file = [file for file in self.elphy if file.date == date][0]
        print(file.xpar)
        print('Lick Number : ', file.xpar['fix']['LickNumber'])
        print('Refractory Time :', file.xpar['fix']['RefractoryTime'])
        print('Random Refractory Time :', file.xpar['fix']['RandomRefractoryTime'])

    def score_by_task(self, names=None, plot=True):
        # Output the percentage of success for each type of stimulus
        files = self.elphy

        tasks = [f.tr_type for f in files]
        corr = [f.tr_corr for f in files]

        print([(files[i].path, len(c)) for i, c in enumerate(corr)])

        muul = [np.array(tasks[i]) * np.array(corr[i]) for i, f in enumerate(files)]
        flatten_muul = [int(i) for m in muul for i in m]
        flatten_tasks = [int(i) for t in tasks for i in t]

        valid = dict(Counter(flatten_muul))
        alls = dict(Counter(flatten_tasks))
        valid.pop(0, None)
        alls.pop(0, None)

        scores = [valid[i]*100/alls[i] if alls[i] !=0 else 0 for i in list(set(flatten_tasks))]

        
        if plot:
            if not names:
                names = range(1, len(scores)+1)
            plt.bar(x=names, height=scores)
            plt.savefig('{}.png'.format(self.ID))
            plt.show()
            plt.close()

        return scores

    def lick_number_by_task(self, names=None, plot=True):
        # Output the percentage of success for each type of stimulus
        files = self.elphy

        tasks = [f.tr_type for f in files]
        licks = [f.tr_licks for f in files]

        flatten_tasks = [int(i) for t in tasks for i in t]
        flatten_licks = [int(i) for t in licks for i in t]

        counters = {}
        for t in list(set(flatten_tasks)):
            counters[t] = []
        
        for i, (t, l) in enumerate(zip(flatten_tasks, flatten_licks)):
            counters[t].append(l)

        for k in counters:
            counters[k] = np.mean(counters[k])

        counter_list = []
        for i in range(1, 7):
            counter_list.append(counters[i])

        
        if plot:
            if not names:
                names = range(1, len(scores)+1)
            plt.bar(names, counter_list)
            plt.savefig('{}.png'.format(self.ID))
            plt.show()
            plt.close()

        return counters




        # lol = np.concatenate(muul)
        # lol = lol.flatten()
        # # Then simply count number of occurance in this and compared it to raw task 

        # tasks = [t[:len(corr[i])] for i, t in enumerate(tasks)]

        # scores = {}
        # for i in np.unique(tasks[0]):
        #     scores[i] = []
        #     for j, t in enumerate(tasks):
        #         masked_array = np.ma.masked_equal(t, i)
        #         masked_correctness = corr[j] * masked_array.mask
        #         curr_score = np.sum(masked_correctness)*100/np.sum(masked_array.mask)
        #         scores[i].append(curr_score)

        # final_scores = [np.mean(scores[k]) for k in scores]
        # final_std = [np.std(scores[k]) for k in scores]
        # if not names:
        #     names = range(1, len(final_scores)+1)

       
        # plt.bar(x=names, height=final_scores, yerr=final_std)
        # plt.savefig('{}.png'.format(self.ID))
        # plt.close()


    class File(object):
        """DAT file as an object for better further use"""
        def __init__(self, path, rmgaps, rmblocks):
            self.path = path
            self.rmgaps = rmgaps
            self.rmblocks = rmblocks
            self.__filename_parser(os.path.basename(self.path))
            self.__extract_data(self.path, rmgaps)
            if rmblocks is not None: self.__removeBadBlocks(rmblocks[0], rmblocks[1])

        def __extract_data(self, path, rmgaps):
            recordings, vectors, xpar = ertd.read_behavior(os.path.join(path), verbose=False)
            self.xpar = xpar
            self.recordings = recordings
            self.first_lick = [np.argmax(t > 1000) for t in self.recordings]


            #print(self.xpar['soundlist'])

            if rmgaps == 'Brice':
                self.tr_type, self.tr_licks, self.tr_corr, self.ta_type, self.first_lick = self.__remove_gaps_brice(vectors['TRECORD'],
                                                                                            vectors['LICKRECORD'],
                                                                                            vectors['correct'],
                                                                                            vectors['taskType'],
                                                                                            self.first_lick,
                                                                                            xpar)
            elif rmgaps == 'Antonin':
                self.tr_type, self.tr_licks, self.tr_corr, self.ta_type, self.first_lick = self.__remove_gaps_antonin(vectors['TRECORD'],
                                                                                            vectors['LICKRECORD'],
                                                                                            vectors['correct'],
                                                                                            vectors['taskType'],
                                                                                            self.first_lick,
                                                                                            xpar)
            else:
                self.tr_type = list(vectors['TRECORD'])
                self.tr_licks = list(vectors['LICKRECORD'])
                self.tr_corr = list(vectors['correct'])
                self.ta_type =  list(vectors['taskType'])

        def __filename_parser(self, filename):
            parsed_filename = filename.split('_')
            self.tag, self.date, self.ID, self.nfile = parsed_filename
            #self.tag += '_'

        def __removeBadBlocks(self, threshold, bloc_size):
            if len(self.tr_corr)%bloc_size == 0:
                div = len(self.tr_corr)//bloc_size
                blocks_to_keep = [i for i, bloc in enumerate(np.split(np.array(self.tr_corr), div)) if sum(bloc)/len(bloc) > threshold]
                if blocks_to_keep:
                    self.tr_type = np.concatenate([bloc for i, bloc in enumerate(np.split(np.array(self.tr_type), div)) if i in blocks_to_keep])
                    self.tr_licks = np.concatenate([bloc for i, bloc in enumerate(np.split(np.array(self.tr_licks), div)) if i in blocks_to_keep])
                    self.tr_corr = np.concatenate([bloc for i, bloc in enumerate(np.split(np.array(self.tr_corr), div)) if i in blocks_to_keep])
                    self.ta_type = np.concatenate([bloc for i, bloc in enumerate(np.split(np.array(self.ta_type), div)) if i in blocks_to_keep])
                    self.first_lick = np.concatenate([bloc for i, bloc in enumerate(np.split(np.array(self.first_lick), div)) if i in blocks_to_keep])
                else:
                    self.tr_type = []
                    self.tr_licks = []
                    self.tr_corr = []
                    self.ta_type = []
                    self.first_lick = []

        def __remove_gaps_antonin(self, ttype, licks, corr, tatype, flicks, xpar):
            """ Remove gaps when the mouse is not licking at all so the data is not corrupted by a bored mouse
            """
            licks = list(licks)
            ttype = list(ttype)
            corr = list(corr)
            tatype = list(tatype)
            flicks = list(flicks)
            str_licks = [1 if l >= 1 else l for l in licks] # Assure que le nombre de lick soit un chiffre unique
            str_licks = [str(i) for i in str_licks]
            str_licks = ''.join(str_licks)
            no_licks = [[m.start(), m.end()-1] for m in re.finditer('[0]+', str_licks) if m.end() - m.start() > 15]

            for gap in reversed(no_licks):
                del licks[gap[0]:gap[1]]
                del ttype[gap[0]:gap[1]]
                del corr[gap[0]:gap[1]]
                del tatype[gap[0]:gap[1]]
                del flicks[gap[0]:gap[1]]


            # print('Gaps removed - {} : '.format(self.date))
            # for gap in no_licks:
            #     print(gap)

            return ttype, licks, corr, tatype, flicks

        def __remove_gaps_brice(self, ttype, licks, corr, tatype, flicks, xpar):
            """ Remove gaps when the mouse is not licking at all so the data is not corrupted by a bored mouse
            """
            licks = list(licks)
            ttype = list(ttype)
            corr = list(corr)
            tatype = list(tatype)
            flicks = list(flicks)

            go_idx = [i for i, g in enumerate(tatype) if g == 1]
            go_corr = [int(corr[i]) for i in go_idx]

            str_gos = [str(i) for i in go_corr]
            str_gos = ''.join(str_gos)
            no_licks = [[m.start(), m.end()-1] for m in re.finditer('[0]+', str_gos) if m.end() - m.start() > 5]

            for gap in reversed(no_licks):
                del licks[go_idx[gap[0]]:go_idx[gap[1]]]
                del ttype[go_idx[gap[0]]:go_idx[gap[1]]]
                del corr[go_idx[gap[0]]:go_idx[gap[1]]]
                del tatype[go_idx[gap[0]]:go_idx[gap[1]]]
                del flicks[go_idx[gap[0]]:go_idx[gap[1]]]

            # print('Gaps removed - {} : '.format(self.date))
            # for gap in no_licks:
            #     print(gap)

            return ttype, licks, corr, tatype, flicks

# mouse = Mouse('/home/user/share/gaia/Data/Behavior/Antonin/741151', tag=['PC'], collab=False)
# mouse.psychoacoustic(tag=['PC'], stim_freqs=np.geomspace(4e3, 16e3, 16), plot=True, threshold=70)
#mouse.get_session_info('04032021')
#mouse.correct_graph('02022021')
# for m in batch.id_first_collab:
#     mouse = Mouse('/home/user/share/gaia/Data/Behavior/Antonin/{}/'.format(m), tag=['OPTO'], date='01062021', collab=True)
#     mouse.score_by_task()#names=['Blank_NOL', '50ms_NOL', '150ms_NOL', 'Blank_L', '50ms_L', '150ms_L'])
#mouse.weight(plot=True)
#mouse.summary(tag=['DISAM'], show=True, stim_freqs=[1, 2, 3], threshold=0)

# make a function to find specific files for one mouse and be able to call it

