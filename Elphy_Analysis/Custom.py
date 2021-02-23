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

#from auth import spid

class Mouse(object):
    """docstring for Mouse"""
    def __init__(self, path=None, ID=None, output='../Output', rmgaps=False):
        self.ID = ID
        self.output = output

        if self.ID:
            self.df_beh = self.__get_data_from_gsheet()
        # else:
        #     print('Please provide an ID to retrieve data from Google Sheets')
        if path:
            self.ID = os.path.basename(os.path.normpath(path))
            self.elphy = self.__process_elphy_at_file(path)
            self.df_beh = self.__get_data_from_gsheet()
        else:
            print('Please provide a path to retrieve data from elphy dat files')

    def __get_data_from_gsheet(self):
        """ Retrieve behavioural etadat from Google Sheet"""
        SAMPLE_SPREADSHEET_ID_input = '1PNvkKMTGbVxGGG-2eyWFEtG9dcv3ZVb9m9zVixjRlfc'
        SAMPLE_RANGE_NAME = 'A1:AA1000'

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

        # Extract behavioural data on a daily basis and return a pd dataframe
        dates, weights, water_profile, health, protocol, estimated_perf = [], [], [], [], [], []

        for row in range(11, len(all_data[mouse_idx]), 6):
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

    def __process_elphy_at_file(self, folder):
        """ Order rax elphy data into an usable dictionary
        """
        files = []
        for file in os.listdir(folder):
            files.append(self.File(os.path.join(folder, file)))

        sorted_dates = np.argsort([datetime.datetime.strptime(f.date, '%d%m%Y') for f in files])

        files = [files[i] for i in sorted_dates]

        return files

    def save(self):
        pass

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

    def weight(self, plot=True):
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

    def perf(self, tag=['DIS', 'PC'], plot=False):
        """ Compute evolution of mouse's performance following the task's type"""
        if tag:
            files = [file for file in self.elphy if file.tag in tag]

        correct_tr = [100*sum(f.tr_corr)/len(f.tr_corr) for f in files]
        dates = [f.date for f in files]
        tags = [f.tag for f in files]

        dates = pd.to_datetime(dates, format='%d%m%Y')


        correct_tr = [correct_tr[i] for i in np.argsort(dates)]
        tags = [tags[i] for i in np.argsort(dates)]
        dates = np.sort(dates)

        if len(files) > 1 and plot:
            print(plot)
            self.__perf_fig(dates, correct_tr)

        return correct_tr, dates

    def psychoacoustic_fig(self, frequencies, prob, stim_freqs):

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.set_xscale('log')

        if stim_freqs is not None:
            ax.plot(stim_freqs, prob)
            ax.axvline(x=(stim_freqs[int(len(stim_freqs)/2)-1]+stim_freqs[int(len(stim_freqs)/2)])/2, c='red', ls='--', linewidth=1)
        else:
            ax.plot(frequencies, prob)

        plt.show()

    def psychoacoustic(self, tag='PC', last=False, stim_freqs=None, plot=True, date=None, threshold=None):
        if date:
            files = [file for file in self.elphy if file.date == date]
        else:
            files = [file for file in self.elphy if file.tag in tag]

        if last:
            files = [files[-1]]

        if threshold:
            ps, _ = self.perf(tag=tag)
            files = [f for f, p in zip(files, ps) if p > threshold]


        licks = np.array([item for f in files for item in f.tr_licks])
        tasks = np.array([item for f in files for item in f.tr_type])


        ######### Careful !!! Must have the same lick threshold than during the task, retrieve from elphy
        licks = (licks >= 5)


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

        ax2.set_title(label='Psychoacoustic',
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
        print('Lick Number : ', file.xpar['fix']['LickNumber'])
        print('Refractory Time :', file.xpar['fix']['RefractoryTime'])
        print('Random Refractory Time :', file.xpar['fix']['RandomRefractoryTime'])

    class File(object):
        """DAT file as an object for better further use"""
        def __init__(self, path, rmgaps=True):
            self.path = path
            self.__filename_parser(os.path.basename(self.path))
            self.__extract_data(self.path, rmgaps)

        def __extract_data(self, path, rmgaps):
            recordings, vectors, xpar = ertd.read_behavior(os.path.join(path), verbose=False)

            self.xpar = xpar

            if rmgaps:
                self.tr_type, self.tr_licks, self.tr_corr = self.__removeGaps(vectors['TRECORD'], vectors['LICKRECORD'], vectors['correct'])
            else:
                self.tr_type = vectors['TRECORD']
                self.tr_licks = vectors['LICKRECORD']
                self.tr_corr = vectors['correct']

        def __filename_parser(self, filename):
            parsed_filename = filename.split('_')
            self.tag, self.date, self.ID, self.nfile = parsed_filename

        def __removeGaps(self, ttype, licks, corr):
            """ Remove gaps when the mouse is not licking at all so the data is not corrupted by a bored mouse
            """
            licks = list(licks)
            ttype = list(ttype)
            corr = list(corr)

            str_licks = [str(i) for i in licks]

            str_licks = ''.join(str_licks)
            
            no_licks = [[m.start(), m.end()] for m in re.finditer('[^1-9]+', str_licks) if m.end() - m.start() > 10]
            
            for gap in reversed(no_licks):
                del licks[gap[0]:gap[1]]
                del ttype[gap[0]:gap[1]]
                del corr[gap[0]:gap[1]]

            print('Gaps removed - {} : '.format(self.date))
            for gap in no_licks:
                print(gap)

            return ttype, licks, corr

mouse = Mouse('/home/user/share/gaia/Data/Behavior/Antonin/660463', rmgaps=True)
#mouse.get_session_info('22022021')
mouse.correct_graph('22022021')
mouse.summary(tag=['PC'], show=True, stim_freqs=np.geomspace(6e3, 16e3, 16), threshold=80)
#mouse.summary(tag=['DISAM'], show=True, stim_freqs=[1, 2, 3], threshold=80)

# make a function to find specific files for one mouse and be able to call it

