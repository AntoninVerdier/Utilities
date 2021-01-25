import os
import pickle
import numpy as np
import pandas as pd
import elphy_reader as ertd
import matplotlib.pyplot as plt


from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow,Flow
from google.auth.transport.requests import Request

from Hearis import get_P_lick

#from auth import spid

class Mouse(object):
    """docstring for Mouse"""
    def __init__(self, path=None, ID=None):
        self.ID = ID

        if self.ID:
            self.df_beh = self.__get_data_from_gsheet()
        else:
            print('Please provide an ID to retrieve data from Google Sheets')
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
            weights.append(float(all_data[mouse_idx][row + 1]))
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

        return files


    def save(self):
        pass

    def weight_fig(self):
        weights = self.df_beh['weight']
        date = self.df_beh['date']
        print(weights[0])

        plt.figure(figsize=(12, 9))

        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(c='gainsboro')
        ax.plot(date, weights, 'ro-')

        mults = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
        cs = ['chartreuse', 'chartreuse', 'gold', 'gold', 'firebrick', 'firebrick']

        for mult, c in zip(mults, cs):
            ax.axhline(y=weights[0]+weights[0]*mult, c=c, ls='--', linewidth=1)

        plt.ylim([20, 31])
        plt.title(label='Weight evolution of {}'.format(self.ID),
                  fontsize=15,
                  fontstyle='italic')

        # For further improvment, add a legend for dashed lines


        plt.show()

    def perf_fig(self, tag='DIS', stims=['blank', '12k', '20k'], last=False):
        """ Show evolution of mouse's performance following the task's type"""
        if tag:
        	files = [file for file in self.elphy if file.tag == tag]

        if last:
        	files = [files[-1]]

        correct_tr = [100*sum(f.tr_corr)/len(f.tr_corr) for f in files]
        dates = [f.date for f in files]
        print(dates, correct_tr)

        dates = pd.to_datetime(dates, format='%d%m%Y')


        correct_tr = [correct_tr[i] for i in np.argsort(dates)]
        dates = np.sort(dates)

      
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(c='gainsboro', ls='--')


        ax.plot(dates, correct_tr)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.linspace(0, 100, 11))
        plt.show()

        return dates, correct_tr

    def mouse_summary(self):
        """ Display general infos about the current mice (average weight, sex, strain, etc., maybe age ??)
        """
        pass


    class File(object):
        """DAT file as an object for better further use"""
        def __init__(self, path):
            self.path = path
            self.__filename_parser(os.path.basename(self.path))
            self.__extract_data(self.path)

        def __extract_data(self, path):
            recordings, vectors, xpar = ertd.read_behavior(os.path.join(path), verbose=False)

            self.tr_type = vectors['TRECORD']
            self.tr_licks = vectors['LICKRECORD']
            self.tr_corr = vectors['correct']

        def __filename_parser(self, filename):
            parsed_filename = filename.split('_')
            self.tag, self.date, self.ID, self.nfile = parsed_filename


mouse = Mouse(path='/home/user/share/gaia/Data/Behavior/Antonin/660270')
mouse.perf_fig()
