import os
import numpy as np
import pandas as pd
import elphy_reader as ertd

class Mouse(object):
	"""docstring for Mouse"""
	def __init__(self, ID=0, sex='M', strain='C57BL/6J'):
		self.ID = ID
		self.sex = sex 
		self.strain = strain
		self.elphy = {}
		self.beh = {}
		self.results = {}

	def populate(self, folder):
		""" Look for data to populate the object
		"""
		dat_files = os.listdir(folder)

		self.ID = os.path.basename(os.path.normpath(folder)) 
		__process_elphy_at_file(folder)


	def __get_data_from_gsheet(self):
		""" Retrieve behavioural ùetadat from Google Sheet"""
		pass

	def __process_elphy_at_file(self, folder):
		""" Order rax elphy data into an usable dictionary
		"""
		for file in os.listdir(folder):
			recordings, vectors, xpar = ertd.read_behavior(os.path.join(foler, file), verbose=False)
			
			session = {}
			session['tr_type'] = vectors['TRECORD']
			session['tr_licks'] = vectors['LICKRECORD']
			session['tr_correct'] = vectors['correct']

			df_session = pd.DataFrame(data=session)

			return df_session

	
	def save(self):
		pass
