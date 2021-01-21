import os
import numpy as np
import pandas as pd
import elphy_reader as ertd

class Mouse(object):
	"""docstring for Mouse"""
	def __init__(self, ID=0, sex='M', strain='C57BL/6J'):
		self.ID = None
		self.sex = sex
		self.strain = strain
		self.elphy = {}
		self.beh = {}
		self.results = {}

	def populate(self, folder):
		""" Look for data to populate the object
		"""
		self.ID = os.path.basename(os.path.normpath(folder))
		elphy_data = __process_elphy_at_file(folder)

		self.elphy = elphy_data

	def __get_data_from_gsheet(self):
		""" Retrieve behavioural ùetadat from Google Sheet"""
		spreedsheet_id = '1PNvkKMTGbVxGGG-2eyWFEtG9dcv3ZVb9m9zVixjRlfc'

	def __process_elphy_at_file(self, folder):
		""" Order rax elphy data into an usable dictionary
		"""
		files = []
		for file in os.listdir(folder):
			files.append(File(os.path.join(folder, file)))

		return files


	def save(self):
		pass

	class File(object):
		"""DAT file as an object for better further use"""
		def __init__(self, path):
			self.path = path
			__filename_parser(os.path.basename(self.path))
			__extract_data(self.path)

		def __extract_data(self, path):
			recordings, vectors, xpar = ertd.read_behavior(os.path.join(path), verbose=False)

			self.tr_type = vectors['TRECORD']
			self.tr_licks = vectors['LICKRECORD']
			self.tr_corr = vectors['correct']

		def __filename_parser(self, filename):
			parsed_filename = filname.split('_')
			self.tag, self.date, self.ID, self.nfile = parsed_filename





