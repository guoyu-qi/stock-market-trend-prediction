import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

class DataProcessor:

	def __init__(self):
		#Path for data
		self.raw_data_path = "data/raw_data/"
		self.proc_data_path = "data/proc_data/"

	def getFileName(self):
		file_list = os.listdir(self.raw_data_path)
		return(file_list)

	def imputeData(self, file_name, header_format, header):
		header_format = [int(i)-1 for i in header_format]
		raw_date = np.genfromtxt(self.raw_data_path+file_name ,delimiter="," , autostrip = True, usecols = (header_format[0]), dtype=str)
		raw_date = np.reshape(raw_date, (raw_date.shape[0], 1))
		raw_ohlcv = np.genfromtxt(self.raw_data_path+file_name ,delimiter="," , autostrip = True, usecols = tuple(header_format[1:]) )
		imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
		imp_mean.fit(raw_ohlcv)
		proc_ohlcv = imp_mean.transform(raw_ohlcv)

		proc_data = np.concatenate((raw_date,proc_ohlcv),axis=1) 
		pd.DataFrame(proc_data).to_csv(self.proc_data_path+file_name[:-4] + "_PROC" + file_name[-4:], header=header, index = None)		