from matplotlib import pyplot
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
    
class Arima:

	def __init__(self, filename, train_test_split, prediction_type, prediction_col):
		#string to int
		train_test_split = list(map(int, train_test_split))
		self.filepath = "indicators/"
		self.filename = filename
		self.x_train_start = train_test_split[0]
		self.train_size = train_test_split[1]
		self.test_size = train_test_split[2]
		self.prediction_col = prediction_col
		if(prediction_type[:10] == "Open/Close"):
			self.prediction_type = "oc"
			self.oc_col = [7,8]
			self.abs_prediction_col = 8
		elif(prediction_type[:11] == "Close/Close"):
			self.prediction_type = "cc"
			self.oc_col = [2,5]
			self.abs_prediction_col = 5
		self.data = np.genfromtxt(self.filepath+self.filename ,delimiter = ',' , autostrip = True)	
		self.train_result = self.data[self.x_train_start : self.x_train_start+self.train_size, self.prediction_col]
		self.train_data = self.data[self.x_train_start:self.x_train_start+self.train_size, self.abs_prediction_col]
		self.test_data = self.data[self.x_train_start+self.train_size:self.x_train_start+self.train_size+self.test_size,self.abs_prediction_col]
		self.test_result = self.data[self.x_train_start+self.train_size:self.x_train_start+self.train_size+self.test_size,self.prediction_col]

		#creating series data
		dateparse = lambda dates: [pd.datetime.strptime(d, '%d-%m-%Y') for d in dates]
		temp_data = pd.read_csv(self.filepath+self.filename, header=1, parse_dates=["date"], date_parser=dateparse)	
		self.series_data = pd.Series(data = temp_data["c_tday"].tolist(),index = temp_data["date"])	


	def GenerateCnfMatrix(self, Y_pred, Y_test):
		Y_p = []
		Y_t = []
		for i in range(len(Y_pred)):
			Y_p 	= np.hstack((Y_p,Y_pred[i]))
		for i in range(len(Y_test)):
			Y_t 	= np.hstack((Y_t,Y_test[i]))
		cnf_matrix 	= confusion_matrix(Y_t, Y_p, labels = [1,-1])
		return cnf_matrix

	def ComputeDistribution(self, Y_train, Y_test):
		lp = 0.0
		lm = 0.0
		Y_train = np.sign(Y_train)
		Y_test = np.sign(Y_test)
		plus = Y_train == [1]*len(Y_train)
		minus = Y_train == [-1]*len(Y_train)
		plusses = Y_train[plus]
		minuses = Y_train[minus]
		lp = lp + len(plusses) + 0.0
		lm = lm + len(minuses) + 0.0
		plus = Y_test == [1]*len(Y_test)
		minus = Y_test == [-1]*len(Y_test)
		plusses = Y_test[plus]
		minuses = Y_test[minus]
		lp = lp + len(plusses)
		lm = lm + len(minuses)
		return (lp/(lp + lm))*100, (lm/(lp+lm))*100

	def ComputeAccuracyForOne(self, cnf_mat):	
#	[[tp, fn]
#	 [fp, tn]]
		tp, fn, fp, tn 		= (cnf_mat).ravel()+(0.0,0.0,0.0,0.0)
		precision 			= (tp/(tp + fp))*100
		recall 				= (tp/(tp + fn))*100
		specificity			= (tn/(tn + fp))*100
		accuracy_total 		= ((tp + tn)/(tp + tn + fp + fn))*100
		accuracy_plus 		= (tp/(tp + fp))*100
		accuracy_minus 		= (tn/(tn + fn))*100
		percent_plus		= ((tp+fp)/(tp + tn + fp + fn))*100
		percent_minus		= ((tn+fn)/(tp + tn + fp + fn))*100
		precent_list		= list((percent_plus, percent_minus))
		accuracy_list		= list((accuracy_total, accuracy_plus, accuracy_minus))
		return tuple(precent_list), tuple(accuracy_list)

	def ComputeAccuracy(self, cnf_mat_test, cnf_mat_train, name, 	actual_dist, need_print):
		per_train, acc_train 	= self.ComputeAccuracyForOne(cnf_mat_train)
		per_test, acc_test 		= self.ComputeAccuracyForOne(cnf_mat_test)
		if(need_print == 1):
			print (name + '_dist_actual_total          : ' + '%.3f %%,\t %.3f %%' %actual_dist)
			print (name + '_dist_pred_train            : ' + '%.3f %%,\t %.3f %%' %per_train)
			print (name + '_dist_pred_test             : ' + '%.3f %%,\t %.3f %%' %per_test)
			print (name + '_accuracy_train_[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_train)
			print (name + '_accuracy_test__[T,+,-]     : ' + '%.3f %%,\t %.3f %%,\t %.3f %%' %acc_test)
			print ('\n')
		return (per_test, per_train, acc_test, acc_train)

	#Function that calls ARIMA model to fit and forecast the data
	def StartARIMAForecasting(self, P, D, Q, timepoint):
		# if(timepoint==0):
		# 	stepwise_model = auto_arima(self.series_data, start_p=0, start_q=0,
  #                      max_p=4, max_q=3, trace=True,error_action="ignore",seasonal=False)
		# 	print(stepwise_model.summary())
		# else:
		# 	stepwise_model = auto_arima(self.series_data, start_p=0, start_q=0,
  #                      max_p=3, max_q=3, error_action="ignore",seasonal=False)
		# stepwise_model.fit(self.train_data)
		# prediction = stepwise_model.predict(n_periods=1)[0]

		model = ARIMA(self.train_data, (P,D,Q))
		model_fit = model.fit(disp=0)
		prediction = model_fit.forecast()[0]
		return prediction

	def ReturnAllPredicted(self, P = 1, D = 1, Q = 1):
		#in a for loop, predict values using ARIMA model
		self.predictions = list()
		for timepoint in range(self.test_size):
			ActualValue =  self.test_data[timepoint]
			#forcast value
			Prediction = self.StartARIMAForecasting(P,D,Q, timepoint)    
			#add it in the list
			if(self.prediction_type == "oc"):				
				self.predictions.append(np.sign(Prediction-self.data[self.x_train_start+self.train_size+timepoint,self.oc_col[0]]))
			elif(self.prediction_type == "cc"):
				self.predictions.append(np.sign(Prediction-self.data[self.x_train_start+self.train_size+timepoint-1,self.oc_col[1]]))
			np.append(self.train_data, ActualValue)

		cnf_mat_test 	= self.GenerateCnfMatrix(self.predictions, self.test_result)
		cnf_mat_train 	= self.GenerateCnfMatrix(self.train_result, self.train_result)
		actual_dist 	= self.ComputeDistribution(self.train_result, self.test_result)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, "ARIMA MODEL", actual_dist,1)

	def PlotData(self):
		#print(self.series_data.mean())
		#print(self.series_data.var())
		#pyplot.plot(self.series_data.rolling(50).var())
		#pyplot.plot(self.series_data)
		#pyplot.title("Reliance Inds. Lim.")
		#pyplot.xlabel("Date")
		#pyplot.ylabel("Variance over rolling window of 50 days")

		#pyplot.show()
		period = 1
		diff = self.series_data.diff(periods=period)
		diff = diff[period:]
		pyplot.plot(diff.rolling(2).var())
		pyplot.show()
		pyplot.plot(diff)
		pyplot.title("NIFTY 50 S&P " + str(period) +" Difference")
		pyplot.xlabel("Date")
		pyplot.ylabel("Difference value")
		pyplot.show()
		plot_acf(diff, lags=200)
		pyplot.title("NIFTY 50 S&P ACF")
		pyplot.xlabel("Lags")
		pyplot.ylabel("ACF value")
		pyplot.show()
		plot_acf(diff, lags=200)
		pyplot.title("NIFTY 50 S&P PACF")
		pyplot.xlabel("Lags")
		pyplot.ylabel("PACF value")
		pyplot.show()

	def DickeyFuller(self):
		X = self.series_data.values
		result = adfuller(X)
		print (0, result[1])
		for period in [1,2,3]:
			diff = self.series_data.diff(periods=period)
			diff = diff[period:]
			X = diff.values
			result = adfuller(X)
			print (period, result[1])


