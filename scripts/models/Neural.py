import numpy as np
import math

# importing models
from sklearn.ensemble import RandomForestClassifier

#feature selection
from sklearn.feature_selection import RFE

# importing helper functions
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer,log_loss, f1_score, average_precision_score,explained_variance_score, log_loss
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.decomposition import PCA

#for neural networks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

class NeuralNetworkModels:

	def __init__(self, filename, train_test_split, features, prediction_type, prediction_col):
		train_test_split = list(map(int, train_test_split))
		self.filepath = "indicators/"
		self.filename = filename
		self.x_train_start = train_test_split[0]
		self.train_size = train_test_split[1]
		self.test_size = train_test_split[2]
		self.window_size = train_test_split[3]
		self.no_windows = train_test_split[4]
		self.prediction_col = prediction_col
		if(prediction_type[:10] == "Open/Close"):
			self.prediction_type = "oc"
			self.oc_col = [7,8]
			self.features_col = list(range(9,120))
			self.abs_prediction_col = 8
		elif(prediction_type[:11] == "Close/Close"):
			self.prediction_type = "cc"
			self.oc_col = [2,5]
			self.features_col = list(range(8,119))
			self.abs_prediction_col = 5
		self.data = np.genfromtxt(self.filepath+self.filename ,delimiter = ',' , autostrip = True)	
		self.train_result = self.data[self.x_train_start : self.x_train_start+self.train_size, self.abs_prediction_col]
		self.test_result = self.data[self.x_train_start+self.train_size:self.x_train_start+self.train_size+self.test_size,self.abs_prediction_col]

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
			self.accuracy = self.accuracy + acc_test[0]
			print ('\n')
		return (per_test, per_train, acc_test, acc_train)

	def RunAll(self):
		models = [self.RunLSTM]
		for model in models:
			self.accuracy = 0.0
			#for basic models without validation
			for i in range(0, self.no_windows):
				start = self.x_train_start + i*self.window_size
				self.x_train = self.data[ start : start + self.train_size, self.features_col]
				self.y_train = self.data[ start : start + self.train_size, self.prediction_col]

				start = start + self.train_size
				self.x_test = self.data[ start : start + self.test_size, self.features_col]
				self.y_test = self.data[ start : start + self.test_size, self.prediction_col]
				self.Impute()
				model()
			print("Average accuracy : " + str(self.accuracy/self.no_windows))
			return

	def Impute(self):
		self.x_train			= SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(self.x_train)		
		self.x_test			    = SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(self.x_test)		
	
	def FeatureSelection(self, model):
		self.relevant_features = RFE(model, n_features_to_select=int(len(self.features_col)/4), step=1000, verbose=0).fit(self.x_train, self.y_train).support_
	
	def RunLSTM(self):
		name = "LSTM"
		model_for_feature_selction = RandomForestClassifier()
		self.FeatureSelection(model_for_feature_selction)
		x_train			= self.x_train[:,0:5]
		x_test				= self.x_test[:,0:5]
		drop = 0.1
		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		model = Sequential()
		model.add(LSTM(300, return_sequences=True, input_shape=(x_train.shape[1], 1)))
		model.add(LSTM(300))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(x_train, self.train_result, epochs=100, batch_size=1, verbose=2)
		pred_train_abs = model.predict(x_train)[:,0]
		pred_test_abs = model.predict(x_test)[:,0]
		print(pred_test_abs)
		print(pred_train_abs)
		y = self.data[self.x_train_start:self.x_train_start+self.train_size,self.oc_col[0]]
		print(y.shape)
		if(self.prediction_type == "oc"):				
			pred_train = np.sign(pred_train_abs-self.data[self.x_train_start:self.x_train_start+self.train_size,self.oc_col[0]])
			pred_test = np.sign(pred_test_abs-self.data[self.x_train_start+self.train_size:self.x_train_start+self.train_size+self.test_size,self.oc_col[0]])
		elif(self.prediction_type == "cc"):
			pred_train = np.sign(pred_train_abs-self.data[self.x_train_start-1:self.x_train_start+self.train_size-1,self.oc_col[1]])
			pred_test = np.sign(pred_test_abs-self.data[self.x_train_start+self.train_size-1:self.x_train_start+self.train_size+self.test_size-1,self.oc_col[1]])
		
		print(pred_train)
		print(pred_test)
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)
		exit()


	
		

			
