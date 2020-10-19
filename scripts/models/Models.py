import numpy as np
import math

# importing models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

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

class Models:

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
			self.features_col = list(range(9,120))
			self.oc_col = [7,8]
		elif(prediction_type[:11] == "Close/Close"):
			self.prediction_type = "cc"
			nself.features_col = list(range(8,119))
			self.oc_col = [2,5]

		self.data = np.genfromtxt(self.filepath+self.filename ,delimiter = ',' , autostrip = True)	
		
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
		models = [self.RunLDA, self.RunLAS, self.RunRIDGE, self.RunKNN, self.RunNB, self.RunRF, self.RunSVM]
		#models = [self.RunSVM] 
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

	def Impute(self):
		self.x_train			= SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(self.x_train)		
		self.x_test			    = SimpleImputer(missing_values = np.nan, strategy = 'mean').fit_transform(self.x_test)		
	
	def FeatureSelection(self, model):
		self.relevant_features = RFE(model, n_features_to_select=int(len(self.features_col)/4), step=1000, verbose=0).fit(self.x_train, self.y_train).support_

	def RunLAS(self):
		name = "Lasso"
		model 				= Lasso()
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_train 		= np.sign(model.predict(x_train_))
		pred_test 		= np.sign(model.predict(x_test_))
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)

	def RunRIDGE(self):
		name = "Ridge"
		model 				= Ridge()
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_train 		= np.sign(model.predict(x_train_))
		pred_test 		= np.sign(model.predict(x_test_))
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)


	def RunLDA(self):
		name = "LDA"
		model 				= LinearDiscriminantAnalysis()
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_test 		= model.predict(x_test_)
		pred_train 	= model.predict(x_train_)
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)
	
	def RunNB(self):
		name = "Naive Bayes"
		model 				= GaussianNB()
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_test 		= model.predict(x_test_)
		pred_train 	= model.predict(x_train_)
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)

	def RunKNN(self):
		name = "KNN"
		model 				= KNeighborsClassifier()
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_test 		= model.predict(x_test_)
		pred_train 	= model.predict(x_train_)
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)

	def RunRF(self):
		#try changing other hyper-parameters like min_sample_split, it would boost the result
		name = "Random Forest"
		param_grid =  [{ 'n_estimators': [50*i for i in range(1,8)]}]
		model = RandomForestClassifier()
		clf = model_selection.GridSearchCV(model, param_grid, scoring=None, cv = TimeSeriesSplit(n_splits = 5))
		clf.fit(self.x_train, self.y_train)
		x = (clf.best_params_ )
		model.set_params(**x)
		self.FeatureSelection(model)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_test 		= model.predict(x_test_)
		pred_train 	= model.predict(x_train_)
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)

	def RunSVM(self):
		name = "Support Vector Machine"
		G_range_ = [0.001,0.005,0.01,0.05,0.1,0.15,0.25,0.28,0.75,1]
		C_range  = [0.8,1,2,6,7,8,10,50,100,1000,2500]
		G_range  = [1.0/i for i in G_range_]
		param_grid =  [{ 'C': C_range, 'gamma' : G_range}]
		model 			= SVC(kernel = 'rbf')
		clf = model_selection.GridSearchCV(model, param_grid)
		clf.fit(self.x_train, self.y_train)
		x = (clf.best_params_ )
		model.set_params(**x)
		model_for_feature_selction = RandomForestClassifier()
		self.FeatureSelection(model_for_feature_selction)
		x_train_			= self.x_train[:,self.relevant_features]
		x_test_				= self.x_test[:,self.relevant_features]
		model.fit(x_train_, self.y_train)
		pred_test 		= model.predict(x_test_)
		pred_train 	= model.predict(x_train_)
		
		cnf_mat_test 	= self.GenerateCnfMatrix(pred_test, self.y_test)
		cnf_mat_train 	= self.GenerateCnfMatrix(pred_train, self.y_train)
		actual_dist 	= self.ComputeDistribution(self.y_train, self.y_test)	
		accuracy 		= self.ComputeAccuracy(cnf_mat_test, cnf_mat_train, name, actual_dist,1)


	
		

			
