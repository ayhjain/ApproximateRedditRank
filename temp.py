import csv, sys, math
import numpy as np
import time
import matplotlib.pyplot as plt
from dataParser import parseCSV, x, y
from sklearn import linear_model

#LAMBDA = 1
kFold = 5
trainingDataPortion = 0.7
noOfTrainingEntries = 27750
noOfFeatures = 60 # 61 -url -shares +w0
noOfTestEntries = 11894
wLSE = np.zeros( shape = [noOfFeatures, 1] )
wGD = np.ones( shape = [noOfFeatures, 1] )



def initialize(filename, entriesToProcess) :
	
	parseCSV(filename, entriesToProcess)
	return x[:noOfTrainingEntries,:], y[:noOfTrainingEntries,:]
#	print x.shape
#	print y

def validate(X, Y, weight) :
	n, m = X.shape
#	print "\n"
#	print n, m
	e_avg = 0
	for i in range(n) :
		y_est = np.dot(X[i,:], weight[:,0])
		e = math.pow((Y[i] - y_est), 2)/n
#		print Y[i], y_est
		e_avg += e
	return e_avg


	
	
def learningRidgeRegression(X , Y) :
	
	global wLSE
	#global LAMBDA
#	print X.shape
#	print Y.shape
	
	# k-Fold Validation for getting best value for lambda
	plot_e_test_sum = []
	plot_e_train_sum = []
	
	for LAMBDA in range(1,200,5) : 
		print "******************************************************************"
		print "Evaluation for lamba ={0}".format(LAMBDA)
		e_test_sum = e_train_sum = 0
		
		for i in range(kFold):
			
			#get sliced X and Y after removing the kth Block
		#	print "For i = {0} and k-Fold = {1}".format(i, kFold)
			# range(s,e) denotes the validation portion
			s = i * (noOfTrainingEntries / kFold)
			e = s + (noOfTrainingEntries / kFold)
			
			x_training = np.concatenate((X[:s,:],X[e:,:]), axis=0)
			y_training = np.concatenate((Y[:s],Y[e:]), axis=0)
			
			x_test = X[s:e,:]
			y_test = Y[s:e]
			
		#	print x_training.shape
		#	print x_test.shape
			
			#Applying Ridge Regularization to Linear Regression Model
			# w = (X'.X + Lamda I)^-1 . X'.Y
			regr = linear_model.LinearRegression()
			regr.fit(x_training, y_training)
			y_est = regr.predict(x_test)
			n,m =y_est.shape
			e_avg = 0
			for i in range(n) :
				#print y_est[i], y_test[i]
				e = math.pow((y_test[i] - y_est[i]), 2)/n
				e_avg += e
			
		print
		print "Avg. Testing Error = {0}".format(e_avg)
		print "******************************************************************"
		print "\n"
	'''for i in range(noOfFeatures) :
		print ("wLSE({0}) = {1}".format(i, wLSE[i,0]) )
	print "\n\n"
	
	plt.plot(plot_e_train_sum, range(1,200,5), 'ro', plot_e_test_sum, range(1,200,5), 'bs)
	'''

	
def printWeights(weight) :
	
	for i in range(noOfFeatures) :
		print ("weight({0}) = {1}".format(i, weight[i,0]) )
	print "\n\n"

		
		
		
if __name__ == "__main__" : 
	
	global wLSE
	global wGD
	global noOfTestEntries
	global noOfTrainingEntries

	filename = sys.argv[1]
	entriesToProcess = int(sys.argv[2])
	kFold = int(sys.argv[3])
	
	noOfTrainingEntries = int(trainingDataPortion *  entriesToProcess)
	noOfTestEntries = entriesToProcess - noOfTrainingEntries
	
	print filename, noOfTestEntries, noOfTrainingEntries, kFold
	#	print entriesToProcess
	
	X_Training , Y_Training = initialize(filename, entriesToProcess)

	learningRidgeRegression(X_Training , Y_Training)
	print np.average(y)
	print "Bye"