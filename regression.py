import csv, sys, math
import numpy as np
import time
import matplotlib.pyplot as plt
from dataParser import parseCSV

#LAMBDA = 1
kFold = 5
trainingDataPortion = 0.7
noOfTrainingEntries = 27750
noOfFeatures = 59 # 61 -url -shares -timedelta +w0
noOfTestEntries = 11894
wLSE = np.zeros( shape = [noOfFeatures, 1] )
wGD = np.random.uniform(0,1,[noOfFeatures,1])



def initialize(filename, entriesToProcess) :
	x ,y = parseCSV(filename, entriesToProcess, noOfFeatures, False)
	return x[:noOfTrainingEntries,:], y[:noOfTrainingEntries,:]
#	print x.shape
#	print y
	


def validate(X, Y, weight) :
	n,m = X.shape
	e_avg = 0
	y_est = np.dot(X,weight)
	e_avg = np.sum((Y-y_est) ** 2) / (2*n)
	return e_avg


	
	
def learningRidgeRegression(X , Y) :
	
	global wLSE
	#global LAMBDA
#	print X.shape
#	print Y.shape
	
	# k-Fold Validation for getting best value for lambda
	plot_e_test_sum = []
	plot_e_train_sum = []
	
	for LAMBDA in range(1,6,5) : 
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
			# w = (X'.X + Lamda.I)^-1 . X'.Y
			
			temp = np.dot(x_training.T, x_training)
		#	print temp
			temp = np.add(temp, LAMBDA*np.matrix(np.identity(noOfFeatures)))
		#	print np.linalg.det(temp)
			wLSE = np.dot(np.dot(np.linalg.inv(temp), x_training.T), y_training)
		#	printWeights(wLSE)
			e_train = validate(x_training, y_training, wLSE)
		#	print "Training error = {0}".format(e_train)
			e_test = validate(x_test, y_test, wLSE)
		#	print "Testing error = {0}".format(e_test)
			
			e_test_sum += e_test
			e_train_sum += e_train
		
		np.append(plot_e_test_sum,e_test_sum/kFold)
		np.append(plot_e_train_sum, e_train_sum/kFold)
		print
		print "Avg. Training Error = {0}\tAvg. Testing Error = {1}".format(e_train_sum/kFold , e_test_sum/kFold)
		print "******************************************************************"
		print "\n"
	'''for i in range(noOfFeatures) :
		print ("wLSE({0}) = {1}".format(i, wLSE[i,0]) )
	print "\n\n"
	'''
	plt.plot(plot_e_train_sum, range(1,200,5), 'ro')
	'''
	, plot_e_test_sum, range(1,200,5), 'bs')
	'''



	
	
def learningGradientDescent(X, Y, noOfIterations) :
	
	global wGD
	
	
	plot_e_test_sum = []
	plot_e_train_sum = []
	
	e_test_sum = e_train_sum = 0
	for iter in range(1000, noOfIterations, 500):
		
		e_test_sum = e_train_sum = 0
		
		for i in range(kFold):
			
			#get sliced X and Y after removing the kth Block
			# range(s,e) denotes the validation portion
			s = i * (noOfTrainingEntries / kFold)
			e = s + (noOfTrainingEntries / kFold)
			
			x_training = np.concatenate((X[:s,:],X[e:,:]), axis=0)
			y_training = np.concatenate((Y[:s],Y[e:]), axis=0)
			n, m = x_training.shape
			
			x_test = X[s:e,:]
			y_test = Y[s:e]
			
			wGD = np.random.uniform(0,1,[noOfFeatures,1])
			
			cost = 0.0
			for j in range(iter):
				alfa = 0.000000000001#/((j+1)**2)
				hypothesis = np.dot(x_training, wGD)
				loss = hypothesis - y_training
				cost = np.sum(loss ** 2) / (2*n)
			#	print("Iteration %d | Cost: %f" % (j, cost))
				gradient = np.dot(x_training.T, loss) / n
				wGD -= alfa*gradient
			
			e_train_sum += (validate(x_training, y_training, wGD)/kFold)
			e_test_sum += (validate(x_test, y_test, wGD)/kFold)
			
		print "Iteration = {0}\tAvg. Training Error = {1}\tAvg. Testing Error = {2}".format(iter, e_train_sum, e_test_sum)
		print "******************************************************************\n"
		
		plot_e_train_sum.append(e_train_sum)
		plot_e_test_sum.append(e_test_sum)
	
	plot_e_test_sum = np.array(plot_e_test_sum)
	plot_e_train_sum = np.array(plot_e_train_sum)
	
	
	
	
	
	plt.plot(range(1000, noOfIterations, 500), plot_e_train_sum, 'ro', label = "Training Error")
	plt.plot(range(1000, noOfIterations, 500), plot_e_test_sum, 'bs', label = "Testing Error")
	plt.ylabel("Error")
	plt.xlabel("No. of Iterations")
	plt.legend()
	plt.show()
	
	


	
	
def printWeights(weight) :
	
	for i in range(noOfFeatures) :
		print ("weight({0}) = {1}".format(i, weight[i,0]) )
	print "\n\n"






if __name__ == "__main__" : 
	
	global wLSE
	global wGD
	global noOfTestEntries
	global noOfTrainingEntries
	global noOfFeatures
	
	filename = sys.argv[1]
	entriesToProcess = int(sys.argv[2])
	noOfFeatures = int(sys.argv[3])
	kFold = int(sys.argv[4])
	
	noOfTrainingEntries = int(trainingDataPortion *  entriesToProcess)
	noOfTestEntries = entriesToProcess - noOfTrainingEntries
	
	print filename, noOfTestEntries, noOfTrainingEntries, kFold
	#	print entriesToProcess
	
	X_Training , Y_Training = initialize(filename, entriesToProcess)
	
#	learningRidgeRegression(X_Training , Y_Training)
#	printWeights(wLSE)
		
	learningGradientDescent(X_Training, Y_Training, 10001)
#	printWeights(wGD)
	
	
	######################
	#
	#	Code to test for testing
	#
	######################
	
	print "Bye"