import csv, sys
import numpy as np
import time
from dataParser import parseCSV, x, y

LAMBDA = 2
noOfInstances = 10000
noOfFeatures = 60 # 61 -url -shares +w0
wLSE = np.zeros( shape = [noOfFeatures, 1] )
wGD = np.zeros( shape = [noOfFeatures, 1] )



def initialize(filename, entriesToProcess) :
	
	parseCSV(filename, entriesToProcess)
	print x.shape
	print y
	


def learningRidgeRegression() :
	global wLSE
	#Applying Ridge Regularization to Linear Regression Model
	# w = (X'.X + Lamda I)^-1 . X'.Y
	temp = np.dot(x.T, x)
#	print temp
	temp = np.add(temp, LAMBDA*np.matrix(np.identity(noOfFeatures)))
#	print np.linalg.det(temp)
	wLSE = np.dot(np.dot(np.linalg.inv(temp), x.T), y)
	'''for i in range(noOfFeatures) :
		print ("wLSE({0}) = {1}".format(i, wLSE[i][0]) )
	print "\n\n"
	'''


def learningGradientDescent() :
	
	####Do something here
	global wGD
	i=1
	
	
	
def printWeights(weight) :
	
	for i in range(noOfFeatures) :
		print ("weight({0}) = {1}".format(i, weight[i][0]) )
	print "\n\n"


		
		
		
		
if __name__ == "__main__" : 
	
	global wLSE
	global wGD

	filename = sys.argv[1]
	entriesToProcess = sys.argv[2]
#	print entriesToProcess
	
	initialize(filename, entriesToProcess)
	
	learningRidgeRegression()
	printWeights(wLSE)
		
	learningGradientDescent()
	printWeights(wGD)
	
	
	######################
	#
	#	Code to test for testing
	#
	######################
	
	print "Bye"