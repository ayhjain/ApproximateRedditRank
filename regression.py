import csv, sys
import numpy as np
import time
from dataParser import parseCSV, x, y

LAMBDA = 2
noOfInstances = 200
noOfFeatures = 60 # 61 -url -shares +w0
w = np.zeros( shape = [noOfFeatures, 1] )




if __name__ == "__main__" : 
	
	filename = sys.argv[1]
	entriesToProcess = sys.argv[2]
#	print entriesToProcess
	
	parseCSV(filename, entriesToProcess)
	print x.shape
	print y
	
	#Applying Ridge Regularization to Linear Regression Model
	# w = (X'.X + Lamda I)^-1 . X'.Y
	temp = np.dot(x.T, x)
	temp = np.add(temp, LAMBDA*np.matrix(np.identity(noOfFeatures)))
#	print np.linalg.det(temp)
	w = np.dot(np.dot(np.linalg.inv(temp), x.T), y)
	
	for i in range(noOfFeatures) :
		print ("w(%d) = %d", i, w[i] )
	
	
	
	
	print "Bye"