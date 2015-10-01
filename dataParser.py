import csv, sys
import numpy as np
import time

	
def normalize(x):
	x_min = np.amin(x, axis = 0)
	x = x - x_min
	x[:,0] = 1.0
	x_max = np.amax(x, axis =  0)
	x = x/x_max
	return x

def parseCSV(filename, entriesToProcess, noOfFeatures, normalize_flag = False):
	
	x = np.ones(shape=[entriesToProcess,noOfFeatures])
	y = np.zeros(shape=[entriesToProcess, 1])
	
	with open(filename) as inputs:
		reader = csv.DictReader(inputs)

		i = 0
		for row in reader :
			x[i,0] = 1.0
			j = 1			# for adding w0 value to the matrix
			
			for key, value in row.items() :
#				print j, key, value
				if (key not in ['url', ' timedelta']) :
#					print "array[%d][%d] = %f", i, j,value
					try:
						if (key not in [' shares', 'score']) : 
							x[i,j] = float(value)
							j += 1
						else : 
							y[i,0] = float(value)
					except:
						print value, i, j, key
						# control shouldn't be in this portion of Code
			i += 1
			if (i>=int(entriesToProcess)): break
	
	print ("Done parsing!")
	
	if (normalize_flag):
		x = normalize(x)

	return x,y


if __name__ == "__main__" : 
	
	filename = sys.argv[1]
	entriesToProcess = sys.argv[2]
	noOfFeatures = sys.argv[3]
	
	parseCSV(filename, entriesToProcess, noOfFeatures)
	
#	print x[0]
	print y
	print "Bye"