import csv, sys
import numpy as np
import time

totalDataRows = 40000
noOfFeatures = 60 # 61 -url -shares +w0
	
#x = np.ones(shape=[totalDataRows,noOfFeatures])
#y = np.zeros(shape=[totalDataRows, 1])



def normalize(x):
	x_max = np.amax(x, axis =  0)
	x_min = np.amin(x, axis = 0)
	x = (x[:,]-x_min)/x_max	
	return x

def parseCSV(filename, entriesToProcess):
	
	x = np.ones(shape=[totalDataRows,noOfFeatures])
	y = np.zeros(shape=[totalDataRows, 1])
	
	with open(filename) as inputs:
		reader = csv.DictReader(inputs)

		i = 0
		for row in reader :
			x[i][0] = 1.0
			j = 1
			for key, value in row.items() :
#				print j, key, value
				if (key != 'url') :
#					print "array[%d][%d] = %f", i, j,value
					try:
						if (key != ' shares') : 
							#to remove
							if j>49: continue
							x[i][j] = float(value)
							j += 1
						else : 
							y[i][0] = float(value)
					except:
						print value, i, j
						# control shouldn't be in this portion of Code
			i += 1
			if (i>=int(entriesToProcess)): break
	
	print ("Done parsing!")
	x = normalize(x)
	print x
	return x,y


if __name__ == "__main__" : 
	
	filename = sys.argv[1]
	entriesToProcess = sys.argv[2]
#	print entriesToProcess
	
	parseCSV(filename, entriesToProcess)
	
#	print x[0]
	print y
	print "Bye"