import csv, sys
import numpy as np
import time

noOfInstances = 200
noOfFeatures = 60 # 61 -url -shares +w0
	
x = np.zeros(shape=[noOfInstances,noOfFeatures])
y = np.zeros(shape=[noOfInstances, 1])



def parseCSV(filename, entriesToProcess):
	
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


if __name__ == "__main__" : 
	
	filename = sys.argv[1]
	entriesToProcess = sys.argv[2]
#	print entriesToProcess
	
	parseCSV(filename, entriesToProcess)
	
#	print x[0]
#	print y
	print "Bye"