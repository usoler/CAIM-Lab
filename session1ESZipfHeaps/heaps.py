# Imports
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

import argparse

# Functions
def getArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--log', action='store_true', default=False, help='Show log plot')
	
	return parser.parse_args()


def getTotalPalabrasAndPalabrasDistintas(filenameList):
	print(f'***************************')
	print(f'Getting total words and different words...')
	print(f'***************************')

	totalPalabras = []
	palabrasDistintas = []
	
	for filename in filenameList:
		n = 0
		d = 0

		file = open(filename, 'r')

		lines = file.readlines()
		lines = lines[:len(lines)-4]
		
		for line in lines:
			frenquency = float(line.strip().split(',')[0])
			n = n + frenquency
			d = d + 1
	
		totalPalabras.append(n)
		palabrasDistintas.append(d)

		file.close()

	return totalPalabras, palabrasDistintas


def getLogValues(totalPalabras, palabrasDistintas):
	print(f'***************************')
	print(f'Getting log total words and log different words values...')
	print(f'***************************')

	logTotalPalabras = map(lambda value: np.log(value), totalPalabras)
	logPalabrasDistintas = map(lambda value: np.log(value), palabrasDistintas)

	return logTotalPalabras, logPalabrasDistintas


def heaps(N, k, beta):
	return k*(N**beta)


def getOptimalValues(totalPalabras, palabrasDistintas):
	print(f'*************************')
	print(f'Getting Optimal Values...')
	print(f'*************************')

	# popt -> optimal values
	# pcov -> estimated covariance of popt
	popt, pcov = curve_fit(heaps, totalPalabras, palabrasDistintas)
	k = popt[0]
	beta = popt[1]

	return k, beta


def getHeapsValuesToPlot(k, beta, totalPalabras):
	print(f'******************************')
	print(f'Getting Heaps Values To Plot...')
	print(f'******************************')

	heapsValues = map(lambda value: heaps(value, k, beta), totalPalabras)
	logHeapsfValues = map(lambda value: np.log(value), heapsValues)

	return heapsValues, logHeapsfValues


def showPlot(totalPalabras, palabrasDistintas, heapsValues):
	print(f'********************')
	print(f'Showing Heaps Plot...')
	print(f'********************')

	plt.plot(totalPalabras, palabrasDistintas, color='b', label='Real', linewidth=2)
	plt.plot(totalPalabras, list(heapsValues), color='r', label='Heaps', linewidth=2, linestyle='--')
	plt.legend()
	plt.xlabel('Total words')
	plt.ylabel('Different words')
	plt.show()


def showLogPlot(logTotalPalabras, logPalabrasDistintas, logHeapsValues):
	print(f'************************')
	print(f'Showing Heaps Log Plot...')
	print(f'************************')

	logTotalPalabras = list(logTotalPalabras)
	plt.plot(logTotalPalabras, list(logPalabrasDistintas), color='b', label='Log real', linewidth=2)
	plt.plot(logTotalPalabras, list(logHeapsValues), color='r', label='Log Heaps', linewidth=2, linestyle='--')
	plt.legend()
	plt.xlabel('Log total words')
	plt.ylabel('Log different words')
	plt.show()


# Main
if __name__ == '__main__':
	args = getArguments()

	filenameList = ["output.txt", "output2.txt", "output3.txt", "output4.txt", "output5.txt", "output6.txt"]

	totalPalabras, palabrasDistintas = getTotalPalabrasAndPalabrasDistintas(filenameList)

	logTotalPalabras, logPalabrasDistintas = getLogValues(totalPalabras, palabrasDistintas)

	k, beta = getOptimalValues(totalPalabras, palabrasDistintas)
	print(f'k={k}, beta={beta}')

	heapsValues, logHeapsValues = getHeapsValuesToPlot(k, beta, totalPalabras)

	if args.log:
		showLogPlot(logTotalPalabras, logPalabrasDistintas, logHeapsValues)
	else:
		showPlot(totalPalabras, palabrasDistintas, heapsValues)
