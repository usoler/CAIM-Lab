# Imports
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

import argparse

# Global variables
a = 1

# Functions
def getArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--log', action='store_true', default=False, help='Show log plot')
	
	return parser.parse_args()


def orderByFrequency(line):
	return float(line.strip().split(',')[0])


def getFrequencyValues(file):
	print(f'***************************')
	print(f'Getting Frequency values...')
	print(f'***************************')

	frequencyValues = []
	logFrequencyValues = []

	lines = file.readlines()
	lines = lines[:len(lines)-4]
	lines.sort(reverse=True, key=orderByFrequency)

	for line in lines:
		frequency = float(line.strip().split(',')[0])
		frequencyValues.append(frequency)
		logFrequencyValues.append(np.log(frequency))

	return frequencyValues,logFrequencyValues


def generateRankValues(frequencyValues):
	print(f'*************************')
	print(f'Generating Rank Values...')
	print(f'*************************')

	return range(1, len(frequencyValues)+1)


def zipf(rank, b, c):
	return c/(rank+b)**a


def getOptimalValues(rankValues, frequencyValues):
	print(f'*************************')
	print(f'Getting Optimal Values...')
	print(f'*************************')

	# popt -> optimal values
	# pcov -> estimated covariance of popt
	popt, pcov = curve_fit(zipf, rankValues, frequencyValues)
	b = popt[0]
	c = popt[1]

	return b,c


def getZipfValuesToPlot(b, c, rankValues):
	print(f'******************************')
	print(f'Getting Zipf Values To Plot...')
	print(f'******************************')

	zipfValues = map(lambda value: zipf(value, b, c), rankValues)
	logZipfValues = map(lambda value: np.log(value), zipfValues)

	return zipfValues, logZipfValues


def showPlot(frequencyValues, zipfValues):
	print(f'********************')
	print(f'Showing Zipf Plot...')
	print(f'********************')

	plt.plot(np.log(range(1,len(frequencyValues)+1)), frequencyValues, color='b', label='Frequencies', linewidth=2)
	plt.plot(np.log(range(1,len(frequencyValues)+1)), list(zipfValues), color='r', label='Zipf', linewidth=2, linestyle='--')
	plt.legend()
	plt.xlabel('Word-s Rank')
	plt.ylabel('Word-s Frequency')
	plt.show()


def showLogPlot(logFrequencyValues, logZipfValues):
	print(f'************************')
	print(f'Showing Zipf Log Plot...')
	print(f'************************')

	plt.plot(np.log(range(1,len(logFrequencyValues)+1)), reversed(logFrequencyValues), color='b', label='Log Frequencies', linewidth=2)
	plt.plot(np.log(range(1,len(logFrequencyValues)+1)), reversed(list(logZipfValues)), color='r', label='Log Zipf', linewidth=2, linestyle='--')
	plt.legend()
	plt.xlabel('Log Word-s Rank')
	plt.ylabel('Log Word-s Frequency')
	plt.show()


# Main
if __name__ == '__main__':
	args = getArguments()

	file = open('output.txt', 'r')

	frequencyValues, logFrequencyValues = getFrequencyValues(file)

	file.close()

	rankValues = generateRankValues(frequencyValues)

	b, c = getOptimalValues(rankValues, frequencyValues)
	print(f'Optimal values: b={b} c={c}')
	print()

	zipfValues, logZipfValues = getZipfValuesToPlot(b, c, rankValues)

	if args.log:
		showLogPlot(logFrequencyValues, logZipfValues)
	else:
		showPlot(frequencyValues, zipfValues)
