"""
.. module:: CountWords

CountWords
*************

:Description: CountWords

	Generates a list with the counts and the words in the 'text' field of the documents in an index

:Authors: bejar
	

:Version: 

:Created on: 04/07/2017 11:58 

"""

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch.exceptions import NotFoundError, TransportError
from string import punctuation, digits

import argparse

__author__ = 'bejar'

def filterNonLetterCharacters(word):
	for letter in word:
		if (letter in punctuation and not letter == "'") or (letter in digits):
			return False

	return True


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--index', default=None, required=True, help='Index to search')
	parser.add_argument('--alpha', action='store_true', default=False, help='Sort words alphabetically')
	args = parser.parse_args()

	index = args.index

	try:
		client = Elasticsearch()
		voc = {}
		sc = scan(client, index=index, query={"query" : {"match_all": {}}})
		for s in sc:
			try:
				tv = client.termvectors(index=index, id=s['_id'], fields=['text'])
				if 'text' in tv['term_vectors']:
					for t in tv['term_vectors']['text']['terms']:
						if t in voc:
							voc[t] += tv['term_vectors']['text']['terms'][t]['term_freq']
						else:
							voc[t] = tv['term_vectors']['text']['terms'][t]['term_freq']
			except TransportError:
				pass
		lpal = []

		for v in voc:
			lpal.append((v.encode("utf-8", "ignore"), voc[v]))

		numOfWords = 0
		for pal, cnt in sorted(lpal, key=lambda x: x[0 if args.alpha else 1]):
			if filterNonLetterCharacters(pal.decode("utf-8")):
				print(f'{cnt}, {pal.decode("utf-8")}')
				numOfWords += 1
		print('--------------------')
		print(f'{numOfWords} Words')
		print()
	except NotFoundError:
		print(f'Index {index} does not exists')