"""
Copyright (c) 2020 CRISP

The abstract class for Convolutional Dictionary Learner

:author: Andrew H. Song
"""

from abc import ABCMeta, abstractmethod


class BaseCDL(metaclass=ABCMeta):
	def __init__(self, dlen, numOfelements):
		self.dlen = dlen
		self.numOfelements = numOfelements

	@abstractmethod
	def updateDictionary(self, y_train):
		pass
