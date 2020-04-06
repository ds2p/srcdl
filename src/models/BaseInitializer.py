"""
Copyright (c) 2020 CRISP

The abstract class for Initializer

:author: Andrew H. Song
"""

from abc import ABCMeta, abstractmethod

class BaseInitializer(metaclass=ABCMeta):
	def __init__(self):
		pass

	@abstractmethod
	def initialize(self, y_train, d):
		pass
