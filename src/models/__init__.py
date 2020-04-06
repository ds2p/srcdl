"""
Copyright (c) 2020 CRISP

generator for CSC and CDL

:author: Andrew H. Song
"""

import numpy as np

from .BaseCSC import BaseCSC
from .BaseCDL import BaseCDL
from .BaseInitializer import BaseInitializer
from .COMP import COMP
from .CKSVD import CKSVD

def generateCSC(dlen, error_tol, sparsity_tol, pflag, csc_type):
	"""
	Construct CSC object
	"""

	if csc_type=='comp':
		print("COMP selected")
		return COMP(dlen, error_tol, sparsity_tol, pflag)
	else:
		pass

def generateCDL(dlen, numOfelements, cdl_type):
	"""
	Construct CDL object
	"""

	if cdl_type=='cksvd':
		print("CKSVD selected")
		return CKSVD(dlen, numOfelements)
	else:
		pass
