"""
utils.py

 - Common helper functions:
     -  load/save JSON or YAML configurations
     -  inject random noise into parameter sets
     -  logging and progress-bar utilities
     -  date/time stamping for output files
"""

from os import environ as ENV
import json
import os, sys, time
from contextlib import contextmanager

from tqdm import tqdm

TEMP = ENV['HOME'] + '/.cache/'
DATA = 'data/'

@contextmanager
def log():
	os.makedirs(DATA, exist_ok=True)
	timestamp = time.strftime('%Y%m%d-%H%M%S')
	logfile = f'{DATA}/{timestamp}.log'
	with open(logfile, 'wt+') as f:
		_o = sys.stdout
		_e = sys.stderr
		sys.stdout = f
		sys.stderr = f

		try: yield
		finally:
			sys.stdout = _o
			sys.stderr = _e
