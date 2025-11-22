"""
utils.py

 - Common helper functions:
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
def pushd(new_dir):
	previous_dir = os.getcwd()
	os.makedirs(new_dir, exist_ok=True)
	os.chdir(new_dir)
	try: yield
	finally:
		os.chdir(previous_dir)


@contextmanager
def log():
	timestamp = time.strftime('%Y%m%d-%H%M%S')
	with pushd(f'{DATA}/{timestamp}'):
		with open('output.log', 'wt+') as f:
			_o = sys.stdout
			_e = sys.stderr
			sys.stdout = f
			sys.stderr = f
			try: yield
			finally:
				sys.stdout = _o
				sys.stderr = _e
