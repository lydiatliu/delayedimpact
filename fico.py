"""Reading FICO Data"""

from __future__ import print_function
import argparse
import math
import numpy as np
import pandas as pd
import pylab

DATA_DIR='../'

perf = 'transrisk_performance_by_race_ssa.csv'
cdf_by_race = 'transrisk_cdf_by_race_ssa.csv'
overall = 'totals.csv'

files = dict(cdf_by_race = cdf_by_race,
             performance_by_race = perf,
             overview= overall 
)

def cleanup_frame(frame):
  """Make the columns have better names, and ordered in a better order"""
  frame = frame.rename(columns={'Non- Hispanic white': 'White'})
  frame = frame.reindex(columns=['Asian', 'White', 'Hispanic', 'Black'])
  #frame = frame[['Asian', 'White', 'Black']]
  return frame

def read_totals(data_dir=DATA_DIR):
  """Read the total number of people of each race"""
  frame = cleanup_frame(pd.DataFrame.from_csv(data_dir+files['overview']))
  return {r:frame[r]['SSA'] for r in frame.columns}

def convert_percentiles(idx):
  pdf = [(300, 2.1),
         (350, 4.2),
         (400, 5.4),
         (450, 6.5),
         (500, 7.9),
         (550, 9.6),
         (600, 12.0),
         (650, 13.8),
         (700, 17.0),
         (750, 15.8),
         (800, 5.7),
         (850, 0),
         ]
  def convert_one(x):
    partial = 0
    for ((v, s), (v2, _)) in zip(pdf, pdf[1:]):
      if partial + s >= x:
        return v + (v2-v) * (x - partial) / s
      partial += s
  return np.array(list(map(convert_one, idx)))

def parse_data(data_dir=DATA_DIR, filenames=None):
  """Parse sqf data set."""
  if filenames is None:
    filenames = [files['cdf_by_race'], files['performance_by_race']]

  cdfs = cleanup_frame(pd.DataFrame.from_csv(data_dir+filenames[0]))
  performance = 100-cleanup_frame(pd.DataFrame.from_csv(data_dir+filenames[1]))
  return (cdfs/100., performance/100.)

def get_FICO_data(data_dir=DATA_DIR,do_convert_percentiles=True):
  data_pair = parse_data(data_dir)
  totals = read_totals(data_dir)

  if do_convert_percentiles:
    for v in data_pair:
      v.index = convert_percentiles(v.index)
  cdfs = data_pair[0]
  performance = data_pair[1]
  return cdfs, performance, totals