"""Reading FICO Data"""

from __future__ import print_function
import numpy as np
import pandas as pd

DATA_DIR = '../'

PERF = 'transrisk_performance_by_race_ssa.csv'
CDF_BY_RACE = 'transrisk_cdf_by_race_ssa.csv'
OVERALL = 'totals.csv'

FILES = dict(cdf_by_race=CDF_BY_RACE,
             performance_by_race=PERF,
             overview=OVERALL
             )


def cleanup_frame(frame):
    """Make the columns have better names, and ordered in a better order"""
    frame = frame.rename(columns={'Non- Hispanic white': 'White'})
    frame = frame.reindex(columns=['Asian', 'White', 'Hispanic', 'Black'])
    return frame


def read_totals(data_dir=DATA_DIR):
    """Read the total number of people of each race"""
    # NOTE: the pandas from_csv is no longer functional, so update the command to read_csv
    #frame = cleanup_frame(pd.DataFrame.from_csv(data_dir + FILES['overview']))
    frame = cleanup_frame(pd.read_csv(data_dir + FILES['overview']))
    # NOTE: the below with 'SSA' did not work, so use 0 instead to return the totals
    #return {r: frame[r]['SSA'] for r in frame.columns}
    return {r: frame[r][0] for r in frame.columns}

def convert_percentiles(idx):
    """Convert percentiles"""
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
                return v + (v2 - v) * (x - partial) / s
            partial += s

    return np.array(list(map(convert_one, idx)))

def parse_data(data_dir=DATA_DIR, filenames=None):
    """Parse sqf data set."""
    if filenames is None:
        filenames = [FILES['cdf_by_race'], FILES['performance_by_race']]
    # NOTE: the pandas from_csv is no longer functional, so update the two lines below to read_csv
    cdfs = cleanup_frame(pd.read_csv(data_dir + filenames[0]))
    performance = 100 - cleanup_frame(pd.read_csv(data_dir + filenames[1]))
    return (cdfs / 100., performance / 100.)

def get_FICO_data(data_dir=DATA_DIR, do_convert_percentiles=True):
    """Get FICO data in desired format"""
    data_pair = parse_data(data_dir)
    totals = read_totals(data_dir)
    # NOTE: the below line is necessary for convert percentiles to work correctly
    scores = pd.read_csv(data_dir + FILES['performance_by_race'], usecols=['Score'])

    if do_convert_percentiles:
        for v in data_pair:
            # NOTE: using v.index as the parameter wasn't working because the indexes went
            #   from 0-197 (step 1) instead of 0-100 (step 0.5) so ended up with incorrect scores and
            #   NAN indexes after convert_percentiles
            # to fix this: input the scores from the csv
            #v.index = convert_percentiles(v.index)      # the original line
            v.index = convert_percentiles(scores['Score'])
    cdfs = data_pair[0]
    performance = data_pair[1]
    return cdfs, performance, totals
