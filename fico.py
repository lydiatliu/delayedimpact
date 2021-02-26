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
    #frame = cleanup_frame(pd.DataFrame.from_csv(data_dir + FILES['overview']))
    frame = cleanup_frame(pd.read_csv(data_dir + FILES['overview']))
    return {r: frame[r][0] for r in frame.columns}
    #return {r: frame[r]['SSA'] for r in frame.columns}

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

    # TODO 2: create a list containing the values 0 to 100 with 0.5 steps in between
    # create a global variable which is the count
    # update the count after every if before the return
    # the x value can be a float!!
    def convert_one(x):
        print('IN CONVERT ONE FUNC IN CONVERT PERCENTILES')
        print('this is the x score we are looking at:')
        print(x)
        print(type(x))
        print('after x type conversion:')
        print(type(x))
        partial = 0
        for ((v, s), (v2, _)) in zip(pdf, pdf[1:]):
            print(type(v))
            print(type(s))
            if partial + s >= x:
                print('entered into the if so will return:')
                print(v + (v2 - v) * (x - partial) / s)
                return v + (v2 - v) * (x - partial) / s
            partial += s

    return np.array(list(map(convert_one, idx)))

def parse_data(data_dir=DATA_DIR, filenames=None):
    """Parse sqf data set."""
    if filenames is None:
        filenames = [FILES['cdf_by_race'], FILES['performance_by_race']]

    #cdfs = cleanup_frame(pd.DataFrame.from_csv(data_dir + filenames[0]))
    #performance = 100 - cleanup_frame(pd.DataFrame.from_csv(data_dir + filenames[1]))

    #updated version of original below...but now issues with scores and -100 so need to fix:
    #performance = 100 - cleanup_frame(pd.read_csv(data_dir + filenames[1]))

    cdfs = cleanup_frame(pd.read_csv(data_dir + filenames[0]))
    performance = 100 - cleanup_frame(pd.read_csv(data_dir + filenames[1]))
    return (cdfs / 100., performance / 100.)

def get_FICO_data(data_dir=DATA_DIR, do_convert_percentiles=True):
    # NOTE: WHEN do_convert_percentiles is FALSE there are no NaN values and jupyter notebook runs!!
    """Get FICO data in desired format"""
    data_pair = parse_data(data_dir)
    totals = read_totals(data_dir)
    # FOR TROUBLESHOOTING
    #print('IN get_fico_data func')
    #print('data_pair values after running parse_data func')
    #print(data_pair)  # it's the same as what parse data output is in parse_data function

    #first_tuple = data_pair[1]
    #print(first_tuple.Score)
    #exit(0)

    if do_convert_percentiles:
        for v in data_pair:  # data_pair has a len of 2 bc it contains two tuples
            # TODO 1: might need to change the v.index
            # try creating a rangeIndex and specify the dtype as a float
            print('original v.index from data_pair in for loop')
            print(v.index)
            print(type(v.index))
            v.index = convert_percentiles(v.index)
            print('updated v.index:')
            print(v.index)
    cdfs = data_pair[0]

    # FOR TROUBLESHOOTING
    print('cdfs after do convert percentiles')
    print(cdfs)
    performance = data_pair[1]
    return cdfs, performance, totals
