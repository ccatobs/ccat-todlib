#!/bin/python

from sotodlib.io.bookbinder import *
import json
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--RootDir', type=str)
parser.add_argument('-o', '--OutputDir', type=str)
#parser.add_argument('-hk', '--HKFiles', type=list)
parser.add_argument('-d', '--DetFiles', type=str)


# making the detector files in the DetFiles folder into a dictionary

args = parser.parse_args()
g3_files = sorted(glob.glob(os.path.join(args.DetFiles, "*.g3")))

first_name = os.path.basename(g3_files[0])
run_id = "_".join(first_name.split("_")[:-1])

detfiles = {run_id: g3_files}

##################

hkfields = {'az' : 'observatory.acu1.feeds.Azimuth',
            'el' : 'observatory.acu1.feeds.Elevation'}

bbrun = BookBinder(args.RootDir, args.OutputDir, hkfields, [],
                       detfiles, require_acu = False, allow_bad_timing = True) #until PTP is resolved

bbrun.bind(pbar=True)

#Done.

'''
python bookbinder_testrun.py \
-r /data/shwetha \
-o /data/shwetha/bb_output/bbv0 \
-d /data/shwetha/det_files/rfsoc01_drone1/
'''
