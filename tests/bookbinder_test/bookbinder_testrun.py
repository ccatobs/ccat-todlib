#!/bin/python

from sotodlib.io.bookbinder import *
import json
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--RootDir', type=str)
parser.add_argument('-o', '--OutputDir', type=str)
parser.add_argument('-hk', '--HKFiles', type=str)
parser.add_argument('-d', '--DetFiles', type=str)


# Making the detector files in the DetFiles folder into a dictionary

args = parser.parse_args()
g3_files = sorted(glob.glob(os.path.join(args.DetFiles, "*.g3")))

first_name = os.path.basename(g3_files[0])
run_id = "_".join(first_name.split("_")[:-1])

detfiles = {run_id: g3_files}

hk_files = sorted(glob.glob(os.path.join(args.HKFiles, "*.g3")))
if len(hk_files) == 0:
    raise RuntimeError(f"No .g3 HK files found in {args.HKFiles}")

##################

hkfields = {'az' : 'observatory.acu.acu_udp_stream.Azimuth',
            'el' : 'observatory.acu.acu_udp_stream.Elevation'}

bbrun = BookBinder(args.RootDir, args.OutputDir, hkfields, hk_files,
                       detfiles, require_acu = True, allow_bad_timing = True) #until PTP is resolved

bbrun.bind(pbar=True)

#Done.

'''
python bookbinder_testrun.py \
-r /data/shwetha \
-o /data/shwetha/bb_output/bbv2 \
-hk /data/shwetha/hk_files/17655/timeshifted \
-d /data/shwetha/det_files/rfsoc01_drone1/
'''
