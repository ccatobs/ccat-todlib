from modify_acudata import *
from datetime import datetime
import spt3g.core as core


det_file = '/data/shwetha/det_files/rfsoc01_drone1/r01d1_1755824003_000.g3'
acu_file = '/data/shwetha/hk_files/17655/1765561579.g3'

# location where the modified g3 files will be written
outloc = '/data/shwetha/hk_files/17655/timeshifted'
outfile = '/1765561579_m.g3'

# --- ACU ---
g3f = g3.G3File(acu_file)
frame = g3f.next()
acu_start = frame['start_time']
print("ACU start time :", acu_start)

# --- DETECTOR ---
g3f = g3.G3File(det_file)
frame = g3f.next()
det_time = frame['time']
# Convert G3Time to UNIX time
det_unix_time = det_time.time / 1e8

print("Detector start time :", det_unix_time)

# ist of acu g3 files to be modified
flist = [acu_file]

# original start time of the acu simulated data streams
otime = acu_start

# original start time of the detector data streams, acu times will be shifted to
# this time window
mtime = det_unix_time

run_timeshift(flist, outloc, otime, mtime)

offset = mtime - otime
print("Time shift has been offset by:", offset)

# --- ACU ---

shifted_acu_file = outloc + outfile
g3f = g3.G3File(shifted_acu_file)
frame = g3f.next()
print("New ACU start time :", frame['start_time'])

# --- DETECTOR ---
g3f = g3.G3File(det_file)
frame = g3f.next()
det_time = frame['time']

# Convert G3Time → UNIX time
det_unix_time = det_time.time / 1e8

print("Detector start time :", det_unix_time)


