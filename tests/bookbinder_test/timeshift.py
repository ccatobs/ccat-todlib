from modify_acudata import *
from datetime import datetime
import spt3g.core as core
import numpy as np


det_file = '/data/shwetha/det_files/rfsoc01_drone1/r01d1_1755824003_000.g3'
acu_file = '/data/shwetha/hk_files/17655/1765561579.g3'

# location where the modified g3 files will be written
outloc = '/data/shwetha/hk_files/17655/timeshifted'
outfile = '/1765561579_m.g3'

# --- ACU: Get actual data block times ---
g3f = g3.G3File(acu_file)
acu_data_times = []
for fr in g3f:
    if 'blocks' in fr:
        for blk in fr['blocks']:
            if hasattr(blk, 'times'):
                acu_data_times.extend(np.array(blk.times) / g3.G3Units.s)
if not acu_data_times:
    raise RuntimeError("No ACU data block times found!")
acu_start = min(acu_data_times)
print("ACU data start time :", acu_start)

# --- DETECTOR: Get actual scan frame times ---
g3f = g3.G3File(det_file)
det_times = []
for fr in g3f:
    if fr.type == g3.G3FrameType.Scan and 'data' in fr:
        det_times.extend(np.array(fr['data'].times) / g3.G3Units.s)
        break  # Just need first scan
if not det_times:
    raise RuntimeError("No detector scan times found!")
det_start = min(det_times)
print("Detector data start time :", det_start)

# List of acu g3 files to be modified
flist = [acu_file]

# Compute offset to align ACU data start with detector data start
offset = det_start - acu_start

print(f"Computed offset: {offset:.1f} seconds ({offset/86400:.2f} days)")

# Apply timeshift
run_timeshift(flist, outloc, acu_start, det_start)

print(f"\nTime shift applied: offset = {offset:.1f} seconds")

# --- Verify shifted ACU file ---
shifted_acu_file = outloc + outfile
g3f = g3.G3File(shifted_acu_file)
shifted_data_times = []
for fr in g3f:
    if 'blocks' in fr:
        for blk in fr['blocks']:
            if hasattr(blk, 'times'):
                shifted_data_times.extend(np.array(blk.times) / g3.G3Units.s)
if shifted_data_times:
    shifted_start = min(shifted_data_times)
    shifted_end = max(shifted_data_times)
    print(f"Shifted ACU data start time: {shifted_start} ({datetime.utcfromtimestamp(shifted_start)})")
    print(f"Shifted ACU data end time:   {shifted_end} ({datetime.utcfromtimestamp(shifted_end)})")
    print(f"Detector data start time:    {det_start} ({datetime.utcfromtimestamp(det_start)})")
    
    if shifted_start <= det_start <= shifted_end:
        print("✓ Shifted ACU data covers detector start time")
    else:
        print("✗ Warning: Shifted ACU data may not overlap with detector data")



