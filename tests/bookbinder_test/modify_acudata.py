#!/bin/python

'''
This script takes the ACU data files and shifts the time streams to
align within the detector data time window, for the purpose of testing
bookbinder module.
'''

import numpy as np
from sotodlib import core as so
from spt3g import core as g3

def modify_time(iframe, t_offset):
    '''Takes an input acu data frame and shifts
    the data block with the time offset t_offset,
    returns output frame.

    Args:
        iframe (G3Frame): input frame
        t_offset (float): unix time offset (10ns unit)
    '''
    oframe = g3.G3Frame(iframe.type)
    for k in iframe.keys():
        #copy all other objects except the times.
        if k not in ['start_time', 'timestamp', 'blocks']:
            oframe[k] = iframe[k]
        #shift the time objects
        elif k in ['timestamp', 'start_time']:
            oframe[k] = iframe[k] + t_offset/g3.G3Units.s #in seconds
        elif k=='blocks':
            data = iframe[k][0]
            data['Time'] = data['Time'] + t_offset/g3.G3Units.s #in seconds
            data.times = g3.G3VectorTime(np.array(data.times) + int(t_offset)) #in 10ns
            oframe[k] = g3.G3VectorFrameObject([data])
    return oframe

def run_timeshift(files, outloc, ogtime, modtime):
    '''Performs the timeshift of the postion data streams of the
    given list of files.

    Args:
        files (list): List of G3 files with the absolute paths.
        outloc (str): Absolute path of the output file location.
        ogtime (float): Unix time of the original start time of the
            position time stream.
        modtime (float): Unix start time of the detector time stream,
            position times will be shifted to this modified time.
    '''
    offset = modtime - ogtime
    for f in files:
        fname = f.split('/')[-1]
        oname = outloc + '/' + fname.split('.')[0] + '_m' + '.g3'
        writer = g3.G3Writer(oname)

        print (f"Processing file: {f}")
        g3f = g3.G3File(f)
        while True:
            try:
                frame = g3f.next()
            except:
                print ("Reached the end of the file")
                break

            oframe = modify_time(frame, offset)
            writer(oframe)

        writer(g3.G3Frame(g3.G3FrameType.EndProcessing))

    print ("Done.")
    return

def inspect_file(fname):
    '''A function to inspect the time information of the first three frames
    of the given file. This is to verify that the time shifting operation
    is working correctly.

    Args:
        fname (str): Name of the input g3 file.
    '''

    g3f = g3.G3File(fname)

    #frame 1
    frame = g3f.next()
    print (f"start_time: {frame['start_time']}")
    stime = frame['start_time']

    #frame 2
    frame = g3f.next()
    print (f"timestamp: {frame['timestamp']}")

    #frame 3: the data stream objects starts from this frame
    frame = g3f.next()
    data = frame['blocks'][0]
    print (np.array(data.times)[:10]) #showing only first 10 elements
    print (np.array(data['Time'])[:10])

    return stime

