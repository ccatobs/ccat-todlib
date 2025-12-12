#!/bin/python
'''
Base test script for ccat bookbinder implementation.
'''

from typing import Optional, Dict
from dataclasses import dataclass, fields

import so3g
from so3g.proj import Ranges
from spt3g import core
import itertools
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
from tqdm.auto import tqdm
import os
import logging
import sys
import shutil
import yaml
import datetime as dt
from zipfile import ZipFile
import sotodlib
from sotodlib.site_pipeline.util import init_logger

#from .datapkg_utils import walk_files

log = logging.getLogger('bookbinder')
if not log.hasHandlers():
    init_logger('bookbinder')

class TimingSystemOff(Exception):
    """Exception raised when we try to bind books where the timing system is 
    found to be off and the books have imprecise timing counters
    """
    pass

class NoScanFrames(Exception):
    """Exception raised when we try and bind a book but the SMuRF file contains
    not Scan frames (so no detector data)
    """
    pass

class NoHKFiles(Exception):
    """Exception raised when we cannot find any HK data around the book time"""
    pass

class NoMountData(Exception):
    """Exception raised when we cannot find mount data"""
    pass

class NoHWPData(Exception):
    """Exception raised when we cannot find HWP data"""
    pass

class DuplicateAncillaryData(Exception):
    """Exception raised when we find the HK data has copies of the same
    timestamps
    """
    pass

class BookDirHasFiles(Exception):
    """Exception raised when files already exist in a book directory"""
    pass

class NonMonotonicAncillaryTimes(Exception):
    """Exception raised when we find the HK data has timestamps that are not strictly increasing monotonically"""
    pass

def setup_logger(logfile=None):
    """
    This setups a logger for the bookbinder. If a logfile is passed, it will
    write to that file as well as stdout. It is useful to create a one-off
    logger here instead of using `getLogger` because it allows us to set
    a separate log-file for each bookbinder instance.
    """
    fmt = '%(asctime)s: %(message)s (%(levelname)s)'
    log = logging.Logger('bookbinder', level=logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if logfile is not None:
        ch = logging.FileHandler(logfile)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    
    return log

def get_frame_iter(files):
    """
    Returns a continuous iterator over frames for a list of files
    """
    return itertools.chain(*[core.G3File(f) for f in files])

def close_writer(writer):
    """
    Closes out a G#FileWriter with an end-processing frame. If None is passed,
    this will not do anything.
    """
    if writer is None:
        return
    writer(core.G3Frame(core.G3FrameType.EndProcessing))

def next_scan(it):
    """
    Returns the next scan frame, along with any intermediate frames for an
    iterator.
    """
    interm_frames = []
    for frame in it:
        if frame.type == core.G3FrameType.Scan:
            return frame, interm_frames
        interm_frames.append(frame)
    return None, interm_frames

class HkDataField:
    """
    Class data container for a single field of HK data.
    
    Args
    -----
    instance_id: str
        Instance ID of the agent producing the data.
    feed: str
        Feed name for the hk data feed.
    field: str
        Field name for the hk data feed.
        
    Attributes
    -----------
    times: np.ndarray
        HK timestamps sample.
    data: np.ndarray
        HK sample data.
    finalized: bool
        True if data has been processed and finalized.
    """
    def __init__(self, instance_id: str, feed: str, field: str):
        self.instance_id = instance_id
        self.feed = feed
        self.field = field

        self.times = []
        self.data  = []
        self.finalized = False
    
    def __len__(self):
        """Calculate the length of the time/data samples"""
        return len(self.times)
    
    @property
    def addr(self):
        """Returns full address of the data field."""
        return f"{self.instance_id}.{self.feed}.{self.field}"
    
    def process_frame(self, frame):
        """Capture and Update data based on G3Frame"""
        address = frame['address'] #addr format: <site>.<instance_id>.<feeds>.<feed_name>
        spl = address.split('.')
        instance_id, feed = spl[1], spl[3]
        #check for correct data feed
        if instance_id != self.instance_id or feed != self.feed:
            return
        
        for block in frame['blocks']:
            if self.field not in block:
                continue
            self.times.append(np.array(block.times) / core.G3Units.s)
            self.data.append(block[self.field])
    
    def finalize(self, drop_duplicates=False, require_monotonic_times=False):
        """Finalize data, and store in numpy array."""
        self.times = np.hstack(self.times, dtype=np.float64)
        self.data  = np.hstack(self.data)
        self.finalized = True

        #check for duplicates
        clean_times, idxs = np.unique(self.times, return_index=True)
        if len(self.times) != len(clean_times):
            if not drop_duplicates:
                raise DuplicateAncillaryData(
                    f"HK Data from {self.addr} has"
                    " duplicate timestamps"
                )
            else:
                log.warning(
                    f"HK Data from {self.addr} has duplicate timestamps"
                )
            self.times = self.times[idxs]
            self.data  = self.data[idxs]
        if not np.all(np.diff(self.times) > 0):
            bad = np.sum(np.diff(self.times) <= 0)
            msg = f"Times from {self.addr} have {bad} samples that are " \
                    "not strictly ascending in order."
            if require_monotonic_times:
                raise NonMonotonicAncillaryTimes(msg)
            else:
                log.warning(msg)
 

@dataclass
class HkData:
    """
    Class containing HK Data (ACU) for bookbinding.
    """
    az: Optional[HkDataField] = None
    el: Optional[HkDataField] = None
    #boresight: Optional[HkDataField] = None
    #corotator_enc: Optional[HkDataField] = None
    #az_mode: Optional[HkDataField] = None

    @classmethod
    def from_dict(cls, d: Dict[str, str]):
        """
        Creates HK Data object from a dict of field addresses. Addresses must
        be formatted like ``<instance_id>.<feed>.<field>``. Keys of dict must
        be valid fields of HkData class, e.g. ``az`` or ``el``.
        """
        kw = {}
        for k,v in d.items():
            try:
                _ , instance, feed, field = v.split('.')
            except Exception as exc:
                raise ValueError(f"Could not parse: {v}."
                                 "Must be formatted <instance_id>.<feed>.<field>") from exc
            kw[k] = HkDataField(instance, feed, field)
        return cls(**kw)
    
    def process_frame(self, frame):
        """Processes G3Frame, and updates corresponding HkDataFields."""
        for fld in fields(self):
            f = getattr(self, fld.name)
            if isinstance(f, HkDataField):
                f.process_frame(frame)

    def finalize(self, drop_duplicates=True):
        """Finalizes HkDataFields."""
        for fld in fields(self):
            f = getattr(self, fld.name)
            if isinstance(f, HkDataField):
                f.finalize(drop_duplicates=drop_duplicates)

class AncilProcessor:
    """
    Processor class for ancillary (ACU) data.
    
    Params
    -------
    files : list
        List of HK files to process.
    book_id : str
        ID of the book being bound.
    hk_fields : dict
        Dictinary of fields corresponding to the relevant HK Data. See the 
        HkData class for what housekeeping fields are allowed. For example::
            >> hk_fields = {
                'az': 'acu.acu_udp_stream.Corrected_Azimuth',
                'el': 'acu.acu_udp_stream.Corrected_Elevation',
                'boresight': 'acu.acu_udp_stream.Corrected_Boresight',
                'az_mode':  'acu.acu_status.Azimuth_mode',
            }
    
    Attributes
    -----------
    hkdata : HkData
        Class containing relevant Hk data for the duration of the book.
    times : np.ndarray
        Timestamps for anc data. This will be populated after preprocess.
    anc_frame_data : List[G3TimestreamMap]
        List of G3TimestreamMap saved for each bound frame. This will be
        populated on bind and should be used to add copies of the anc data to
        the detector frames.
    """
    def __init__(self, files, book_id, hk_fields: Dict,
                 drop_duplicates=False, require_acu=True,
                 log=None):
        self.hkdata: HkData = HkData.from_dict(hk_fields)

        self.files = files
        self.anc_frame_data = None
        self.out_files = []
        self.book_id = book_id
        self.preprocessed = False
        self.drop_duplicates = drop_duplicates
        self.require_acu = require_acu

        if log is None:
            self.log = logging.getLogger('bookbinder')
        else:
            self.log = log

        if len(self.files) == 0:
            if self.require_acu:
                raise NoHKFiles("No HK files specified for book")
            self.log.warning("no HK files found for the book")
            for fld in ['az', 'el']:#, 'boresight', 'az_mode']:
                setattr(self.hkdata, fld, None)
        
        if self.require_acu is None:
            self.log.warning("No ACU data specified in hk_fields!")
        
    def preprocess(self):
        """Preprocesses HK data and populates the `data` and `times` objects."""
        if self.preprocessed:
            return
        
        self.log.info("Preprocessing HK Data")
        frame_iter = get_frame_iter(self.files)

        #iterate over the frames and process them
        for fr in frame_iter:
            if fr['hkagg_type'] != 2:
                continue
            self.hkdata.process_frame(fr)

        # Look for ACU fields that are configured but not found in HK data files
        # will not check fields that are NOT in the configuration file.
        # for fld in ['az', 'el', 'boresight']:
        for fld in ['az', 'el']:
            f = getattr(self.hkdata, fld)
            if f is not None:
                if self.require_acu and len(f) == 0:
                    raise NoMountData(
                        f"Did not find ACU data in {self.files} for {fld}",
                    )
                elif len(f) == 0:
                    # requiring ACU data is False and we didn't find any
                    self.log.warning(
                        f"Did not find ACU data for {fld}. Bypassed because"
                        " required_ACU is False"
                    )
                    setattr(self.hkdata, fld, None)

        self.hkdata.finalize(drop_duplicates=self.drop_duplicates)
        self.preprocessed = True

    def bind(self, outdir, times, frame_idxs, file_idxs):
        """
        Binds ancillary data.
        
        Params
        -------
        outdir : path
            Path where files should be written.
        times : np.ndarray
            Array of timestamps to write to book.
        frame_idxs : np.ndarray
            Array mapping sample (in times) to output frame index.
        file_idxs : np.ndarray
            Array mapping output frame idx to output file idx.
        """
        self.log.info("Binding ancillary data")

        # Handle file writers
        writer = None
        cur_file_idx = None
        out_files = []

        def validate_mount_field(hk_field: HkDataField, max_dt=None):
            """Validate the samples in the field that are considered
            for processing and binding.
            """
            m = (times[0] <= hk_field.times) & (hk_field.times <= times[-1])
            if m.sum() < 2:
                raise NoMountData(
                    f"No mount data overlapping with detector data: "
                    " {hk_field.addr}"
                )
            if max_dt is not None:
                _max_dt = np.max(np.diff(hk_field.times[m]))
                if _max_dt > max_dt:
                    raise NoMountData(
                        f"Max data spacing {_max_dt}s is higher than"
                         " {max_dt}s for {hk_field.addr}. "
                         "Interpolation may be questionable!"
                    )
            
        # Proceed and interpolate ACU times to detector times
        acu_interp_data = {}
        for fld in ['az', 'el']:#, 'boresight']:
            f = getattr(self.hkdata, fld)
            if f is not None:
                try:
                    validate_mount_field(f, max_dt=10)#why 10s?
                    acu_interp_data[fld] = np.interp(
                        times, f.times, f.data
                    )
                except NoMountData as e:
                    if self.require_acu:
                        raise e
                    else:
                        self.log.warning(e)
                        acu_interp_data[fld] = None
            else:
                acu_interp_data[fld] = None
        
        az = acu_interp_data['az']
        el = acu_interp_data['el']
        #boresight = acu_interp_data['boresight']

        anc_frame_data = []
        for oframe_idx in np.unique(frame_idxs):
            # Update file writer if starting a new output file
            if file_idxs[oframe_idx] != cur_file_idx:
                close_writer(writer)
                cur_file_idx = file_idxs[oframe_idx]
                fname = os.path.join(outdir, f'A_ancil_{cur_file_idx:0>3}.g3')
                out_files.append(fname)
                writer = core.G3Writer(fname)
            
            m = frame_idxs == oframe_idx
            ts = times[m]

            oframe = core.G3Frame(core.G3FrameType.Scan)

            i0, i1 = np.where(m)[0][[0, -1]]#initial and final timestamps?
            oframe['sample_range'] = core.G3VectorInt([int(i0), int(i1+1)])#+1?
            oframe['book_id'] = self.book_id

            anc_data = core.G3TimesampleMap()
            anc_data.times = core.G3VectorTime(ts * core.G3Units.s)
            if az is not None:
                anc_data['az_enc'] = core.G3VectorDouble(az[m])
                anc_data['el_enc'] = core.G3VectorDouble(el[m])
            #if boresight is not None:
            #    anc_data['boresight_enc'] = core.G3VectorDouble(boresight[m])
            oframe['ancil'] = anc_data
            writer(oframe)
            anc_frame_data.append(anc_data)

            self.add_acu_summary_info(oframe, ts[0], ts[-1])

        # Save the processed anc data to be added to detector files
        self.anc_frame_data = anc_frame_data
        self.out_files = out_files
    
    def add_acu_summary_info(self, frame, t0, t1):
        """
        Adds ACU summary information to a G3Frame. This will add the following
        info if data is present in the HK dataset::
            - azimuth_mode: (str)
                Azimuth_mode, pulled from the ACU summary data. `ProgramTrack`
                means that this frame contains scan data, and `Preset` means the
                telescope is slewing.
            - azimuth_velocity_mean / azimuth_velocity_std: (float / float)
                Mean and standard deviation of the az velocity (deg / sec)
            - elevation_velocity_mean / elevaction_velocity_std: (float / float)
                Mean and standard deviation of the el velocity (deg / sec)
        
        Params
        -------
        frame : G3Frame
            Frame to add data to
        t0 : float
            Start time of the frame (unix time), inclusive
        t1 : float
            Stop time of the frame (unix time), inclusive
        """
        #az_mode = self.hkdata.az_mode
        az = self.hkdata.az
        el = self.hkdata.el
        #if az_mode is not None:
        #    m = (t0 <= az_mode.times) & (az_mode.times <= t1)
        #    if not np.any(m):
        #        frame['azimuth_mode'] = 'None'
        #    elif 'ProgramTrack' in az_mode.data[m]:
        #        frame['azimuth_mode'] = 'ProgramTrack' #scanning
        #    else:
        #        frame['azimuth_mode'] = 'Preset' #slewing

        if az is not None:
            m = (t0 <= az.times) & (az.times <= t1)
            if m.sum() >= 2:
                dt = np.diff(az.times[m]).mean()
                az_vel = np.diff(az.data[m]) / dt
                frame['azimuth_velocity_mean'] = np.mean(az_vel)
                frame['azimuth_velocity_stdev'] = np.std(az_vel)
        for k in ['azimuth_velocity_mean', 'azimuth_velocity_stdev']:
            if k not in frame:
                frame[k] = np.nan
        
        if el is not None:
            m = (t0 <= el.times) & (el.times <= t1)
            if m.sum() >= 2:
                dt = np.diff(el.times[m]).mean()
                el_vel = np.diff(el.data[m]) / dt
                frame['elevation_velocity_mean'] = np.mean(el_vel)
                frame['elevation_velocity_stdev'] = np.std(el_vel)
        for k in ['elevation_velocity_mean', 'elevation_velocity_stdev']:
            if k not in frame:
                frame[k] = np.nan

class RfsocStreamProcessor:
    """
    Processor class for RFSoC detector data.

    Params
    -------
    obs_id : str
        Observation ID corresponding to the detector data.
    files : list
        List of detector data files
    book_id : str
        Book ID related to the observation.
    readout_ids : list
        List of readout IDs of the RFSoC streams
    allow_bad_timing : bool
        Whether to allow missing detector time stream values.
    """
    def __init__(self, obs_id, files, book_id,
                 log=None, allow_bad_timing=False):
        self.files = files
        self.obs_id = obs_id
        self.stream_id = None
        self.times = None
        self.frame_idxs = None
        self.nchans = None
        self.nframes = None
        self.bias_names = None
        self.primary_names = None
        self.timing_paradigm = None
        self.session_id = None
        self.slow_primary = None
        self.ccatstream_version = None
        self.out_files = []
        self.book_id = book_id
        self.allow_bad_timing = allow_bad_timing

        if log is None:
            self.log = logging.getLogger('bookbinder')
        else:
            self.log = log
    
    def preprocess(self):
        """Extract file times, nchans, and nframes from file list."""
        if self.times is not None: # If already preprocessed
            return
        
        self.log.info(f"Preprocessing rfsoc obsid {self.obs_id}")

        self.nframes = 0
        ts = []
        rfsoc_frame_counters = []
        #fc_idx = None
        sample_offset = 0 #alternate to SO's frame-counter implementation
        frame_idxs = []
        frame_idx = 0
        timing = True
        for frame in get_frame_iter(self.files):
            if frame.type != core.G3FrameType.Scan:
                continue

            # Populate attributes from the first scan frame
            if self.nchans is None:
                self.nchans = frame['channel_count']
                self.readout_ids = list(frame['data'].names)
                #self.nchans = len(self.readout_ids) # is this right for ccat? should just be 'channel_count'?
                # self.primary_names = frame['primary'].names
                #fc_idx = frame['frame_num'] #this is simply current frame index?
                # self.bias_names = frame['tes_biases'].names
                # self.timing_paradigm = frame['timing_paradigm']
                self.timing_paradigm = 'Low Precision'
                self.session_id = frame['session_id']
                if 'slow_primary' in frame:
                    self.slow_primary = frame['slow_primary']
                self.ccatstream_version = frame['ccatstream_version']
                self.stream_id = frame['ccatstream_id']

            good, t = get_frame_times(frame, self.allow_bad_timing)
            timing = timing and good #will be False for ccat (for now)
            ts.append(t)
            # frame-counter differs from SO
            num_sample = frame['num_samples']
            frame_counter = sample_offset + np.arange(num_sample)
            sample_offset += num_sample
            rfsoc_frame_counters.append(frame_counter)
            frame_idxs.append(np.full(len(t), frame_idx, dtype=np.int32))

            self.nframes += 1
            frame_idx += 1

        if len(ts) == 0:
            raise NoScanFrames(f"{self.obs_id} has no detector data!")
        self.times = np.hstack(ts)
        self.rfsoc_frame_counters = np.hstack(rfsoc_frame_counters)
        self.frame_idxs = np.hstack(frame_idxs)

        timing = timing and (not self.timing_paradigm=='Low Precision')

        if (not self.allow_bad_timing) and (not timing):
            raise TimingSystemOff(
                f"Observation {self.obs_id} does not have high precision timing"
                " information. Pass `allow_bad_timing=True` to bind anyway"
            )
        
        # If low-precision, we need to linearize timestamps in order for
        # bookbinder to work properly
        if not timing:
            self.log.warning(
                "Timestamps are Low Precision, linearizing from frame-counter"
            )
            dt, offset = np.polyfit(self.rfsoc_frame_counters, self.times, 1)
            self.times = offset + dt * self.rfsoc_frame_counters

    def bind(self, outdir, times, frame_idxs, file_idxs, pbar=False, ancil=None,
             atol=1e-2):
        """
        Binds RFSoC data.
        
        Params
        -------
        outdir : str
            Output directory to put bound files.
        times : np.ndarray
            Full array of timestamps that should be contained in the book.
        frame_idxs : np.ndarray
            Output frame idx for each specified timestamps.
        file_idxs : np.ndarray
            Output file idx for each specified output frame.
        pbar : bool or tqdm.tqdm
            If True, will create a new progress bar. If False, will disable.
            If a progress bar is passed, will use that instead.
        ancil : AncilProcessor
            Ancillary processor object (must be already bound). Ancil data will
            be copied into the output frames.
        atol : float
            Absolute tolerance between smurf-timestamps and book-timestamps.
            Samples mapped to times that are further away than atol will
            be considered unmapped.
        """
        if pbar is True:
            pbar = tqdm(total=self.nframes)
        elif pbar is False:
            pbar = tqdm(total=self.nframes, disable=True)

        pbar.set_description(f"Binding {self.stream_id}")

        self.log.info(f"Binding rfsoc obsid {self.obs_id}")

        # Here `times` is the full array of times in the book (gapless reference
        # times) and `self.times`` is timestamps from the pre-processed L2 data
        # (may contain gaps)
        sample_map = find_ref_idxs(times, self.times)
        mapped = np.abs(times[sample_map] - self.times) < atol
        oframe_idxs = frame_idxs[sample_map] # out-frame idx for each in sample
        oframe_idxs[~mapped] = -1
        _, offsets = np.unique(frame_idxs, return_index=True)
        # Sample idx within each out-frame for every input sample
        out_offset_idxs = sample_map - offsets[frame_idxs[sample_map]]

        iframe_idxs = self.frame_idxs #in-frame idx for each in sample
        _, offsets = np.unique(self.frame_idxs, return_index=True)
        # Sample idx within each in-frame for every input sample
        in_offset_idxs = np.arange(len(self.times)) - offsets[self.frame_idxs]

        # Handle file writers
        writer = None
        cur_file_idx = None
        out_files = []

        inframe_iter = get_frame_iter(self.files)
        iframe, interm_frames = next_scan(inframe_iter)
        iframe_idx = 0
        oframe_num = 0
        pbar.update()

        for oframe_idx in np.unique(frame_idxs):
            # Update writer
            if file_idxs[oframe_idx] != cur_file_idx:
                close_writer(writer)
                cur_file_idx = file_idxs[oframe_idx]
                fname = os.path.join(
                    outdir, f'D_{self.stream_id}_{cur_file_idx:0>3}.g3'
                )
                out_files.append(fname)
                writer = core.G3Writer(fname)

            # Initialize stuff
            m = frame_idxs == oframe_idx
            nsamp = np.sum(m)
            ts = times[m]
            data = np.zeros((int(self.nchans * 2), nsamp), dtype=np.int32) # factor of 2 appears due to I and Q
            # biases = np.zeros((len(self.bias_names), nsamp), dtype=np.int32)
            # primary = np.zeros((len(self.primary_names), nsamp), dtype=np.int64)
            filled = np.zeros(nsamp, dtype=bool)
            # Loop over the in_frames and filling current out_frame
            while True:
                # First, write any intermediate frames like observation and wiring
                for fr in interm_frames:
                    if 'frame_num' in fr:
                        del fr['frame_num']
                    fr['frame_num'] = oframe_num # Update this so they remain ordered
                    oframe_num += 1
                    writer(fr)


                m = (oframe_idxs == oframe_idx) & (iframe_idxs == iframe_idx)
                outsamps = out_offset_idxs[m]
                insamps = in_offset_idxs[m]
                # Below is equivalent to:
                #    >> data[:, outsamps] = iframe['data'][:, insamps]
                #    >> ...
                # However it is much faster to copy arrays by mapping contiguous
                # chunks to contiguous chunks using slices, like:
                #    >> arr_out[:, o0:o1] = arr_in[:, i0:i1]
                # since numpy does not need to create a temporary copy of the
                # data, and can just do a direct mem-map. This speeds up binding
                # by a factor of ~4.
                #
                # Here we are splitting outsamps and insamps into a list
                # of ranges where both arrays are contiguous. Then we loop
                # through each sub-range and copy data via slicing.

                #split_idxs = 1 + np.where(
                #    (np.diff(outsamps) > 1) | (np.diff(insamps) > 1))[0]
                #outsplits = np.split(outsamps, split_idxs)
                #insplits = np.split(insamps, split_idxs)
                #for i in range(len(outsplits)):
                #    if len(insplits[i]) == 0:
                #        continue

                #    in0, in1 = insplits[i][0], insplits[i][-1] + 1
                #    out0, out1 = outsplits[i][0], outsplits[i][-1] + 1

                #    data[:, out0:out1] = iframe['data'].data[:, in0:in1]
                #    #biases[:, out0:out1] = iframe['tes_biases'].data[:, in0:in1]
                #    #primary[:, out0:out1] = iframe['primary'].data[:, in0:in1]
                #    filled[out0:out1] = 1
                data[:, outsamps] = iframe['data'].data[:, insamps]
                filled[outsamps] = 1
                # If there are any remaining samples in the next in_frame, pull it and repeat
                if np.any((oframe_idxs == oframe_idx) & (iframe_idxs > iframe_idx)):
                    iframe, interm_frames = next_scan(inframe_iter)
                    iframe_idx += 1
                    pbar.update()
                    continue
                else:
                    break

            # Interpolate data where there are gaps
            if np.all(~filled):
                self.log.error(
                    f"No samples properly mapped in oframe frame {oframe_idx}!"
                    "Cannot properly interpolate for this frame."
                )
                raise ValueError(f"Cannot finish binding {self.obs_id}")
            elif np.any(~filled):
                self.log.debug(
                    f"{np.sum(~filled)} missing samples in out-frame {oframe_idx}"
                )
                # Missing samples at the beginning / end of a frame will be
                # filled with the first / last sample in the frame
                fill_value = (data[:, filled][:, 0], data[:, filled][:, -1])
                data[:, ~filled] = interp1d(
                    ts[filled], data[:, filled], axis=1, assume_sorted=True,
                    kind='linear', fill_value=fill_value, bounds_error=False
                )(ts[~filled])

            m = frame_idxs == oframe_idx
            i0, i1 = np.where(m)[0][[0, -1]]

            oframe = core.G3Frame(core.G3FrameType.Scan)

            if ancil is not None:
                t0, t1 = ts[0], ts[-1]
                oframe['ancil'] = ancil.anc_frame_data[oframe_idx]
                ancil.add_acu_summary_info(oframe, t0, t1)

            oframe['book_id'] = self.book_id
            oframe['sample_range'] = core.G3VectorInt([int(i0), int(i1+1)])
            oframe['flag_smurfgaps'] = core.G3VectorBool(~filled)

            ts = core.G3VectorTime(ts * core.G3Units.s)
            oframe['signal'] = so3g.G3SuperTimestream(self.readout_ids, ts, data)
            #oframe['primary'] = so3g.G3SuperTimestream(self.primary_names, ts, primary)
            #oframe['tes_biases'] = so3g.G3SuperTimestream(self.bias_names, ts, biases)
            oframe['stream_id'] = self.stream_id
            oframe['frame_num'] = oframe_num

            oframe_num += 1
            writer(oframe)

        close_writer(writer)
        self.out_files = out_files

        # In case there were remaining frames left over
        pbar.update(self.nframes - iframe_idx - 1)
        if pbar.n >= pbar.total:
            pbar.close()

class BookBinder:
    """
    Class for combining smurf and hk L2 data to create books containing detector
    timestreams.

    Currently, this class works without the database implementation, and
    requires the user to pass in the relevant files and information. Initialization
    can be modified/updated to use the database in the future through imprinter.

    Parameters
    ----------
    book : sotodlib.io.imprinter.Books
        Book object to bind
    obsdb : dict
        Result of imprinter.get_g3tsmurf_obs_for_book. This should be a dict
        from obs-id to G3tSmurf Observations.
    filedb : dict
        Result of imprinter.get_files_for_book. This should be a dict from
        obs-id to the list of smurf-files for that observation.
    hkfiles : list
        List of HK files to process.
    max_samps_per_frame : int
        Max number of samples per frame. This will be used to split frames
        when the ACU data isn't present or cannot be used.
    max_file_size : int
        Max file size in bytes. This will be used to rotate files.
    readout_ids : dict, optional
        Dict of readout_ids to use for each stream_id. If provided, these
        will be used to set the `names` in the signal frames. If not provided,
        names will be taken from the input frames.
    ignore_tags : bool, optional
        if true, will ignore tags if the level 2 observations have unmatched
        tags
    ancil_drop_duplicates: bool, optional
        if true, will drop duplicate timestamp data from ancillary files. added
        to deal with an occassional hk aggregator error where it is picking up
        multiple copies of the same data
    require_acu: bool, optional
        if true, will throw error if we do not find Mount data
    require_hwp: bool, optional
        if true, will throw error if we do not find HWP data
    allow_bad_time: bool, optional
        if not true, books will not be bound if the timing systems signals are not found.
    
    Attributes
    -----------
    ancil : AncilProcessor
        Processor for ancillary data
    streams : dict
        Dict of SmurfStreamProcessor objects, keyed by stream_id
    times : np.ndarray
        Array of times for all samples in the book
    frame_idxs : np.ndarray
        Array of output frame indices for all samples in the book
    file_idxs : np.ndarray
        Array of output file indices for all output frames in the book
    """
    def __init__(self, data_root, outdir, hk_fields,
                 hkfiles, detfiles, ancil_drop_duplicates=False, 
                 max_samps_per_frame=50_000, max_file_size=1e9,
                 require_acu=True, allow_bad_timing=False,
                 ):
        self.data_root = data_root
        self.hk_root = os.path.join(data_root, 'hk')
        self.meta_root = os.path.join(data_root, 'rfsoc')

        self.outdir = outdir

        self.max_samps_per_frame = max_samps_per_frame
        self.max_file_size = max_file_size
        self.allow_bad_timing = allow_bad_timing

        if os.path.exists(outdir):
            # don't count hidden files, possibly from NFS processes
            nfiles = len([f for f in os.listdir(outdir) if f[0] != '.'])
            if nfiles > 1:
                raise BookDirHasFiles(
                    f"Output directory {outdir} contains files. Delete to retry"
                    " bookbinding"
                )
            elif nfiles == 1:
                assert os.listdir(outdir)[0] == 'Z_bookbinder_log.txt', \
                    f"only acceptable file in new book path {outdir} is " \
                    " Z_bookbinder_log.txt"
        else:
            os.makedirs(outdir)
        
        logfile = os.path.join(outdir, 'Z_bookbinder_log.txt')
        self.log = setup_logger(logfile)

        try:
            self.hkfiles = hkfiles
        except NoHKFiles as e:
            if require_acu:
                self.log.error(
                    "HK files are required if we require ACU data"
                )
                raise e
            self.log.warning(
                "Found no HK files during book time, binding anyway because "
                "require_acu and require_hwp are False"
            )
            self.hkfiles = []

        self.ancil = AncilProcessor(
            self.hkfiles,
            'bookID',
            hk_fields,
            log=self.log,
            drop_duplicates=ancil_drop_duplicates,
            require_acu=require_acu,
        )
        self.streams = {}
        for stream_id, files in detfiles.items():
            self.streams[stream_id] = RfsocStreamProcessor(
                'obsID', files, 'bookID', log=self.log,
                allow_bad_timing=self.allow_bad_timing,
            )
        
        self.times = None
        self.frame_idxs = None
        self.file_idxs = None
        self.meta_files = None

    def preprocess(self):
        """
        Runs preprocessing steps for the book. Preprocesses smurf and ancillary
        data. Creates full list of book-times, the output frame idx for each
        sample, and the output file idx for each frame.
        """
        if self.times is not None:
            return

        for stream in self.streams.values():
            stream.preprocess()

        t0 = np.max([s.times[0] for s in self.streams.values()])
        t1 = np.min([s.times[-1] for s in self.streams.values()])
        # prioritizes the last stream
        # implicitly assumes co-sampled (this is where we could throw errors
        # after looking for co-sampled data)
        ts, _ = fill_time_gaps(stream.times) 
        m = (t0 <= ts) & (ts <= t1)
        ts = ts[m]

        self.ancil.preprocess()

        # Divide up frames, only look within detector data and +/-30 seconds
        frame_splits = None #for now

        if frame_splits is None:
            frame_idxs = np.arange(len(ts)) // self.max_samps_per_frame
        else:
            frame_idxs = np.digitize(ts, frame_splits)
            frame_idxs -= frame_idxs[0]
            new_frame_idxs = frame_idxs.copy()
            # Divide up frames that are too long
            for fidx in np.unique(frame_idxs):
                m = frame_idxs == fidx
                new_frame_idxs += np.cumsum(m) // self.max_samps_per_frame
            frame_idxs = new_frame_idxs

        # Divide up files
        samp_size = 4 # bytes
        max_chans = np.max([s.nchans for s in self.streams.values()])
        totsize = samp_size * max_chans * np.arange(len(ts))
        file_idxs = []
        for fr in np.unique(frame_idxs):
            idx = np.where(frame_idxs == fr)[0][-1]
            file_idxs.append(totsize[idx] // self.max_file_size)
        file_idxs = np.array(file_idxs, dtype=int)

        self.log.info("Finished preprocessing data")

        self.times = ts
        self.frame_idxs = frame_idxs
        self.file_idxs = file_idxs

    def bind(self, pbar=False):
        """
        Binds data.

        Params
        ---------
        pbar : bool
            If True, will enable a progress bar.
        """
        self.preprocess()

        self.log.info(f"Binding data to {self.outdir}")
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # Bind Ancil Data
        self.ancil.bind(self.outdir, self.times, self.frame_idxs, self.file_idxs)
        
        tot = np.sum([s.nframes for s in self.streams.values()])
        pbar = tqdm(total=tot, disable=(not pbar))
        for stream in self.streams.values():
            stream.bind(self.outdir, self.times, self.frame_idxs,
                        self.file_idxs, pbar=pbar, ancil=self.ancil)

        self.log.info("Finished binding data. Exiting.")
        return True


# testmode: bypass to list of hk files
def get_hk_files(flist):
    return flist
def fill_time_gaps(ts):
    """
    Fills gaps in an array of timestamps.

    Parameters
    -------------
    ts : np.ndarray
        List of timestamps of length `n`, potentially with gaps
    
    Returns
    --------
    new_ts : np.ndarray
        New list of timestamps of length >= n, with gaps filled.
    mask : np.ndarray
        Returns a mask that tells you which elements of `new_ts` are
        taken from real data and which are interpolated. `~mask` gives
        the indices of the interpolated elements.
    """
    # Find indices where gaps occur and how long each gap is
    dts = np.diff(ts)
    dt = np.median(dts)
    missing = np.round(dts/dt - 1).astype(int)
    total_missing = int(np.sum(missing))

    # Create new array with the correct number of samples
    new_ts = np.full(len(ts) + total_missing, np.nan)

    # Insert old timestamps into new array with offsets that account for gaps
    offsets = np.concatenate([[0], np.cumsum(missing)])
    i0s = np.arange(len(ts))
    new_ts[i0s + offsets] = ts

    # Use existing data to interpolate and fill holes
    m = np.isnan(new_ts)
    xs = np.arange(len(new_ts))
    interp = interp1d(xs[~m], new_ts[~m])
    new_ts[m] = interp(xs[m])

    return new_ts, ~m
#_primary_idx_map = {} # counter for ccat?
def get_frame_times(frame, allow_bad_timing=False):
    """
    Returns timestamps for a G3Frame of detector data.

    Parameters
    --------------
    frame : G3Frame
        Scan frame containing detector data
    allow_bad_timing: bool, optional
        if not true, raises an error if it finds data with imprecise timing

    Returns
    --------------
    high_precision : bool
        If true, timestamps are computed from timing counters. If not, they are
        software timestamps
    
    timestamps : np.ndarray
        Array of timestamps (sec) for samples in the frame

    """
    #if len(_primary_idx_map) == 0:
    #    for i, name in enumerate(frame['primary'].names):
    #        _primary_idx_map[name] = i
        
    #c0 = frame['primary'].data[_primary_idx_map['Counter0']]
    #c2 = frame['primary'].data[_primary_idx_map['Counter2']]

    #counters = np.all( np.diff(c0)!=0 ) and np.all( np.diff( c2 )!=0)

    #if counters:
    #    return True, counters_to_timestamps(c0, c2)
    #elif allow_bad_timing:
    #    return False, np.array(frame['data'].times) / core.G3Units.s
    if allow_bad_timing:
        return False, np.array(frame['data'].times) / core.G3Units.s
    else:
        ## don't change this error message. used in Imprinter CLI
        raise TimingSystemOff("Timing counters not incrementing")

def find_ref_idxs(refs, vs):
    """
    Creates a mapping from a list of timestamps (vs) to a list of reference
    timestamps (refs). Returns an index-map of shape `vs.shape`, that maps each
    timestamp of the array `vs` to the closest timestamp in the array `refs`.

    This assumes that `refs` and `vs` are sorted in ascending order.

    Parameters
    ----------
    refs : array_like
        List of reference timestamps
    vs : array_like
        List of timestamps

    Returns
    -------
    idxs : array_like
        Map of shape `vs.shape` that maps each element of `vs` to the closest
        element in `refs`.
    """
    # Find the indices of the samples in the list of timestamps (vs)
    # that are closest to the reference timestamps
    idx = np.searchsorted(refs, vs, side='left')
    idx = np.clip(idx, 1, len(refs)-1)
    # shift indices to the closest sample
    left = refs[idx-1]
    right = refs[idx]
    idx -= vs - left < right - vs
    return idx

    