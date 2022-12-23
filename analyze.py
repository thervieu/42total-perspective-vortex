from mne.datasets import eegbci # Corresponds to the EEGBCI motor imagery
from mne import Epochs, pick_channels, pick_types, find_events, read_events, events_from_annotations, annotations_from_events
from mne.io import concatenate_raws, read_raw_edf
from mne.viz import *
from mne.channels import make_standard_montage

# avoid classification of evoked responses by using epochs that start 1 second after
# cue onset.

tmin, tmax = -1., 4.
event_ids=dict(hands=2, feet=3)   # 2 -> hands   | 3 -> feet
subject = 1  # need only one
runs_move = [5, 9, 13]  # hand and feet data
runs_imagine = [6, 10, 14]  # hand and feet data

raw_files = []

# load data, change events/annotations name and concatenate in single raw
raw_fnames_move = eegbci.load_data(subject, runs_move)
raw_fnames_imagine = eegbci.load_data(subject, runs_imagine)

raw_move = concatenate_raws([read_raw_edf(fname, preload=True, stim_channel='auto') for fname in raw_fnames_move])
raw_imagine = concatenate_raws([read_raw_edf(fname, preload=True, stim_channel='auto') for fname in raw_fnames_imagine])

events_move, _ = events_from_annotations(raw_move)
events_imagine, _ = events_from_annotations(raw_imagine)

mapping_move = {1:'other', 2:'move/hands', 3:'move/feet'}
mapping_imagine = {1:'other', 2:'imagine/hands', 3:'imagine/feet'}

annot_from_events_move = annotations_from_events(
    events=events_move, event_desc=mapping_move, sfreq=raw_move.info['sfreq'],
    orig_time=raw_move.info['meas_date'])
annot_from_events_imagine = annotations_from_events(
    events=events_imagine, event_desc=mapping_imagine, sfreq=raw_imagine.info['sfreq'],
    orig_time=raw_imagine.info['meas_date'])

raw_move.set_annotations(annot_from_events_move)
raw_imagine.set_annotations(annot_from_events_imagine)

raw_files.append(raw_move)
raw_files.append(raw_imagine)

raw = concatenate_raws(raw_files)
events, _ = events_from_annotations(raw_move)

