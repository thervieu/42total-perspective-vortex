import numpy as np
import matplotlib.pyplot as plt
from mne import set_log_level
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, annotations_from_events, events_from_annotations, set_log_level, annotations_from_events
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP


def show_graph():
    set_log_level("WARNING")
    subject_raws = []
    raw_fnames = eegbci.load_data(1, [3, 7, 11])
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    
    events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
    annot_from_events = annotations_from_events(
        events=events, event_desc={0: "rest", 1: "left fist", 2: "right fist"}, sfreq=raw.info["sfreq"]
    )

    raw.set_annotations(annot_from_events)

    # -----------------------------

    eegbci.standardize(raw)  # set channel names
    
    montage = make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing='ignore')

    montage.plot()
    plt.show()

    raw.plot(scalings=dict(eeg=200e-6))
    plt.show()

    raw.compute_psd().plot()
    plt.show()

    raw.plot_psd(average=True)
    plt.show()

    # Select channels
    channels = raw.info["ch_names"]
    good_channels = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                      "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
                     "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    raw.compute_psd().plot()
    plt.show()

    # Apply band-passfilter
    raw.notch_filter(60, method="iir")
    raw.compute_psd().plot()
    plt.show()

    raw.filter(7.0, 32.0, fir_design="firwin")
    raw.compute_psd().plot()
    plt.show()

    raw.plot(scalings=dict(eeg=200e-6))
    plt.show()

    exit()

if __name__ == '__main__':
    set_log_level('CRITICAL')
    
    show_graph()