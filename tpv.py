import os, sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from mne import Epochs, pick_types, annotations_from_events, events_from_annotations, set_log_level
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP, SPoC
from mne.viz import plot_events, plot_montage

set_log_level("WARNING")

# Task: 1: left of right fist, 2: imagine left or right fist, 3: both fists or feets, 4: imagine both fists or feets
# T0 - Always rest
#    - 1, 3 real 2, 4 imagine
# T1 - 3, 7, 11 - Left fist
#    - 4, 8, 12 - Imagine left fist
#    - 5, 9, 13 - Both fists
#    - 6, 10, 14 - Imagine both fists
# T2 - 3, 7, 11 - Right fist
#    - 4, 8, 12 - Imagine right fist
#    - 5, 9, 13 - Both feets
#    - 6, 10, 14 - Imagine both feets
experiments = [
    {
        "runs": [3, 7, 11],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "runs": [4, 8, 12],
        "mapping": {0: "rest", 1: "imagine left fist", 2: "imagine right fist"},
    },
    {
        "runs": [5, 9, 13],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
    {
        "runs": [6, 10, 14],
        "mapping": {0: "rest", 1: "imagine both fists", 2: "imagine both feets"},
    },
    # {
    #     "runs": [3, 7, 11, 4, 8, 12],
    #     "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    # },
    # {
    #     "runs": [5, 9, 13, 6, 10, 14],
    #     "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    # },
]

def analyze_subject(subject_number=1, experiment=experiments[0]):
    # #############################################################################
    # # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    tmin, tmax = -1.0, 4.0

    subject_raws = []
    raw_fnames = [f"/mnt/nfs/homes/thervieu/sgoinfre/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_number:03d}/S{subject_number:03d}R{run:02d}.edf" for run in experiment["runs"]]
    experiment_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    events, _ = events_from_annotations(experiment_raw, event_id=dict(T1=1, T2=2))
    annot_from_events = annotations_from_events(
        events=events, event_desc=experiment["mapping"], sfreq=experiment_raw.info["sfreq"]
    )
    experiment_raw.set_annotations(annot_from_events)
    subject_raws.append(experiment_raw)

    raw = concatenate_raws(subject_raws)
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # Select channels
    channels = raw.info["ch_names"]
    # print(channels)
    good_channels = ["FC3", "FC1", "FCz", "FC2", "FC4",
                      "C3",  "C1", "Cz",  "C2",  "C4",
                      "CP3", "CP1", "CPz", "CP2", "CP4"]
    bad_channels = [x for x in channels if x not in good_channels]
    raw.drop_channels(bad_channels)

    # Apply band-pass filter
    raw.notch_filter(60, method="iir")
    raw.filter(7.0, 32.0, fir_design="firwin", skip_by_annotation="edge")
    # # print(raw.info)

    # Debug show data
    raw.plot_psd()
    raw.plot(n_channels=25, start=0, duration=40, scalings=dict(eeg=100e-6))
    plt.show()

    ########################################################################################################################

    # fig = plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id)
    # fig.subplots_adjust(right=0.7)  # make room for legend

    # Read epochs
    # Testing will be done with a running classifier
    events, event_id = events_from_annotations(raw)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1]
    # print("labels", labels)

    # Define a monte-carlo cross-validation generator (reduce variance):
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.3, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    # fit our pipeline to the experiment
    clf.fit(epochs_data, labels)

    # Visualize
    # csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)
    # plt.show()

    return accuracy_score(epochs.events[:, -1], clf.predict(epochs_data))

if __name__=="__main__":
    if len(sys.argv) == 2 and (sys.argv[1]=="-h" or sys.argv[1]=="--help"):
        print('usage:')
        print('\t\'python3 tpv.py subject_number=(int) subject_number=(int) mode="train"|"predict"\' to do the mode on specific subject')
        print('\t\'python3 tpv.py\' to train and predict all experiments on all subjects')
        print('\texample:\'python3 tpv.py 1 2 train\' to train and predict all experiments on all subjects')
        exit()
    scores_arr = []
    if len(sys.argv) == 1:
        score_exp = []
        four_exp_score = []
        for idx, experiment in enumerate(experiments):
            for i in range(1, 109):
                score_subject = analyze_subject(i, experiment)
                score_exp.append(score_subject)
                print(f'experiment {idx}: subject {i:03d}: accuracy {score_subject:.2%}')
            four_exp_score.append(np.mean(score_exp))

        print(f'mean accuracy of the four diffrent experiments for all 109 subjects: {four_exp_score}')
        print(f'mean accuracy of all experiments for all 109 subjects: {np.mean(four_exp_score, 0):.2%}')
        exit(0)
    else:
        if len(sys.argv) != 4:
            print('should have one argument: [subject_number] [mode="train"|"predict"]')
            exit(0)

        try:
            subject_number = int(sys.argv[1])
            print('subject_number', subject_number)
            if subject_number < 1 or subject_number > 109:
                print('error')
                raise Exception('wrong subject number')
            mode = sys.argv[2]

            score_exp = []
            for experiment in experiments:
                score_subject = analyze_subject(subject_number, experiment)
                score_exp.append(score_subject)

            print(f'accuracy for four diffrent experiments: {score_exp}')
            print(f'mean accuracy for all experiments: {np.mean(score_exp, 0):.2%}')
            exit(0)
        except:
            print('first argument should be patient number between 1 and 109')
            exit(0)
