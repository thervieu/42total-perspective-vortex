import os, sys, time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

from mne import Epochs, pick_types, annotations_from_events, events_from_annotations, set_log_level, read_epochs
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP, SPoC
from mne.viz import plot_events, plot_montage

import joblib

set_log_level("WARNING")

GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[37m"

def color_percentage(percentage: float) -> str:
    if percentage >= 0.9:
        return GREEN + f'{percentage:.2%}' + RESET
    if percentage >= 0.75:
        return YELLOW + f'{percentage:.2%}' + RESET
    return RED + f'{percentage:.2%}' + RESET

def color_truth(truth: bool) -> str:
    if truth==True:
        return GREEN + f'{truth}' + RESET
    return RED + f'{truth}' + RESET

SAVE_PATH=os.environ["SAVE_PATH"]

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
        "description": "move fists",
        "runs": [3, 7, 11],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "imagine movement of fists",
        "runs": [4, 8, 12],
        "mapping": {0: "rest", 1: "imagine left fist", 2: "imagine right fist"},
    },
    {
        "description": "move fists and feets",
        "runs": [5, 9, 13],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
    {
        "description": "imagine movement of fists and feets",
        "runs": [6, 10, 14],
        "mapping": {0: "rest", 1: "imagine both fists", 2: "imagine both feets"},
    },
    {
        "description": "movement (real or imagine) of fists",
        "runs": [3, 7, 11, 4, 8, 12],
        "mapping": {0: "rest", 1: "left fist", 2: "right fist"},
    },
    {
        "description": "movement (real or imagine) of fists or feet",
        "runs": [5, 9, 13, 6, 10, 14],
        "mapping": {0: "rest", 1: "both fists", 2: "both feets"},
    },
]

def get_data(experiment_set=0, subject_number=1, from_scratch=False) -> Epochs:
    experiment = experiments[experiment_set]
    if (from_scratch == True
        or os.path.exists(f'{SAVE_PATH}/epochs/experiment_{experiment_set}/S{subject_number:03d}_epo.fif') is False):
        # #############################################################################
        # # Set parameters and read data

        # avoid classification of evoked responses by using epochs that start 1s after
        # cue onset.
        tmin, tmax = -1.0, 4.0

        subject_raws = []
        raw_fnames = eegbci.load_data(subject_number, experiment["runs"])
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        events, _ = events_from_annotations(raw, event_id=dict(T1=1, T2=2))
        annot_from_events = annotations_from_events(
            events=events, event_desc=experiment["mapping"], sfreq=raw.info["sfreq"]
        )
        raw.set_annotations(annot_from_events)

        eegbci.standardize(raw)  # set channel names
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)

        # Select channels
        channels = raw.info["ch_names"]
        good_channels = [       "Fz",
                        "FC1", "FCz", "FC2",
                  "C3",  "C1",  "Cz",  "C2",  "C4",
                        "CP1", "CPz", "CP2",
                                "Pz",]
        # good_channels = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        #                   "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
        #                  "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
        # good_channels = ["F5",  "F3",  "F1",  "Fz",  "F2",  "F4",  "F6",
        #                 "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        #                  "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
        #                 "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        #                  "P5",  "P3",  "P1",  "Pz",  "P2",  "P4",  "P6"]
        bad_channels = [x for x in channels if x not in good_channels]
        raw.drop_channels(bad_channels)

        # Apply band-pass filter
        raw.notch_filter(60, method="iir")
        raw.filter(7.0, 32.0, fir_design="firwin", skip_by_annotation="edge")
        # # print(raw.info)

        # Debug show data
        # raw.plot_psd()
        # raw.plot(n_channels=25, start=0, duration=40, scalings=dict(eeg=100e-6))
        # plt.show()

        ########################################################################################################################

        # fig = plot_events(events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=event_id)
        # fig.subplots_adjust(right=0.7)  # make room for legend

        # Read epochs
        # Testing will be done with a running classifier
        events, event_id = events_from_annotations(raw)
        picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
        epochs.save(f'{SAVE_PATH}/epochs/experiment_{experiment_set}/S{subject_number:03d}_epo.fif', overwrite=True)
        print("Data has been transformed and saved!") if from_scratch==True else 0
    else:
        print("Tranformed data was gotten from save!") if from_scratch==True else 0
        epochs = read_epochs(f'{SAVE_PATH}/epochs/experiment_{experiment_set}/S{subject_number:03d}_epo.fif')
    
    return epochs


def get_model_and_data(epochs, experiment_set=0, subject_number=1, from_scratch=False) -> float:
    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data()
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=4.0).get_data()
    epochs_shuffled, labels_shuffled = shuffle(epochs_train, labels)

    if (from_scratch == True
        or os.path.exists(f'{SAVE_PATH}/models/experiment_{experiment_set}/S{subject_number:03d}.save') is False):
        # Assemble a classifier
        csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
        lda = LinearDiscriminantAnalysis()
        clf = Pipeline([("CSP", csp), ("LDA", lda)])

        # fit our pipeline to the experiment
        clf.fit(epochs_shuffled, labels_shuffled)

        joblib.dump(clf, f'{SAVE_PATH}/models/experiment_{experiment_set}/S{subject_number:03d}.save')
        print("Model was saved!") if from_scratch==True else 0
    else:
        # get model from models dir
        print("Model was gotten from save!") if from_scratch==True else 0
        clf = joblib.load(f'{SAVE_PATH}/models/experiment_{experiment_set}/S{subject_number:03d}.save')

    cv = ShuffleSplit(10, test_size=0.25, random_state=42)
    scores = cross_val_score(clf, epochs_shuffled[0:int(0.75*len(epochs_shuffled))], labels_shuffled[0:int(0.75*len(labels_shuffled))], cv=cv, n_jobs=None)
    # print(scores)
    # print(np.mean(scores))

    # print(accuracy_score(labels[int(0.75*len(epochs_data)):], clf.predict(epochs_data[int(0.75*len(epochs_data)):])))
    return clf, epochs_shuffled, labels_shuffled

def predict_and_get_acurracy(clf, epochs, labels, from_scratch) -> float:

    if from_scratch==True:
        print(f'epoch nb: [prediction] [truth] equal?')
        for i, prediction in enumerate(clf.predict(epochs[int(0.75*len(epochs)):])):
            print(f'epoch {i:02d}: [{prediction}] [{labels[int(0.75*len(epochs)):][i]}] {color_truth(prediction == labels[int(0.75*len(epochs)):][i])}')
            time.sleep(0.05)
    
    return accuracy_score(labels[int(0.75*len(labels)):], clf.predict(epochs[int(0.75*len(epochs)):]))


if __name__=="__main__":
    for dir_ in [f'{SAVE_PATH}', f'{SAVE_PATH}/models/',f'{SAVE_PATH}/epochs']:
        if os.path.exists(dir_) is False:
            os.mkdir(dir_)
    for i in range(0, 6):
        for subdir_ in [f'models', f'epochs']:
            if os.path.exists(f'{SAVE_PATH}/{subdir_}/experiment_{i}') is False:
                os.mkdir(f'{SAVE_PATH}/{subdir_}/experiment_{i}')

    # help
    if len(sys.argv) == 2 and (sys.argv[1]=="-h" or sys.argv[1]=="--help"):
        print('usage:')
        print('\t\'python3 tpv.py subject_number=(int) subject_number=(int) mode="train"|"predict"\' to do the mode on specific subject')
        print('\t\'python3 tpv.py\' to train and predict all experiments on all subjects')
        print('\texample:\'python3 tpv.py 1 2 train\' to train and predict all experiments on all subjects')
        exit()

    # create all models and do all predictions
    if len(sys.argv) == 1:
        four_exp_acc = []
        for i_exp in range(0, 6):
            accuracies = []
            for i in range(1, 109):
                epochs = get_data(i_exp, i, False)
                clf, epochs, labels = get_model_and_data(epochs, i_exp, i, False)
                subject_accuracy = predict_and_get_acurracy(clf, epochs, labels, False)
                accuracies.append(subject_accuracy)
                print(f'experiment {i_exp}: subject {i:03d}: accuracy = {color_percentage(subject_accuracy)}')
            four_exp_acc.append(np.mean(accuracies))
            print(f'experiment {i_exp} done: accuracy = {color_percentage(np.mean(accuracies))}\n')

        print(f'mean accuracy of the six different experiments for all 109 subjects:')
        for i, exp in enumerate(experiments[0:6]):
            print(f'\'{exp["description"]}\': {color_percentage(four_exp_acc[i])}')
        print(f'\nmean accuracy of first four experiments: {color_percentage(np.mean(four_exp_acc[0:4]))}')
        print(f'mean accuracy of all experiments: {color_percentage(np.mean(four_exp_acc))}')
        exit(0)
    else:
        if len(sys.argv) != 4:
            print('should have three argument: [subject_number (int)] [experiment_set (int)] [mode ("train"|"predict")]')
            exit(0)

        # create model of specified subject specified experiment and predict
        try:
            exp_set = int(sys.argv[1])
            if exp_set < 0 or exp_set > 5:
                raise Exception('tpv: arguments: experiment_set number: must be between 0 and 5')

            subject_nb = int(sys.argv[2])
            if subject_nb < 1 or subject_nb > 108:
                raise Exception('tpv: arguments: subject number: must be between 1 and 108')

            mode = sys.argv[3]
            if mode != "train" and mode != "predict":
                raise Exception('tpv: arguments: experiment_set number: must be \'train\' or \'predict\'')

            if mode=="train":
                epochs = get_data(exp_set, subject_nb, True)
                get_model_and_data(epochs, exp_set, subject_nb, True)
            else:
                epochs = get_data(exp_set, subject_nb, False)
                clf, epochs, labels = get_model_and_data(epochs, exp_set, subject_nb, False)
                score_subject = predict_and_get_acurracy(clf, epochs, labels, True)
                print(f'mean accuracy for all experiments: {color_percentage(score_subject)}')
            exit(0)
        except Exception as e:
            print(e)
            exit(0)
