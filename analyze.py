import matplotlib.pyplot as plt

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
runs_move = [5, 9, 13]  # hands and feet move
runs_imagine = [6, 10, 14]  # hands and feet imagine

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
events, event_dict = events_from_annotations(raw_move)

# channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
# raw.drop_channels(['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.'])


# standardize channel names to put colors in next graph
eegbci.standardize(raw)
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

print(raw.info.ch_names)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
raw.plot_psd(average=False)
plt.show()

fig = plot_events(events, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_dict)
fig.subplots_adjust(right=0.7)

raw.filter(5., 40.)

raw.plot_psd(average=False)
plt.show()


tmin, tmax = -1., 4.

event_id = {'move/hands': 1, 'move/feet': 2, 'imagine/hands': 3, 'imagine/feet': 4}
events, event_dict = events_from_annotations(raw, event_id=event_id)
print (event_dict)


from mne.decoding import LinearModel, Vectorizer, get_coef, Scaler, CSP, SPoC, UnsupervisedSpatialFilter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.model_selection import train_test_split

score = -1
model = None

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
labels = epochs.events[:, -1] - 1
print(labels)

scores = []
labels = epochs.events[:, -1] - 1
epochs_data = epochs.get_data()
print(epochs_data)

cv = ShuffleSplit(10, test_size=0.4, random_state=42)
print(cv)

csp = CSP(n_components=10, reg=None, log=True, norm_trace=False)
lda = LinearDiscriminantAnalysis()

clf = make_pipeline(csp, lda)
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=None, error_score='raise')

print('Accuracy: {:2.2}% (+/- {:0.2}%)'.format(scores.mean(), scores.std()))


log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
