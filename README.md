# total perspective vortex
## foreword
please run with a full path to a new sgoinfre dir<br>
```
export SAVE_PATH=YOUR_DOWLOAD_PATH
```
Transformed files and AI Models will be saved in this directory (GBs of data) 
## run the programs
To visualize data:<br>

```
python3 visualize.py
```
You can run all tests and get averages of predictions for all subjects and all experiences:<br>

```
python3 tpv.py
```
Or you can run one training/prediction for one subject and one experiment:<br>

```
python3 tpv.py [0 <= experience number <= 5]  [1 <= subject number <= 108] [train/predict]
```
Example:<br>
```
python3 tpv.py 2 13 train
```
## mne doesn't work
- examples do not use our dataset 
https://mne.tools/mne-realtime/auto_examples/plot_compute_rt_decoder.html#sphx-glr-auto-examples-plot-compute-rt-decoder-py
- CI is broken since a few years https://app.circleci.com/pipelines/github/mne-tools/mne-realtime