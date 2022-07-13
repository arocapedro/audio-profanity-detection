

## Instructions
An example of all of these steps can be found [here](example_notebook.ipynb).
* Install requeriments `pip install -r requeriments.txt`

### Usage of resampler:
```bash
python -W ignore resampler.py ../cursedataset-mixed-mp3-wav/ ../cursedataset-resampled/ wav 22050
```

### Training with Spectrograms and CNNs
```py
history, model, train_ds, val_ds, test_ds = mn.main('../cursedataset-resampled', mode='specs', val_split=0.2, test_split=0.1, batch_size=128, epochs=100)
```

### Plotting training curves
```py
utils_asr.plot_metrics(history)
```
Or if you are loading from pandas:
```py
utils_asr.plot_metrics_from_pd(history_pd)
```
### Examples
An example of the usage of the functions can be found in the `src\experiments_example.ipynb` notebook.

## Parameters
### Main script
* `dataset_dir`: Location of the data (`str`).
* `mode`: Available: `transfer_yamn`, `transfer_effb0_step1` or `specs` (`str`).
* `batch_size`: Number samples per batch (`int`).
* `val_split`: Percentage of samples for validation (`float`).
* `test_split`: Percentage of samples for testing (`float`).
* `epochs`: Number of epochs (`int`).
