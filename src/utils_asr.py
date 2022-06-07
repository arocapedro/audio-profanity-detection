import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
class_names = ["clean", "curse"]
voice_names = ["human", "robot"]

AUTOTUNE = tf.data.AUTOTUNE

def get_class_weights(directory_path):
  # curse_words_len = len(list(label_ds.filter(lambda x, y, z: (y[0] == 1)).as_numpy_iterator()))
  # clean_words_len = sample_size-curse_words_len
  clean_words_len = len(list(Path(directory_path).glob(f'*{os.sep}{class_names[0]}{os.sep}*')))
  curse_words_len = len(list(Path(directory_path).glob(f'*{os.sep}{class_names[1]}{os.sep}*')))

  neg, pos = clean_words_len, curse_words_len
  total = neg + pos
  print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
      total, pos, 100 * pos / total))
  
  # Scaling by total/2 helps keep the loss to a similar magnitude.
  # The sum of the weights of all examples stays the same.
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)

  class_weight = {0: weight_for_0, 1: weight_for_1}

  print('Weight for class 0: {:.2f}, {}'.format(weight_for_0, class_names[0]))
  print('Weight for class 1: {:.2f}, {}'.format(weight_for_1, class_names[1]))

  return class_weight

def create_splits(main_ds, sample_count, val_ratio, test_ratio): #-> Tuple[tf.Dataset, tf.Dataset, tf.Dataset, int, int, int]:
  val_size = int(sample_count * val_ratio)
  test_size = int(sample_count * test_ratio)
  train_size = sample_count - val_size - test_size

  train_dataset = main_ds.take(train_size)
  test_dataset = main_ds.skip(train_size)

  val_dataset = test_dataset.skip(val_size)
  test_dataset = test_dataset.take(test_size)
  print(f"Created splits: train [{train_size}] val [{val_size}] test [{test_size}]. "+ \
            f"Sanity check [{(sample_count-val_size-test_size-train_size)==0}]")
    
  return train_dataset, val_dataset, test_dataset#, train_size, val_size, test_size

def start_fit(model, train_ds, val_ds, epo=20, weights=None):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            restore_best_weights=True)
    history = model.fit(train_ds,
                      epochs=epo,
                      validation_data=val_ds,
                      callbacks=callback,
                      class_weight=weights)
    return history

def load_wav_for_map(filename, sr=16000):
  labels = get_label(filename)
  return load_wav_16k_mono(filename, sr=sr), labels[0], labels[1]

@tf.function
def load_wav_16k_mono(filename, sr=16000):
    """ Load a WAV file, convert it to a float tensor, resample to SR kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sr_in = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sr_in = tf.cast(sr_in, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sr_in, rate_out=sr)

    # Remove silence
    position = tfio.audio.trim(wav, axis=0, epsilon=0.1)
    start = position[0]
    stop = position[1]
    waveform = wav[start:stop]

    # Fix size+padding
    waveform = wav[:int(sr*1.5)] 
    zero_padding = tf.zeros(
        [int(sr*1.5)] - tf.shape(waveform),
        dtype=tf.float32)
    
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    return equal_length

@tf.function
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = 1 if parts[-2] == 'curse' else 0
  # The third to last is the voice-type-directory
  voice = parts[-3] == voice_names
  # Integer encode the label
  return one_hot, tf.argmax(voice)

def load_wav_for_map(filename, sr=16000):
  labels = get_label(filename)
  return load_wav_16k_mono(filename, sr=sr), labels[0]


def create_yamm_model(my_model=None, yamnet_model_handle=None):
    input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                                trainable=False, name='yamnet')
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = my_model(embeddings_output)
    serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
    serving_model = tf.keras.Model(input_segment, serving_outputs)
    tf.keras.utils.plot_model(serving_model)
    return serving_model

def configure_for_performance(ds, batch_size):
  #ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def plot_training(history, metric='loss'):
  metrics = history.history
  plt.plot(history.epoch, metrics[metric], metrics[f'val_{metric}'])
  plt.legend([metric, f'val_{metric}'])
  plt.show()

def plot_roc(name, fp, tp, **kwargs):
  plt.plot(fp, tp, label=name, linewidth=5, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  plt.figure(figsize=(15,12))
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[1], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


def inference(test_ds, classifier, threshold = 0.5):
  predictions = []
  test_labels = []

  for sample in tqdm(test_ds):
    #print(np.array([sample[0]]).shape)
    predictions.append(classifier.predict(np.array([sample[0]]), verbose=0))
    test_labels.append(sample[1].numpy())
  
  test_labels = np.array(test_labels)
  
  if len(np.array(predictions).shape) > 3:
    predictions = np.mean(predictions, axis=2)
    test_labels = test_labels[:,0]

  
  y_pred = [1 if f > threshold else 0 for f in predictions]
  y_true = test_labels

  test_acc = sum(y_pred == y_true) / len(y_true)
  print(f'Test set accuracy: {test_acc:.0%}')

  confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_mtx,
              xticklabels=["clean", "curse"],
              yticklabels=["clean", "curse"],
              annot=True, fmt='g')
  plt.xlabel('Prediction')
  plt.ylabel('Label')
  plt.show()


def save_history(history, output_path):
  # convert the history.history dict to a pandas DataFrame:     
  hist_df = pd.DataFrame(history.history) 

  # save to csv: 
  hist_csv_file = output_path
  with open(hist_csv_file, mode='w') as f:
      hist_df.to_csv(f)
