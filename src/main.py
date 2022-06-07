import os
import argparse

import tensorflow as tf
from pathlib import Path

from utils_asr import create_splits, load_wav_for_map,start_fit,get_class_weights,configure_for_performance
from transfer_learn import create_model_small,extract_embedding_yamn
from spectrograms import create_model_spectrograms,get_spectrogram_and_label_id

AUTOTUNE = tf.data.AUTOTUNE


def main(dataset_dir='../cursedataset-resampled', mode=None, batch_size=32, val_split=0.2, test_split=0.1, epochs=25):

  if dataset_dir[-1] == '*': dataset_dir=dataset_dir+os.sep
  print("Using path: ", dataset_dir)

  sample_count = len(list(Path(dataset_dir).glob(f'*{os.sep}*{os.sep}*')))
  print("Number of samples: ", sample_count)

  main_ds = tf.data.Dataset.list_files(str(f'{dataset_dir}{os.sep}*{os.sep}*{os.sep}*'), shuffle=True)

  #labels_ds = main_ds.map(lambda x: transfer_learn.get_label(x[1:2])[0])
  main_ds = main_ds.map(load_wav_for_map, num_parallel_calls=AUTOTUNE)
  #main_ds = transfer_learn.configure_for_performance(main_ds, 128)
 
  if mode == 'transfer_yamn':
    main_ds = main_ds.map(extract_embedding_yamn, num_parallel_calls=AUTOTUNE)
    for element, _ in main_ds.take(1):
      input_shape = element.numpy().shape
      print('Input shape:', input_shape)
    classifier_model = create_model_small(input_shape)

  # if mode == 'transfer_wav2vec':
  #   main_ds = main_ds.map(extract_embedding_wav2vec, num_parallel_calls=AUTOTUNE)
  #   for element, _ in main_ds.take(1):
  #     input_shape = element.numpy().shape
  #     print('Input shape:', input_shape)
  #   classifier_model = create_model_small(input_shape)

  elif mode == 'specs':
    main_ds = main_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    for spectrogram, _ in main_ds.take(1):
      input_shape = spectrogram.numpy().shape
      print('Input shape:', input_shape)
    classifier_model = create_model_spectrograms(input_shape)
  else:
    print("Unkown mode: ", mode)
    return


  print(classifier_model.summary())
  train_ds, val_ds, test_ds = create_splits(main_ds, sample_count, val_split, test_split)

  train_ds = configure_for_performance(train_ds, batch_size)
  val_ds = configure_for_performance(val_ds, batch_size)

  class_weights = get_class_weights(dataset_dir)

  for spectrogram, label in val_ds.take(1):
      input_shape = spectrogram.numpy().shape
      print(f"First sample size ({input_shape}), label ({label})")
  

  history = start_fit(classifier_model, train_ds=train_ds, val_ds=val_ds, weights=class_weights, epo=epochs)

  return history, classifier_model, train_ds, val_ds, test_ds




if __name__ == '__main__':
    # dataset_dir='../cursedataset-resampled/robot', mode=None, batch_size=32, val_split=0.2, test_split=0.1, epochs=25
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('--mode')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_split', type=float)
    parser.add_argument('--test_split', type=float)
    parser.add_argument('--epochs', type=int)

    args = parser.parse_args()

    main(**vars(args))