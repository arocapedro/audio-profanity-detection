import tensorflow as tf
import tensorflow_hub as hub

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# # applies the embedding extraction model to a wav data
# def extract_spectrograms(wav_data, label):
#   ''' run YAMNet to extract embedding from the wav data '''
#   _, _, spectrogram = yamnet_model(wav_data)
#   num_embeddings = tf.shape(spectrogram)[0]
#   return (spectrogram,
#             tf.repeat(label, num_embeddings))

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  ##label_id = tf.math.argmax(label == commands)
  return spectrogram, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def create_model_spectrograms(input_shape):
    my_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # Downsample the input.
        tf.keras.layers.Resizing(32, 32),
        # Normalize.
        #norm_layer,
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
      ], name='curseword_detector')

    my_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer="adam",
                metrics = [
                      tf.keras.metrics.TruePositives(name='tp'),
                      tf.keras.metrics.FalsePositives(name='fp'),
                      tf.keras.metrics.TrueNegatives(name='tn'),
                      tf.keras.metrics.FalseNegatives(name='fn'), 
                      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall'),
                      tf.keras.metrics.AUC(name='auc'),
                      tf.keras.metrics.AUC(name='prc', curve='PR')
                    ])

    return my_model
