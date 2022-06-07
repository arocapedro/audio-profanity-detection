import tensorflow as tf
import tensorflow_hub as hub

AUTOTUNE = tf.data.AUTOTUNE
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# wav2vec_model_handle = 'https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1'
# wav2vec_model_model = hub.KerasLayer(wav2vec_model_handle, trainable=True)


# applies the embedding extraction model to a wav data
def extract_embedding_yamn(wav_data, label):
  ''' run YAMNet to extract embedding from the wav data '''
  _, embeddings, _ = yamnet_model(wav_data)
  #num_embeddings = tf.shape(embeddings)[0]
  return tf.reshape(embeddings, [tf.shape(embeddings)[0]*tf.shape(embeddings)[1]]), label
  

# def create_wav2vec_model(input_shape):
#   # For using this model, it's important to set `jit_compile=True` on GPUs/CPUs
#   # as some operations in this model (i.e. group-convolutions) are unsupported without it
#   pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)
#   my_model = tf.keras.Sequential([
#         wav2vec_model,
#         tf.keras.layers.Dense(512, activation='selu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
#   my_model.build(input_shape)
#   my_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                 optimizer="adam",
#                 jit_compile=True,
#                 metrics = [
#                       tf.keras.metrics.TruePositives(name='tp'),
#                       tf.keras.metrics.FalsePositives(name='fp'),
#                       tf.keras.metrics.TrueNegatives(name='tn'),
#                       tf.keras.metrics.FalseNegatives(name='fn'), 
#                       tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#                       tf.keras.metrics.Precision(name='precision'),
#                       tf.keras.metrics.Recall(name='recall'),
#                       tf.keras.metrics.AUC(name='auc'),
#                       tf.keras.metrics.AUC(name='prc', curve='PR')
#                     ])
#   return my_model

def create_model_small(input_shape):
    my_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape, dtype=tf.float32,
                          name='input_embedding'),
    tf.keras.layers.Dense(1024,activation='selu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='selu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='selu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
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

def start_fit(model, train_ds, val_ds, epo=20, weights=None):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            restore_best_weights=True)
    history = model.fit(train_ds,
                      epochs=epo,
                      validation_data=val_ds,
                      callbacks=callback,
                      class_weight=weights)
    return history

class ReduceMeanLayer(tf.keras.layers.Layer):
  def __init__(self, axis=0, **kwargs):
    super(ReduceMeanLayer, self).__init__(**kwargs)
    self.axis = axis

  def call(self, input):
    return tf.math.reduce_mean(input, axis=self.axis)


