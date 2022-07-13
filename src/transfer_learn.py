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
  return tf.reshape(embeddings, [tf.shape(embeddings)[0]*tf.shape(embeddings)[1]]), label
  

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

def convert_to_rgb(spec, label):
  #spec = tf.keras.layers.Rescaling(spec)(255/tf.math.reduce_max(spec))
  return tf.image.grayscale_to_rgb(spec), label

def create_pretrained_efficientnet_model(input_shape):
    efficientnet_layer = tf.keras.applications.EfficientNetB0(
            include_top= False,
            weights="imagenet",
            input_shape=input_shape)
    efficientnet_layer.trainable = False
    
    my_model = tf.keras.Sequential([
        efficientnet_layer,
        tf.keras.layers.GlobalAveragePooling2D(name = "avg_pool"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='selu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
    ])
    
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
    #my_model.build()
    return my_model

def start_fit(model, train_ds, val_ds, epo=20, weights=None):
    callbacks = [
          tf.keras.callbacks.EarlyStopping(
              monitor='val_loss',
              patience=3,
              restore_best_weights=True
            ),
          tf.keras.callbacks.TensorBoard(
              log_dir = 'logs'
            ), 
          tf.keras.callbacks.ModelCheckpoint(
            filepath='../models/ckp',
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
            ),
        ]
    history = model.fit(train_ds,
                      epochs=epo,
                      validation_data=val_ds,
                      callbacks=callbacks,
                      class_weight=weights)
    return history


