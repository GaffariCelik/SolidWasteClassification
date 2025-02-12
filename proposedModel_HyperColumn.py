import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, InceptionV3
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# EfficientNetB0 modelini yükleyin


def Model_HyperColumn(nb_class=6):
  # Install EfficientNetB0 model
  efficientnet_model = EfficientNetB0(include_top=False, input_shape=(224,224, 3))
  efficientnet_model.trainable=True;
  # InceptionV3 EfficientNetB0 model
  inception_model = InceptionV3( include_top=False, input_shape=(224,224, 3))
  inception_model.trainable=True;
  # Output layers 221 and 237 from EfficientNetB0
  efficientnet_layer_221 = efficientnet_model.layers[221].output
  efficientnet_layer_237 = efficientnet_model.layers[237].output

  # Output layers 248 and 310 from InceptionV3
  inception_layer_248 = inception_model.layers[248].output
  inception_layer_310 = inception_model.layers[310].output

  # Reduce size with global average pooling
  '''
  efficientnet_layer_221 = GlobalAveragePooling2D()(efficientnet_layer_221)
  efficientnet_layer_237 = GlobalAveragePooling2D()(efficientnet_layer_237)
  inception_layer_248 = GlobalAveragePooling2D()(inception_layer_248)
  inception_layer_310 = GlobalAveragePooling2D()(inception_layer_310)
  '''
    
  # Combine EfficientNetB0 and InceptionV3 layers
  efficientnet_hypercolumn = Concatenate()([efficientnet_layer_221, efficientnet_layer_237])
  efficientnet_hypercolumn = GlobalAveragePooling2D()(efficientnet_hypercolumn)
  inception_hypercolumn = Concatenate()([inception_layer_248, inception_layer_310])
  inception_hypercolumn = GlobalAveragePooling2D()(inception_hypercolumn)

  # Merge hypercolumns of two models
  combined_hypercolumn = Concatenate()([efficientnet_hypercolumn, inception_hypercolumn])

  # Add result layers
  x = Dense(512, activation='relu',name="dens1")(combined_hypercolumn)
  x = Dense(256, activation='relu',name="dens2")(x)
  x = Dense(128, activation='relu',name="dens3")(x)
  x = Dense(64, activation='relu',name="dens4")(x)
  x = Dense(nb_class, activation='softmax')(x)  

  # Create the model
  #model = Model(inputs=inputs, outputs=x)
  model =Model(inputs=[efficientnet_model.input, inception_model.input], outputs=x)

  # Compile the model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  #model.summary()
  return model