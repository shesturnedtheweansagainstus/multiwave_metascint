# mod1_
"""
def my_model():  # try multiple outputs?? and remove training??
    
    #For hyperparameters, return a tensorflow model
    #FIXME: make filters etc variable
    
    input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 

    stream_one = conv_layer(input, 8, 5)  # (b, 502, 8) (dim, length)
    stream_two = conv_layer(input, 8, 15)  # (b, 168, 8) 

    stream_one = tf.keras.layers.MaxPool1D(pool_size=3, strides=1)(stream_one)  # (b, 500, 8)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=3, strides=1)(stream_two)  # (b, 166, 8)

    stream_one = tf.keras.layers.Dropout(0.5)(stream_one)
    stream_two = tf.keras.layers.Dropout(0.5)(stream_two)

    stream_one = conv_layer(input, 4, 5)  # (b, 100, 4) 
    stream_two = conv_layer(input, 6, 5)  # (b, 34, 6)

    stream_one = conv_layer(input, 4, 5)  # (b, 20, 4) 
    stream_two = conv_layer(input, 4, 5)  # (b, 7, 4)

    stream_one = tf.keras.layers.MaxPool1D(pool_size=5, strides=1)(stream_one)  # (b, 16, 4)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(stream_two)  # (b, 6, 4)

    out = tf.concat([stream_two, stream_two], axis=-2)  # (b, 22, 4)

    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(0.5)(out)

    first_photon_time = tf.keras.layers.Dense(20, activation="relu", kernel_regularizer='l2')(out)
    first_photon_time = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="first_photon_time")(first_photon_time)

    total_and_energy = tf.keras.layers.Dense(30, activation="relu", kernel_regularizer='l2')(out)
    total_and_energy = tf.keras.layers.Dense(15, activation="relu", kernel_regularizer='l2')(total_and_energy)
    total_and_energy = tf.keras.layers.Dense(4, activation="relu", kernel_regularizer='l2')(total_and_energy)

    total_energy = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="total_energy")(total_and_energy)
    energy_share = tf.keras.layers.Dense(3, activation="relu", kernel_regularizer='l2', name="energy_share")(total_and_energy)

    primary_and_process = tf.keras.layers.Dense(30, activation="relu", kernel_regularizer='l2')(out)
    primary_and_process = tf.keras.layers.Dense(15, activation="relu", kernel_regularizer='l2')(primary_and_process)

    primary_pos = tf.keras.layers.Dense(3, activation="relu", name="primary_pos")(primary_and_process)
    process = tf.keras.layers.Dense(3, activation="softmax", name="process")(primary_and_process)

    return tf.keras.Model(inputs=[input], outputs=[first_photon_time, total_energy, energy_share, primary_pos, process])
"""

#mod2_

"""
def my_model():  # try multiple outputs?? and remove training??
    """
    For hyperparameters, return a tensorflow model
    FIXME: make filters etc variable
    """
    input = tf.keras.Input(shape=(2510, 1), name="input")  # (batch, 2510, 1) 

    #stream_one = conv_layer(input, 8, 5)  # (b, 502, 8) (dim, length)
    stream_one = tf.keras.layers.Conv1D(8, 5, kernel_regularizer="l2")(input)
    stream_one = tf.keras.layers.BatchNormalization()(stream_one)  #axis=-1
    stream_one = tf.keras.layers.Activation('relu')(stream_one)

    #stream_two = conv_layer(input, 8, 15)  # (b, 168, 8) 
    stream_two = tf.keras.layers.Conv1D(8, 15, kernel_regularizer="l2")(input)
    stream_two = tf.keras.layers.BatchNormalization()(stream_two)  #axis=-1
    stream_two = tf.keras.layers.Activation('relu')(stream_two)

    stream_one = tf.keras.layers.MaxPool1D(pool_size=10, strides=2)(stream_one)  # (b, 500, 8)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=10, strides=2)(stream_two)  # (b, 166, 8)

    stream_one = tf.keras.layers.Dropout(0.5)(stream_one)
    stream_two = tf.keras.layers.Dropout(0.5)(stream_two)

    #stream_one = conv_layer(input, 4, 5)  # (b, 100, 4) 
    stream_one = tf.keras.layers.Conv1D(4, 5, kernel_regularizer="l2")(stream_one)
    stream_one = tf.keras.layers.BatchNormalization()(stream_one)  #axis=-1
    stream_one = tf.keras.layers.Activation('relu')(stream_one)

    #stream_two = conv_layer(input, 6, 5)  # (b, 34, 6)
    stream_two = tf.keras.layers.Conv1D(6, 5, kernel_regularizer="l2")(stream_two)
    stream_two = tf.keras.layers.BatchNormalization()(stream_two)  #axis=-1
    stream_two = tf.keras.layers.Activation('relu')(stream_two)

    #stream_one = conv_layer(input, 4, 5)  # (b, 20, 4) 
    stream_one = tf.keras.layers.Conv1D(4, 5, kernel_regularizer="l2")(stream_one)
    stream_one = tf.keras.layers.BatchNormalization()(stream_one)  #axis=-1
    stream_one = tf.keras.layers.Activation('relu')(stream_one)

    #stream_two = conv_layer(input, 4, 5)  # (b, 7, 4)
    stream_two = tf.keras.layers.Conv1D(4, 5, kernel_regularizer="l2")(stream_two)
    stream_two = tf.keras.layers.BatchNormalization()(stream_two)  #axis=-1
    stream_two = tf.keras.layers.Activation('relu')(stream_two)

    stream_one = tf.keras.layers.MaxPool1D(pool_size=5, strides=1)(stream_one)  # (b, 16, 4)
    stream_two = tf.keras.layers.MaxPool1D(pool_size=2, strides=1)(stream_two)  # (b, 6, 4)

    out = tf.concat([stream_two, stream_two], axis=-2)  # (b, 22, 4)

    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    #out = tf.keras.layers.GRU(40, kernel_regularizer="l2", recurrent_regularizer="l2", bias_regularizer="l1")(out)  # (b, 40)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(0.5)(out)

    out = tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer='l2')(out)
    out = tf.keras.layers.Dense(1024, activation="tanh", kernel_regularizer='l2')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer="l2")(out)
    out =  tf.keras.layers.Dense(256, activation="relu", kernel_regularizer="l2")(out)

    first_photon_time = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="first_photon_time")(out)

    total_and_energy = tf.keras.layers.Dense(32, activation="tanh", kernel_regularizer='l2')(out)
    total_and_energy = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer='l2')(total_and_energy)

    total_energy = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer='l2', name="total_energy")(total_and_energy)
    energy_share = tf.keras.layers.Dense(3, activation="relu", kernel_regularizer='l2', name="energy_share")(total_and_energy)

    primary_and_process = tf.keras.layers.Dense(32, activation="tanh", kernel_regularizer='l2')(out)
    primary_and_process = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer='l2')(primary_and_process)

    primary_pos = tf.keras.layers.Dense(3, activation="relu", name="primary_pos")(primary_and_process)
    process = tf.keras.layers.Dense(3, activation="softmax", name="process")(primary_and_process)

    return tf.keras.Model(inputs=[input], outputs=[first_photon_time, total_energy, energy_share, primary_pos, process])
"""