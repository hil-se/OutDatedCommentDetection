import tensorflow as tf
import matplotlib.pyplot as plt
from DataReader import Reader
from DualEncoder import DualEncoderAll, create_encoder
import numpy as np
import pandas as pd
import time
import datetime

pd.set_option('display.max_columns', None)
DATA_LOAD_PATH = ""


def validate_and_filter_features(features_dict):
    for key, value in features_dict.items():
        if value is None or np.any([v is None for v in value]):
            features_dict[key] = np.array([v for v in value if v is not None])
            if len(features_dict[key]) == 0:
                raise ValueError(f"All values are None in feature '{key}'")
    return features_dict


def learn(train_data,
          output_size=300,
          epochs=600,
          validation_data=None,
          num_of_layers=1,
          dropout_rate=0.3,
          temperature=0.05,
          lr=0.001,
          decay_steps=10,
          decay_rate=0.96,
          patience=15,
          model_name="model"):
    # Initialize encoder
    encoder = None

    
    
    input_size = train_data.element_spec['input_layer'].shape[1]
    encoder = create_encoder(
            num_of_layers,
            input_size=input_size,  # Corrected the argument name
            output_size=output_size,
            dropout_rate=dropout_rate
        )

    # Set up learning rate schedule and compile model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
    )
    encoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error')
    # Set up early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    # Train the model
    encoder.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[early_stopping],
    )

    encoder.save(f"{model_name}.h5")
    print(f"Model saved as {model_name}.h5")

    return encoder


def train_model(train, val, output_size=5000, batch_size=5000, epochs=300, num_of_layers=1, model_name="model"):
    
    # Shuffle the datasets
    np.random.shuffle(train.values)
    np.random.shuffle(val.values)
    print(len(train.index), len(val.index))

    # Initialize features for train and validation datasets
    train_features = {}
    val_features = {}

    # Check for columns in train and add them to features
    if "Source_Old" in train.columns:
        td_s = train["Source_Old"].to_list()
        train_features["input_layer"] = np.array([emb for emb in td_s])
        

    if "Target_Old" in train.columns and "Target_New" in train.columns:
        td_to = train["Target_Old"].to_list()
        td_tn = train["Target_New"].to_list()
        train_features["input_layer"] = np.array([emb for emb in td_to])
        train_features["input_layer"] = np.array([emb for emb in td_tn])
        

    # Validate and filter features
    train_features = validate_and_filter_features(train_features)

    if not train_features:
        raise ValueError("No valid columns ('Source_Old', 'Target_Old', 'Target_New') found in the training dataset.")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_features).batch(batch_size)

    # Check for columns in validation and add them to features
    if "Source_Old" in val.columns:
        v_s = val["Source_Old"].to_list()
        val_features["input_layer"] = np.array([emb for emb in v_s])

    if "Target_Old" in val.columns and "Target_New" in val.columns:
        td_to = val["Target_Old"].to_list()
        td_tn = val["Target_New"].to_list()
        val_features["target_old"] = np.array([emb for emb in td_to])
        val_features["target_new"] = np.array([emb for emb in td_tn])

    # Validate and filter features
    val_features = validate_and_filter_features(val_features)

    if not val_features:
        raise ValueError("No valid columns ('Source_Old', 'Target_Old', 'Target_New') found in the validation dataset.")

    val_dataset = tf.data.Dataset.from_tensor_slices(val_features).batch(batch_size)

    # Train the dual encoder model
    dual_encoder = learn(train_dataset, output_size=output_size, epochs=epochs,
                         validation_data=val_dataset, num_of_layers=num_of_layers, model_name=model_name)

    return dual_encoder


def allLanguageExperiment(languages,
                          embedding="cbow",
                          output_size=1000,
                          batch_size=1000,
                          test_set_size=1000,
                          num_of_layers=1,
                          threshold=0.5,
                          number_of_experiments=5):
    for lang in languages:
        print(lang.upper())
        r = Reader(embedding)
        r.load("embeddings/java_train")
        train = pd.concat([r.source_series_old], axis=1)

        r = Reader(embedding)
        r.load("embeddings/java_valid")
        val = pd.concat([r.source_series_old], axis=1)

        print("training source encoder")
        print("output_size", "batch_size", "thres")
        print(output_size, batch_size, threshold)
        print("*****************************")

        start_tr = time.time()
        np.random.shuffle(train.values)
        np.random.shuffle(val.values)
        train_model(train, val, output_size, batch_size, num_of_layers=num_of_layers,
                    model_name=f"encoder_{lang}_source")
        end_tr = time.time()
        tr_time = round(end_tr - start_tr, 3)
        print("Training time:", tr_time)

        print("End")

        r = Reader(embedding)
        r.load("java_train")
        train = pd.concat([r.target_series_old, r.target_series_new], axis=1)

        r = Reader(embedding)
        r.load("java_valid")
        val = pd.concat([r.target_series_old, r.target_series_new], axis=1)

        print("training target encoder")
        print("output_size", "batch_size", "thres")
        print(output_size, batch_size, threshold)
        print("*****************************")

        start_tr = time.time()
        np.random.shuffle(train.values)
        np.random.shuffle(val.values)
        train_model(train, val, output_size, batch_size, num_of_layers=num_of_layers,
                    model_name=f"encoder_{lang}_target")
        end_tr = time.time()
        tr_time = round(end_tr - start_tr, 3)
        print("Training time:", tr_time)

        print("End")


languages = ["java"]
allLanguageExperiment(languages=languages,
                      embedding="cbow",
                      output_size=2000,
                      batch_size=2000,
                      test_set_size=1000,
                      threshold=0.5,
                      num_of_layers=1,
                      number_of_experiments=25)
