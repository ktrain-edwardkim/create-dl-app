import numpy as np
from preprocessing import load_dataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # Make NumPy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)

    dataset = load_dataset()
    # Split the data into train and test sets
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    # train_features.info()
    # exit()
    # Separate the target value, the "label", from the features.
    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    # Normalize the data
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))


    def build_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(train_features.shape[1],)),
            normalizer,
            layers.Dense(units=1)
        ])
        return model


    # Create a new model
    model = build_model()
    model.summary()

    # Define loss function and optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    # Training
    history = model.fit(
        train_features,
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2)

    print(history.history['loss'][-1])

    # Print test loss
    test_results = model.evaluate(
        test_features,
        test_labels,
        verbose=0)
    print(test_features.shape)
    print(test_results)

    # Save the model
    model.save('models/model_tf')
    model.save_weights('./models/checkpoint')


if __name__ == '__main__':
    main()