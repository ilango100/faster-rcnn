import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import argparse

args = argparse.ArgumentParser()
args.add_argument("-e", "--epochs", type=int, dest="epochs", default=50)
args.add_argument("-np", "--no-plot", action="store_false", dest="plot")
args = args.parse_args()

# Ready the dataset
svhnb = tfds.builder("svhn_cropped")
svhnb.download_and_prepare()

bsize = 4096
svhntr = svhnb.as_dataset(split=tfds.Split.TRAIN, as_supervised=True).repeat().map(
    lambda x, y: (tf.image.random_brightness(x/255, 0.6), y)).map(
    lambda x, y: (tf.image.random_contrast(x, 0.7, 1.3), y)).map(
    lambda x, y: (tf.image.random_hue(x, 0.4), y)).map(
    lambda x, y: (x*255, y)
).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
svhnte = svhnb.as_dataset(split=tfds.Split.TEST, as_supervised=True).repeat(
).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
trsteps = svhnb.info.splits["train"].num_examples // bsize
testeps = svhnb.info.splits["test"].num_examples // bsize


# Residual block
def residual(inp, filters=128, momentum=0.9):
    inp = x = tf.keras.layers.BatchNormalization(momentum=momentum)(inp)
    x = tf.keras.layers.Conv2D(
        filters, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.add([x, inp])

    return x


# Residual reduction block
def reduction(inp, filters=128, momentum=0.9):
    inp = x = tf.keras.layers.BatchNormalization(momentum=momentum)(inp)
    inp = tf.keras.layers.Conv2D(filters, 1, 2, 'same')(inp)

    x = tf.keras.layers.Conv2D(
        filters, 3, 2, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.add([x, inp])

    return x


# Define model

inp = x = tf.keras.Input(shape=(None, None, 3))

x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.Conv2D(
    64, 5, 2, 'same', activation=tf.keras.activations.relu)(x)

x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.Conv2D(
    128, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

x = residual(x)
x = reduction(x)
x = residual(x)
x = reduction(x)
x = residual(x)
x = residual(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)(x)

x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)


# Compile the model
svhn = tf.keras.Model(inputs=inp, outputs=x)
svhn.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, metrics=[
             tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
print(svhn.summary())

# Restore the model weights
try:
    svhn.load_weights("base.h5")
except:
    pass

# Train the model
try:
    hist = svhn.fit(svhntr, steps_per_epoch=trsteps, epochs=args.epochs, validation_data=svhnte, validation_steps=testeps,
                    callbacks=[
                        tf.keras.callbacks.ReduceLROnPlateau(
                            patience=3, verbose=1),
                        tf.keras.callbacks.EarlyStopping(
                            patience=5, restore_best_weights=True)
                    ])
except:
    pass

# Save the model
svhn.save("base.h5")
print("Model saved as base.h5")

if not args.plot:
    exit()

plt.plot(hist.history["loss"], label="Train")
plt.plot(hist.history["val_loss"], label="Val")
plt.legend()
plt.title("Loss")
plt.show()

plt.plot(hist.history["acc"], label="Train")
plt.plot(hist.history["val_acc"], label="Val")
plt.legend()
plt.title("Accuracy")
plt.show()
