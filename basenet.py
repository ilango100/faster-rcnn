import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import argparse

args = argparse.ArgumentParser(description="Train the base network")
args.add_argument("-e", "--epochs", type=int, dest="epochs", default=50)
args.add_argument("-b", "--batch", type=int, dest="batch_size", default=128)
args.add_argument("-np", "--no-plot", action="store_true", dest="no_plot")
args = args.parse_args()

# Ready the dataset
svhnb = tfds.builder("svhn_cropped", data_dir="D:\\MachineLearning\\tfds")
svhnb.download_and_prepare()

svhntr = svhnb.as_dataset(split=tfds.Split.TRAIN, as_supervised=True).repeat().map(
    lambda x, y: (tf.image.random_brightness(x/255, 0.6), y)).map(
    lambda x, y: (tf.image.random_contrast(x, 0.7, 1.3), y)).map(
    lambda x, y: (tf.image.random_hue(x, 0.4), y)).map(
    lambda x, y: (x*255, y)
).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
svhnte = svhnb.as_dataset(split=tfds.Split.TEST, as_supervised=True).repeat(
).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
trsteps = svhnb.info.splits["train"].num_examples // args.batch_size
testeps = svhnb.info.splits["test"].num_examples // args.batch_size


# Residual block
def residual(inp, filters=128, momentum=0.9):
    # Use BN only for residual and not for skip.
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(inp)
    x = tf.keras.layers.Conv2D(
        filters, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same')(x)
    # Use relu before addition and end up always adding positive residual!

    x = tf.keras.layers.add([x, inp])

    return x


# Residual reduction block
def reduction(inp, filters=128, momentum=0.9):
    # Use BN only for residual and not for skip.
    x = tf.keras.layers.BatchNormalization(momentum=momentum)(inp)

    x = tf.keras.layers.Conv2D(
        filters, 3, 2, 'same', activation=tf.keras.activations.relu)(x)

    x = tf.keras.layers.BatchNormalization(momentum=momentum)(x)
    x = tf.keras.layers.Conv2D(filters, 3, 1, 'same')(x)
    # Never use relu before addition, since output is always positive and explodes on addition.

    inp = tf.keras.layers.Conv2D(filters, 1, 2, 'same')(inp)
    x = tf.keras.layers.add([x, inp])

    return x


# Define model

inp = x = tf.keras.Input(shape=(None, None, 3))

x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.Conv2D(
    64, 5, 2, 'same', activation=tf.keras.activations.relu)(x)

# x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
# x = tf.keras.layers.Conv2D(
#     128, 3, 1, 'same', activation=tf.keras.activations.relu)(x)

x = residual(x, 64)
x = reduction(x, 128)
x = residual(x, 128)
x = reduction(x, 256)
x = residual(x, 256)
x = residual(x, 256)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)

x = tf.keras.layers.Dropout(rate=0.4)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)


# Compile the model
svhn = tf.keras.Model(inputs=inp, outputs=x)
svhn.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, metrics=[
             tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
svhn.summary()

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
                            patience=10, restore_best_weights=True)
                    ])
    errored = False
except:
    errored = True

# Save the model
svhn.save("base.h5")
print("Model saved as base.h5")

if errored or args.no_plot:
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
