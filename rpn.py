import math
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv

from anchors import Anchor
n_anchors = len(Anchor.anchors)
train_steps = len(tf.io.gfile.glob("train/*.png"))
val_steps = len(tf.io.gfile.glob("test/*.png"))


def data_gen(split=b"train"):

    for file in tf.io.gfile.glob("anchors-"+split.decode()+"/*.csv"):

        # Generate image file
        img = file[8:-4]
        img = cv.imread(img)[:, :, ::-1]

        # Pseudo "batchify" the input image
        img.shape = (1, *img.shape)

        # Read csv and sample positives
        df = pd.read_csv(file)
        pos = df.iou > 0.7
        lpos = pos.sum()
        if lpos <= 0:
            pos = df.sort_values("iou", ascending=False)[:32].index
            lpos = 32
        neg = df.iou < 0.3
        eq = min(lpos, neg.sum())
        sample = df.loc[pos].sample(eq)
        sample["label"] = 1

        # Regression target (only for positive samples)
        reg = sample[["y", "x", "i", "dx", "dy", "dw", "dh"]].values
        reg[:, 2] = reg[:, 2]*4
        dx = reg[:, :4]

        dy = reg[:, :3]
        dy[:, 2] = dy[:, 2]+1
        dy = np.concatenate([dy, reg[:, 4:5]], axis=1)

        dw = reg[:, :3]
        dw[:, 2] = dw[:, 2]+2
        dw = np.concatenate([dw, reg[:, 5:6]], axis=1)

        dh = reg[:, :3]
        dh[:, 2] = dh[:, 2]+3
        dh = np.concatenate([dh, reg[:, 6:7]], axis=1)

        reg = np.concatenate([dx, dy, dw, dh])

        # Negative sample
        negsample = df[neg].sample(eq)
        negsample["label"] = 0
        sample = sample.append(negsample, ignore_index=True)

        # Classification target
        clsf = sample[["y", "x", "i", "label"]].values

        yield img, (clsf, reg)


# Load the datasets
train = tf.data.Dataset.from_generator(data_gen, (tf.int64, (tf.int64, tf.float32)),
                                       ((None, None, None, 3), ((None, 4), (None, 4))), ["train"]).repeat()
test = tf.data.Dataset.from_generator(data_gen, (tf.int64, (tf.int64, tf.float32)),
                                      ((None, None, None, 3), ((None, 4), (None, 4))), ["test"]).repeat()


# Load the trained base model
base = tf.keras.models.load_model("svhn.h5", compile=False)
# print(base.summary()) # to see all layers


# Extract the feature map from intermediate layer
features = base.get_layer("add_5").output

# RPN
# Classification output that predicts the probability of the given anchor being a digit
# Apply sigmoid because output is probability
clsf = tf.keras.layers.Conv2D(
    n_anchors, 3, 1, 'same', activation='sigmoid', name="rpn_clsf")(features)

# Regression output that predicts the offsets from given anchor box
# No relu since values may be negative
reg = tf.keras.layers.Conv2D(
    n_anchors*4, 3, 1, 'same', name="rpn_reg")(features)


# Create the model
rpn = tf.keras.Model(inputs=base.input, outputs=[clsf, reg])


# - For classification, crossentropy loss.
def pos_crossentropy(labels, pred):
    xyi = tf.cast(labels[:, :3], tf.int32)

    # p = pred[xyi]
    p = tf.gather_nd(pred[0], xyi)
    l = tf.cast(labels[:, 3], tf.float32)

    return tf.keras.losses.binary_crossentropy(l, p)


# - For regression, huber loss or smooth l1 loss.
def pos_huber(labels, pred):
    xyi = tf.cast(labels[:, :3], tf.int32)

    # p = pred[xyi]
    p = tf.gather_nd(pred[0], xyi)
    l = tf.cast(labels[:, 3], tf.float32)

    try:
        return tf.keras.losses.Huber()(l, p)
    except:
        return tf.losses.huber_loss(l, p)


rpn.compile("adam", [pos_crossentropy, pos_huber],
            target_tensors=[
                tf.placeholder(dtype=tf.int64, shape=(None, 4)),
                tf.placeholder(dtype=tf.float32, shape=(None, 4)),
])


try:
    rpn.fit(train, steps_per_epoch=train_steps,
            validation_data=test, validation_steps=val_steps)
except:
    print("Model will be saved as rpn.h5")

rpn.save("rpn.h5")
print("Model saved as rpn.h5")
