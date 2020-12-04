import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model

import glob
import os
import random
import training_example_pb2

import cv2

DIV2K_RGB_MEAN = np.array([0.5, 0.5, 0.5]) * 255


def edsr(scale, num_filters=32, num_res_blocks=8, res_block_scaling=None):
    """Creates an EDSR model."""
    x_in = Input(shape=(480, 854, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)
    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')
    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5


def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN


class SingleFrameGenerator(object):
    def __init__(self):
        file_names = glob.glob("../data/*.binarypb")
        self.x_raw_bytes = []
        self.y_raw_bytes = []
        for fn in file_names:
            with open(fn, 'rb') as binary_file:
                texps = training_example_pb2.TrainingExamples()
                texps.ParseFromString(binary_file.read())
                for ex in texps.examples:
                    self.x_raw_bytes.append(ex.frames[-1])
                    self.y_raw_bytes.append(ex.high_res_frame)
                    if len(self.x_raw_bytes) % 2000 == 0:
                        print("Loaded " + str(len(self.x_raw_bytes)) + " frames...")
        print("Load examples completed, num total: " + str(len(self.x_raw_bytes)))

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        idx = random.randrange(len(self.x_raw_bytes))
        x_img_encoded = np.frombuffer(self.x_raw_bytes[idx], dtype=np.uint8)
        y_img_encoded = np.frombuffer(self.y_raw_bytes[idx], dtype=np.uint8)
        x_img = cv2.imdecode(
            x_img_encoded, cv2.IMREAD_COLOR).astype(np.float32)
        y_img = cv2.imdecode(
            y_img_encoded, cv2.IMREAD_COLOR).astype(np.float32)
        return x_img, y_img


if __name__ == "__main__":
    # Disable GPU.
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    ds_imgs = tf.data.Dataset.from_generator(SingleFrameGenerator, output_types=(
        tf.float32, tf.float32), output_shapes=((480, 854, 3), (1440, 2562, 3)))
    print(ds_imgs)
    ds_imgs_batch = ds_imgs.batch(6)

    model = edsr(3)
    model.summary()
    keras.utils.plot_model(model, "edsr.png", show_shapes=True)

    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=500)

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=keras.metrics.MeanAbsoluteError())

    history = model.fit(ds_imgs_batch, epochs=3,
                        steps_per_epoch=1000, callbacks=[cp_callback])

    model.save("models/single_frame_model")