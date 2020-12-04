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

DIV2K_RGB_MEAN_3 = np.array([0.5, 0.5, 0.5]) * 255
DIV2K_RGB_MEAN_15 = np.ones((15), dtype=np.float32) * 0.5 *255

class MultiFrameGenerator(object):
    def __init__(self):
        file_names = glob.glob("../data/*.binarypb")
        self.x_raw_bytes = []
        self.y_raw_bytes = []
        for fn in file_names:
            with open(fn, 'rb') as binary_file:
                texps = training_example_pb2.TrainingExamples()
                texps.ParseFromString(binary_file.read())
                for ex in texps.examples:
                    self.x_raw_bytes.append(ex.frames)
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
        x_imgs = np.zeros((480, 854, 15), dtype=np.float32)
        for i in range(5):
            x_img_encoded = np.frombuffer(self.x_raw_bytes[idx][i], dtype=np.uint8)
            x_img = cv2.imdecode(x_img_encoded, cv2.IMREAD_COLOR).astype(np.float32)
            x_imgs[:, :, 3*i:(3*i+3)] = x_img
        y_img_encoded = np.frombuffer(self.y_raw_bytes[idx], dtype=np.uint8)
        y_img = cv2.imdecode(
            y_img_encoded, cv2.IMREAD_COLOR).astype(np.float32)
        return x_imgs, y_img



if __name__ == "__main__":
    ds_imgs = tf.data.Dataset.from_generator(MultiFrameGenerator, output_types=(
        tf.float32, tf.float32), output_shapes=((480, 854, 15), (1440, 2562, 3)))
    print(ds_imgs)

    model = tf.keras.models.load_model('models/multi_frame_model')
    model.summary()

    eval_dir = "../eval/multi/"
    i = 0
    for x_eval, _ in ds_imgs.take(10):
        print(x_eval.shape)
        x_eval_resized = cv2.resize(x_eval[:, :, 12:15].numpy().astype(
            np.uint8), None, None, 3, 3, cv2.INTER_AREA)
        cv2.imshow("HR", x_eval_resized)
        cv2.waitKey()
        cv2.imwrite(eval_dir + str(i)+"_resized.png", x_eval_resized)
        x_eval = np.expand_dims(x_eval, axis=0)
        print(x_eval.shape)
        y_pred = model.predict(x_eval)
        y_pred = np.clip(y_pred, 0, 255)
        y_pred_im = y_pred[0, :, :, :].astype(np.uint8)
        cv2.imshow("HR", y_pred_im)
        cv2.waitKey()
        cv2.imwrite(eval_dir + str(i)+"_predicted.png", y_pred_im)
        i = i+1
