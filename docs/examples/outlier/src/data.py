import tensorflow as tf


class Cifar10(object):
    def __init__(
        self,
    ):
        train, test = tf.keras.datasets.cifar10.load_data()
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test

        self.X_train = self.X_train.astype("float32") / 255
        self.X_test = self.X_test.astype("float32") / 255
        print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)
        self.class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
