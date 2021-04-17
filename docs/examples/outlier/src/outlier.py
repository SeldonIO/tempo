import tensorflow as tf
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.saving import save_detector
from src.data import Cifar10
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, InputLayer, Reshape


def train_outlier_detector(data: Cifar10, artifacts_path: str):
    tf.keras.backend.clear_session()
    # define encoder and decoder networks
    latent_dim = 1024
    encoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(32, 32, 3)),
            Conv2D(64, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2D(128, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2D(512, 4, strides=2, padding="same", activation=tf.nn.relu),
        ]
    )

    decoder_net = tf.keras.Sequential(
        [
            InputLayer(input_shape=(latent_dim,)),
            Dense(4 * 4 * 128),
            Reshape(target_shape=(4, 4, 128)),
            Conv2DTranspose(256, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2DTranspose(64, 4, strides=2, padding="same", activation=tf.nn.relu),
            Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid"),
        ]
    )

    # initialize outlier detector
    od = OutlierVAE(
        threshold=0.015,  # threshold for outlier score
        encoder_net=encoder_net,  # can also pass VAE model instead
        decoder_net=decoder_net,  # of separate encoder and decoder
        latent_dim=latent_dim,
    )

    # train
    od.fit(data.X_train, epochs=50, verbose=False)

    # save the trained outlier detector
    save_detector(od, f"{artifacts_path}/outlier")
