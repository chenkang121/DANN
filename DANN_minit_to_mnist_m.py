import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle as plk
from tensorflow import keras

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
# print(train_image.shape)
train_image = train_image.reshape((-1, 28, 28, 1)).astype('float32')
test_image = test_image.reshape((-1, 28, 28, 1)).astype('float32')
train_label = tf.one_hot(train_label, 10)
test_label = tf.one_hot(test_label, 10)
print('train_label[0]: ', train_label[0])
train_image = tf.repeat(train_image, 3, axis=3)
test_image = tf.repeat(test_image, 3, axis=3)
print(train_image.shape)

# plt.imshow(train_image[0]/255)
# plt.show()

f = open('dataset\mnistm\mnistm_data.pkl', 'rb')
mnistm = plk.load(f)
# print(mnistm['train'][0])

(m_train_image, m_train_label), (m_test_image, m_test_label) = (mnistm['train'], mnistm['train_label']), (mnistm['test'], mnistm['test_label'])
m_train_image = m_train_image.astype('float32')
m_test_image = m_test_image.astype('float32')

m_train_label = tf.one_hot(m_train_label, 10)
m_test_label = tf.one_hot(m_test_label, 10)
# plt.imshow(m_train_image[0]/255)
# plt.show()


def get_feature_extract_model():
    model = keras.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                            padding='valid'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=48, kernel_size=(5, 5), activation='relu',
                            padding='valid'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    return model


def get_discriminator():
    model = keras.Sequential()
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(128, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(32, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1, use_bias=False))

    return model


def get_classifier():
    model = keras.Sequential()
    model.add(layers.Dense(100, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(100, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Bcross_entropy = tf.losses.CategoricalCrossentropy()
Bcross_entropy = tf.losses.BinaryCrossentropy(from_logits=True)
Ccross_entropy = tf.losses.CategoricalCrossentropy()


def discriminator_loss(nomal_feature_out, m_feature_out):
    nomal_feature_loss = Bcross_entropy(tf.ones_like(nomal_feature_out), nomal_feature_out)
    m_feature__loss = Bcross_entropy(tf.zeros_like(m_feature_out), m_feature_out)
    return nomal_feature_loss + m_feature__loss


def discriminator_loss_(nomal_feature_out, m_feature_out):
    nomal_feature_loss = Bcross_entropy(tf.zeros_like(nomal_feature_out), nomal_feature_out)
    m_feature__loss = Bcross_entropy(tf.ones_like(m_feature_out), m_feature_out)
    return nomal_feature_loss + m_feature__loss


def classifier_loss(y_pred_n, y_labels_n):
    loss_n = Ccross_entropy(y_labels_n, y_pred_n)

    return loss_n


lr = 0.7e-4
# extractor_opt = tf.train.
extractor_opt = keras.optimizers.Adam(lr)
# extractor_opt = keras.optimizers.
classifier_opt = keras.optimizers.Adam(lr)
discriminator_opt = keras.optimizers.Adam(lr)

EPOCH = 100

gama = 10


extractor = get_feature_extract_model()
discriminator = get_discriminator()
classifier = get_classifier()


def train_step(normal_images, m_images, y_labels_n, ylabels_m, la, mu):

    with tf.GradientTape() as ext_tape,\
         tf.GradientTape() as disc_tape,\
         tf.GradientTape() as clas_tape:
        normal_features = extractor(normal_images, training=True)
        m_features = extractor(m_images, training=True)

        normal_feature_out = discriminator(normal_features)
        m_feature_out = discriminator(m_features)
        disc_loss = discriminator_loss(normal_feature_out, m_feature_out)
        disc_loss_ = discriminator_loss_(normal_feature_out, m_feature_out)

        y_pred_n = classifier(normal_features)

        clas_loss = classifier_loss(y_pred_n, y_labels_n)
        # print(clas_loss)
        gradient_ext = ext_tape.gradient(mu*(la*disc_loss_+clas_loss), extractor.trainable_variables)
        gradient_disc = disc_tape.gradient(mu*disc_loss, discriminator.trainable_variables)
        gradient_clas = clas_tape.gradient(mu*clas_loss, classifier.trainable_variables)

        extractor_opt.apply_gradients(zip(gradient_ext, extractor.trainable_variables))
        discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))
        classifier_opt.apply_gradients(zip(gradient_clas, classifier.trainable_variables))

    return disc_loss, disc_loss_, clas_loss


def train(dataset, epochs):
    max_acc = 0.8
    acc_list_n = []
    acc_list_m = []
    acc_list_disc = []
    for epoch in range(epochs):
        p = epoch/epochs
        la = (2/(1+tf.exp(-gama*p))-1)*1
        mu = 1/((1+10*p)**0.75)
        # mu = 1
        for data in dataset:
            disc_loss, disc_loss_, clas_loss = train_step(data[0], data[1], data[2], data[3], la, mu)
        n_err, m_err, disc_err = get_test_acc(test_image, test_label, m_test_image, m_test_label)
        print(round(epoch, 1),
              'n_err: %.2f' % (1-n_err),
              'm_acc: %.2f' % (1-m_err),
              'disc_acc: %.2f' % (1-disc_err),
              'la: %.2f' % float(la),
              'mu: %.2f' % mu,
              'disc_loss %.2f:' % float(disc_loss),
              '-disc_loss: %.2f' % float(disc_loss_),
              'clas_loss: %.2f' % float(clas_loss))
        acc_list_n.append(1-n_err)
        acc_list_m.append(1-m_err)
        acc_list_disc.append(1-disc_err)
        if acc_list_m[-1] > max_acc:
            max_acc = acc_list_m[-1]
            discriminator.save('model_save\discriminator_acc_m=%.2f.h5' % acc_list_m[-1])
            extractor.save('model_save\extractor_acc_m=%.2f.h5' % acc_list_m[-1])
            classifier.save('model_save\classifier_acc_m=%.2f.h5' % acc_list_m[-1])

    plt.plot(acc_list_n)
    plt.plot(acc_list_m)
    plt.plot(acc_list_disc)
    plt.show()


def get_test_acc(n_test_image, n_test_label, m_test_image, m_test_label):
    n_err = 0
    m_err = 0
    disc_err = 0

    test_num = n_test_image.shape[0]
    # print('test_num:', test_num)
    n_test_features = extractor(n_test_image, training=False)
    m_test_features = extractor(m_test_image, training=False)
    n_test_outs = classifier(n_test_features)
    m_test_outs = classifier(m_test_features)

    n_test_disc_out = discriminator(n_test_features, training=False)
    m_test_disc_out = discriminator(m_test_features, training=False)


    # print(n_test_outs[0])
    for i in range(test_num):
        if n_test_label[i][tf.argmax(n_test_outs[i])] != 1:
            n_err += 1
        if m_test_label[i][tf.argmax(m_test_outs[i])] != 1:
            m_err += 1
        # if tf.argmax(n_test_disc_out[i]) != 1:
        #         disc_err += 1
        # if tf.argmax(m_test_disc_out[i]) != 0:
        #         disc_err += 1
        if abs(n_test_disc_out[i] - 0) < abs(n_test_disc_out[i] - 1):
            disc_err += 1
        if abs(m_test_disc_out[i] - 1) < abs(m_test_disc_out[i] - 0):
            disc_err += 1

    return n_err/test_num, m_err/test_num, disc_err/(2*test_num)


BUFFER_SIZE = 60000
BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices((train_image, m_train_image, train_label, m_train_label))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(dataset)
train(dataset, EPOCH)

