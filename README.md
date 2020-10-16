# DANN
本例为对 Unsupervised Domain Adaptation by Backpropagation论文的其中mnist-mnist_m部分实现。<br>
论文连接为：https://arxiv.org/abs/1409.7495<br>

## 问题描述
为了无监督的（Unsupervised）解决mnist-m`带背景的手写数字数据集`分类任务，论文采用**迁移学习**的方法，把使用普通手写数字数据集**mnist**训练的模型进行迁移到对mnist-m数据集分类任务中，单纯的训练完成之后迁移的方法识别准确率不高，论文结合对抗**adversarial**的思想，使用一个统一的模型进行迁移训练，其中原域*source domain*为原本的数据集**mnist**，目标域**target domain**为任务要进行分类的相关数据集**mnist_m**。
## 模型结构
对于原域的样本分类结构，论文中将神经网络识别数字的过程看作两部分，第一部分为从样本输入到某一个特定的中间层，表示对图片样本进行**特征提取**的过程，第一部分的输出为提取出的特征；第二部分为从第一部分的输出到最后的**分类**结果，表示根据图片样本抽取出的特征进行分类的过程。<br>
对于目标域的分类，则需要借助原域的训练的分类器，迁移学习中原域与目标域的差别不大（有在某些层面上的相似性），如同mnist和mnist_m中都是数字的识别任务。论文将目标域的样本与原域的样本分别输入原域模型的第一部分：**特征提取器extractor**，进行特征提取，可以获得原域样本提取出的特征**n_features**与目标域样本提取出的特征**m_features**，这些特征都将送入原域模型的第二部分：**分类器classifier**中进行分类得到对应的样本的预测，对于原域的带标签的样本，分类器可以直接得出损失值，并反向传播实现，识别准确率的提升，而对于目标域的不带label的样本，则需要依赖原域的分类器，原域的分类器对目标域样本的分类能力越强，域适应也就越完全。域适应的过程是通过对抗的方式实现的：<br>
在原始模型的基础上，将**特征提取器extractor**的输出接入到新引进的**域分类器discriminator**中，由域分类器来进行判断，输入的特征原本是来源于原域还是目标域的，域分类器的效果（准确率）越好说明了原域于目标域的样本中提取到的特征具有的**共性**较小，若经过特征提取之后的原域样本于目标域样本的分布基本重合则域分类器的准确率应接近0.5，分类效果差。<br>
通过对抗地训练特征提取器和域分类器，可以使得特征提取器可以从无论原域还是目标域样本中提取出具有**共性**的特征。

程序如下所示：
``` python
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


```
