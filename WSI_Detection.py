import tensorflow as tf
import os
import numpy as np
import random
import scipy.misc
from glob import glob
import time
import shutil
import matplotlib.pyplot as plt
out_data_folder='beanbag'
out_model_folder=os.path.join(out_data_folder,'model')
out_image_folder=os.path.join(out_data_folder,'output_image')
base_data = './dataset'
train_folder_feature = os.path.join(base_data, 'train_feature')
train_folder_label = os.path.join(base_data, 'train_label')
test_folder = os.path.join(base_data, 'test_feature')
summary_output_dir = "./summary"
def_shape = [2000, 2000]
dept_coeff= 4                    # while running on pc make it 64


batch_size_train =2
iteration = 2        # max value is total images/batch size
epochs = 1
test_iteration=100


num_classes = 3
learning_rate = 0.0001
save_after=5


def gen_batch_function(feature_folder, label_folder, image_shape):
    def get_batches_fn(batch_size):
        image_paths = os.listdir(feature_folder)
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            train_images = []
            train_label = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                img = scipy.misc.imresize(scipy.misc.imread(os.path.join(feature_folder, image_file)), image_shape)
                label_file_name=image_file
                # label_file_name = image_file[3:len(image_file)]
                # image_file = 'fence' + label_file_name
                label_img = scipy.misc.imresize(scipy.misc.imread(os.path.join(label_folder, image_file)), image_shape)
                if len(label_img.shape) is not 3 or len(img.shape) is not 3:
                    print(label_img.shape, img.shape, ' --- Gray scale image found ', label_file_name)
                    continue
                if label_img.shape[2] > 3 or img.shape[2] > 3:
                    print(label_img.shape, img.shape, ' --- Bad shape image found ,', label_file_name)
                    continue
                label_img = label_img[:, :, 0:4]
                img = img[:, :, 0:4]

                img = (img / 127.5) - 1  # making the
                label_img = label_img / 127.5 - 1
                train_images.append(img)
                train_label.append(label_img)
            yield np.array(train_images), np.array(train_label)

    return get_batches_fn


def conv(name, x, filter_size, in_filters, out_filters, strides, atrous=1, bias=False):
    """Convolution."""
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        w = tf.get_variable('weights', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        if atrous == 1:
            conv = tf.nn.conv2d(x, w, strides, padding='SAME')
        else:
            assert (strides == stride_arr(1))
            conv = tf.nn.atrous_conv2d(x, w, rate=atrous, padding='SAME')
        if bias:
            b = tf.get_variable('biases', [out_filters], initializer=tf.constant_initializer())
            return conv + b
        else:
            return conv


def stride_arr(stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]


def batch_norm(name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y


def elu(x):
    elu = tf.nn.elu(x)
    return elu


def res_func(x, in_filter, out_filter, stride, atrous=1):
    """Bottleneck residual unit with 3 sub layers."""

    orig_x = x

    with tf.variable_scope('block_1'):
        x = conv('conv', x, 1, in_filter, out_filter / 4, stride, atrous)
        x = batch_norm('bn', x)
        x = elu(x)

    with tf.variable_scope('block_2'):
        x = conv('conv', x, 3, out_filter / 4, out_filter / 4, stride_arr(1), atrous)
        x = batch_norm('bn', x)
        x = elu(x)

    with tf.variable_scope('block_3'):
        x = conv('conv', x, 1, out_filter / 4, out_filter, stride_arr(1), atrous)
        x = batch_norm('bn', x)

    with tf.variable_scope('block_add'):
        if in_filter != out_filter:
            orig_x = conv('conv', orig_x, 1, in_filter, out_filter, stride, atrous)
            orig_x = batch_norm('bn', orig_x)
        x += orig_x
        x = elu(x)
    return x


def model(feature):
    # feature size is [batch_size,2000,2000,3]
    H = def_shape[0]     # H = 2000
    W = def_shape[1]     # W = 2000
    filters = [dept_coeff,dept_coeff*4, dept_coeff*8, dept_coeff*16, dept_coeff*32]
    #filters = [64, 256, 512, 1024, 2048]
    with tf.variable_scope('model_main', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('group_1'):
            x = conv('conv1', feature, 7, 3, filters[0], stride_arr(2))  # x[ batch size,1000, 1000, 64]
            x = batch_norm('bn_conv1', x)
            x = elu(x)
            x_div_2 = x
            x_div_2 = tf.layers.conv2d(x_div_2, num_classes, 4, strides=1, padding='SAME', name='skip_conv1')
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')  # [batch size, 500,500,64]
        with tf.variable_scope('group_2_0'):
            x = res_func(x, filters[0], filters[1], stride_arr(1))  # [batch size, 500,500,256]
        for i in range(1, 3):
            with tf.variable_scope('group_2_%d' % i):
                x = res_func(x, filters[1], filters[1], stride_arr(1))  # [batch size, 500,500,256]
        x_div_4 = x
        x_div_4 = tf.layers.conv2d(x_div_4, num_classes, 4, strides=1, padding='SAME', name='skip_conv2')
        # size of x_div_4 is [batch size, 500, 500, 3]
        with tf.variable_scope('group_3_0'):
            x = res_func(x, filters[1], filters[2], stride_arr(2))  # [batch size, 250,250,512]
        for i in range(1, 3):
            with tf.variable_scope('group_3_%d' % i):
                x = res_func(x, filters[2], filters[2], stride_arr(1))  # [batch size, 250,250,512]
        with tf.variable_scope('group_4_0'):
            x = res_func(x, filters[2], filters[3], stride_arr(1), 2)  # [batch size, 250,250,1024]
        for i in range(1, 3):
            with tf.variable_scope('group_4_%d' % i):
                x = res_func(x, filters[3], filters[3], stride_arr(1), 2)  # [batch size, 250,250,1024]

        with tf.variable_scope('group_5_0'):
            x = res_func(x, filters[3], filters[4], stride_arr(1), 4)  # [batch size, 250,250,2048]
        for i in range(1, 3):
            with tf.variable_scope('group_5_%d' % i):
                x = res_func(x, filters[4], filters[4], stride_arr(1), 4)  # [batch size, 250,250,2048]

        with tf.variable_scope('group_last'):
            x = elu(x)

        with tf.variable_scope('fc1_voc12'):
            x0 = tf.layers.conv2d(x, num_classes, 3, strides=1, padding='SAME', dilation_rate=6, name='conv0')
            x1 = tf.layers.conv2d(x, num_classes, 3, strides=1, padding='SAME', dilation_rate=12, name='conv1')
            x2 = tf.layers.conv2d(x, num_classes, 3, strides=1, padding='SAME', dilation_rate=18, name='conv2')
            x3 = tf.layers.conv2d(x, num_classes, 3, strides=1, padding='SAME', dilation_rate=24, name='conv3')

            x = tf.add(x0, x1)
            x = tf.add(x, x2)
            x = tf.add(x, x3)
            logits = x
            x_flat = tf.reshape(x, [-1, num_classes])
            pred = tf.nn.softmax(x_flat)
            pred = tf.reshape(pred, tf.shape(x))  # pred shape is [batc_size,def_shape/8,def_hape/8,3]

            out = tf.layers.conv2d_transpose(pred, filters=3,
                                             kernel_size=4, strides=(2, 2), padding='SAME', name="deconv1")
            # here size will be [batch_Size,def_shape/4,def_shape/4,3]
            out = tf.add(x_div_4, out)
            out = tf.layers.conv2d_transpose(out, filters=3,
                                             kernel_size=4, strides=(2, 2), padding='SAME', name="deconv2")
            # here size will be [batch_Size,def_shape/2,def_shape/2,3]
            out = tf.add(x_div_2, out)
            out = tf.layers.conv2d_transpose(out, filters=3,
                                             kernel_size=4, strides=(2, 2), padding='SAME', name="deconv3")
            # here size will be [batch_Size,def_shape,def_shape,3]

            up = tf.image.resize_bilinear(pred, [H, W])
            out=tf.clip_by_value(out,-1,1)
            return out


def show_img(name,img):
    img = (img[:, :, 0:3] + 1) * 127.5
    img = img.astype(np.uint8)
    scipy.misc.imsave(name,img)
    # plt.imshow(img)
    # plt.show()


def save_model(folder, sess, saver):
    print('Saving model!!')
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(str(folder) + ' deleted')
    os.makedirs(folder)
    print(str(folder) + ' Directory created!! Saving the model. Wait')
    save_path = saver.save(sess, os.path.join(folder, 'model.ckpt'))
    print('Model saved in path: ' + str(save_path))


def optimize(nn_last_layer, correct_label, learning_rate=0.001, num_classes=3):

    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    correct_label_reshaped = tf.reshape(correct_label, [-1,num_classes])

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=correct_label_reshaped)
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")
    # loss_op = tf.reduce_mean(tf.abs(nn_last_layer - correct_label), name="fcn_loss")
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return train_op, loss_op


def find_IOU_score(out,labels):
    out = out / tf.abs(out)
    new_label = tf.clip_by_value(labels, 0, 1)
    new_out = tf.clip_by_value(out, 0, 1)

    intersection = tf.reduce_sum(new_label * new_out, [1, 2])
    union = tf.reduce_sum(tf.clip_by_value(new_label + new_out, 0, 1), [1, 2])
    iou_val = intersection / (union+1)
    iou_val = tf.reduce_mean(iou_val, 0)

    new_label_black = tf.clip_by_value(tf.reduce_sum(new_label, -1), 0, 1)
    new_out_black = tf.clip_by_value(tf.reduce_sum(new_out, -1), 0, 1)
    new_label_black = 1 - new_label_black
    new_out_black = 1 - new_out_black

    intersection_black = tf.reduce_sum(new_label_black * new_out_black, [1, 2])
    union_black = tf.reduce_sum(tf.clip_by_value(new_label_black + new_out_black, 0, 1), [1, 2])
    iou_val_black = intersection_black / (union_black+1)
    iou_val_black = tf.reduce_mean(iou_val_black, 0)

    return iou_val,iou_val_black,intersection,union,intersection_black,union_black



