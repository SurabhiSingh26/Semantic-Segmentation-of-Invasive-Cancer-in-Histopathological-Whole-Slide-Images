import tensorflow as tf
import numpy as np
import time
import os
import random
import shutil
import scipy.misc
from glob import glob
import shutil
import matplotlib.pyplot as plt
from WSI_Detection import model,show_img,find_IOU_score
log_file = 'file_log.txt'
text_file = open(log_file, 'w')



def print_logs(message):
    print(message)
    text_file.write(message)
    text_file.write('\n')


out_data_folder='beanbag'
out_model_folder=os.path.join(out_data_folder,'model')
out_image_folder=os.path.join(out_data_folder,'output_image')
base_data = './dataset'
test_folder = os.path.join(base_data, 'newtest')
def_shape = [2000, 2000]
test_iteration=20
has_labels=True  # to get accuracy of prediction. Make False if true labels are not known

def test(sess=None):
    sum_iou = 0
    avg_iou = 0

    count_iou = 0
    feature = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    out = model(feature)
    iou_val, iou_val_black, intersection, union, intersection_black, union_black = find_IOU_score(out, labels)
    out = out / tf.abs(out)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    t0=time.clock()
    if os.path.exists(out_model_folder):
        saver.restore(sess, os.path.join(out_model_folder, 'model.ckpt'))
        print_logs('Model found !! Restoring it in ' + str(time.clock() - t0))
    else:
        print_logs('Model not found. Ending the process !!')
        return 0
    if os.path.exists(out_image_folder):
        shutil.rmtree(out_image_folder)
        print_logs('Output Directory deleted')
    os.makedirs(out_image_folder)
    print_logs('Output Directory created')
    # test_image_names=glob(os.path.join(test_folder,'*jpg'))
    test_image_names=os.listdir(test_folder)
    random.shuffle(test_image_names)
    count=0
    for image_name in test_image_names:
        if count>=test_iteration:
            break

        img=scipy.misc.imread(os.path.join(test_folder,image_name))
        img=scipy.misc.imresize(img,def_shape)
        img=np.expand_dims(img, axis=0)

        if len(img.shape) is not 4 :
            print_logs('Bad test image spotted !! '+str(img.shape))
            continue
        if img.shape[3] > 3:
            img=img[:,:,:,0:3]
        img = np.array(img) / 127.5 - 1
        if has_labels is True:
            actualgt=scipy.misc.imread(os.path.join('dataset','train_label',image_name))
            actualgt = np.array(np.expand_dims(actualgt, axis=0)) / 127.5 - 1
            show_img(os.path.join(out_image_folder, image_name[0:-4] + '_gt.jpg'), actualgt[ 0,:, :, :])
            output_image,iou_score,iou_score_black = sess.run([out,iou_val,iou_val_black],
                                                              feed_dict={feature: img,labels:actualgt})
            print_logs('Testing done for '+ str(image_name)+'  Accuracy for invasive is : '+str(iou_score[2])+
                       ' and for normal is: '+ str(iou_score_black))
            if iou_score[2]>0.1:
                sum_iou=sum_iou+iou_score[2]
                count_iou=count_iou+1
                avg_iou=sum_iou/count_iou
                print_logs('Average IOU score for '+ str(count_iou)+' images is '+ str(avg_iou))

        else:
            output_image= sess.run(out, feed_dict={feature: img})
            print_logs('testing done for '+str( image_name))
        show_img(os.path.join(out_image_folder,image_name[0:-4]+'_main.jpg' ),img[0, :, :, :])
        show_img(os.path.join(out_image_folder,image_name[0:-4]+'_output.jpg' ),output_image[0, :, :, :])
        count+=1
    print_logs('Testing done !!')


if __name__ == "__main__":
    print_logs('Testing starts!')
    print_logs('Note: Zero values for IOU score means, it isnt applicable in that case')
    test()