import tensorflow as tf
import os
import time
import shutil
from WSI_Detection import gen_batch_function,model,optimize,find_IOU_score,save_model
out_data_folder='beanbag'
out_model_folder=os.path.join(out_data_folder,'model')
base_data = './dataset'
train_folder_feature = os.path.join(base_data, 'train_feature')
train_folder_label = os.path.join(base_data, 'train_label')
summary_output_dir = os.path.join(out_data_folder,"summary")
def_shape = [2000, 2000]
dept_coeff= 4                    # while running on big (GPU >12GB)pc make it 64

batch_size_train =2
iteration = 2        # max value is total images/batch size
epochs = 1

num_classes = 3
learning_rate = 0.0001
save_after=5

def train():
    get_batches_fn = gen_batch_function(train_folder_feature, train_folder_label, def_shape)
    feature = tf.placeholder(dtype=tf.float32, shape=[None, def_shape[0], def_shape[1], 3])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, def_shape[0], def_shape[1], 3])
    # loss, output_img = net_loss(feature, labels)
    out = model(feature)
    train_op, cross_entropy_loss = optimize(out, labels, learning_rate, num_classes)
    iou_val, iou_val_black, intersection, union, intersection_black, union_black=find_IOU_score(out,labels)

    saver = tf.train.Saver()
    print('Creating Session !!')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('Variables initialised !!')

    t=time.clock()
    if os.path.exists(out_model_folder):
        saver.restore(sess, os.path.join(out_model_folder, 'model.ckpt'))
        print('Model found !! Restoring it in ' + str(time.clock() - t))
    else:
        print('Model not found !! Initialising a new model ')
        print('Initialised in ' + str(time.clock() - t))
    tlast=time.clock()
    tmodel=time.clock()
    for e in range(epochs):
        count = 0
        for batch_feature, batch_label in get_batches_fn(batch_size_train):
            if batch_feature.shape[0] is not batch_size_train or batch_label.shape[0] is not batch_size_train:
                print('Bad batch found. discarding it !!   ', batch_feature.shape, batch_label.shape)
                continue
            count = count + 1
            if count > iteration:
                break
            train_out, loss, output_image,semseg_accuracy,semseg_accuracy_black,out_intersection,out_union,out_intersection_black,out_union_black = sess.run([train_op,
                                                    cross_entropy_loss, out,iou_val,iou_val_black,intersection,union,
                                                    intersection_black,union_black],
                                                     feed_dict={feature: batch_feature, labels: batch_label})
            round_time = (time.clock() - tlast)
            tlast = time.clock()

            total_time_left = int(round_time * ((epochs - e - 1) * iteration + iteration - count) / 60)
            hours = int(total_time_left / 60)
            min = total_time_left - hours * 60
            print('Epoch: '+str(e)+', Batch: '+str(count)+', Loss: '+str(loss)+', Extra time:',hours,' hrs and ',min,' minutes.')
            print('IOU Score - >   Normal cell: '+str(semseg_accuracy_black)+', Benign: '+str(semseg_accuracy[0])+
                  ', In situ: '+ str(semseg_accuracy[1])+' , Invasive: ',str(semseg_accuracy[2])+', Note: IOU Score 0 implies not applicable')

            if time.clock() - tmodel > save_after * 60:
                save_model(out_model_folder, sess, saver)
                tmodel = time.clock()
        save_model(out_model_folder,sess,saver)

    if os.path.exists(summary_output_dir):
        shutil.rmtree(summary_output_dir)
    os.makedirs(summary_output_dir)
    writer = tf.summary.FileWriter(summary_output_dir, sess.graph)
    print("Summary saved in path: %s" % summary_output_dir)

    sess.close()


if __name__ == "__main__":
    print('Execution starts ')
    train()