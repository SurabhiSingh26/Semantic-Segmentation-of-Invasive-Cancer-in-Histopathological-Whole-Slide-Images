
import numpy as np
import os
import shutil
import scipy.misc

def save_patches(img,save_image_dir,count,big_images):


    patch_size=[2000,2000]
    [orig_w,orig_h,channel]=img.shape
    resize_shape_For_10=[58401,40001]
    for r in range(0, orig_w, patch_size[0]):
        for c in range(0, orig_h, patch_size[1]):

            if c + patch_size[1] > orig_h and r + patch_size[0] <= orig_w:
                p = orig_h - c
            elif c + patch_size[1] <= orig_h and r + patch_size[0] > orig_w:
                p = orig_w - r
            elif c + patch_size[1] > orig_h and r + patch_size[0] > orig_w:
                p = orig_h - c
                pp = orig_w - r
            else:
                imgtemp = np.array(img[r:r+patch_size[0],c:c+patch_size[1],:], dtype=np.uint8)
                local = 0
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), imgtemp)
                print('Saved ' + str(count) + '_' + str(local) + ' images ' + str(big_images) + 'th image on processing ')
                imgtemp = np.rot90(imgtemp, 1)
                local += 1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), imgtemp)
                print('Saved ' + str(count) + '_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                imgtemp = np.rot90(imgtemp, 1)
                local += 1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), imgtemp)
                print('Saved ' + str(count) + '_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                imgtemp = np.rot90(imgtemp, 1)
                local += 1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), imgtemp)
                print('Saved ' + str(count) + '_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                count += 1
    return count