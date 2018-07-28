
import sys
import os
import openslide
import scipy.misc
from matplotlib import pyplot as plt
from scipy.misc import imsave, imresize
from openslide import open_slide # http://openslide.org/api/python/
import numpy as np
import os
import shutil
import time
count=0
save = False

dir_img = './work/'
out_image='./processed'
save_image_dir='./dataset/train_feature'
if os.path.exists(save_image_dir):
    print('Outputimage directory exists')
    shutil.rmtree(save_image_dir)
else:
    print('Creating image output directory')
    os.mkdir(save_image_dir)

# if permission denied error comes then re run the code
valid_images = ['.svs']

patch_size = (2000,2000)

if os.path.exists(out_image):
    print('Output directory exists')
else:
    print('Creating output directory')
    os.mkdir(out_image)
filename=os.listdir(dir_img)
big_images=0
for f in filename:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    curr_path = os.path.join(dir_img,f)
    print(curr_path)

    # open scan
    scan = openslide.OpenSlide(curr_path)

    print(scan)
    orig_w = np.int(scan.properties.get('aperio.OriginalWidth'))
    orig_h = np.int(scan.properties.get('aperio.OriginalHeight'))
    print(orig_h,orig_w)
    # create an array to store our image
    #img_np = np.zeros((orig_w,orig_h,3),dtype=np.uint8)
    t=time.clock()
    tlast=t
    for r in range(0,orig_w,patch_size[0]):
        for c in range(0, orig_h,patch_size[1]):

            if c+patch_size[1] > orig_h and r+patch_size[0]<= orig_w:
                p = orig_h-c
                img = np.array(scan.read_region((c,r),0,(p,patch_size[1])),dtype=np.uint8)[...,0:3]
            elif c+patch_size[1] <= orig_h and r+patch_size[0] > orig_w:
                p = orig_w-r
                img = np.array(scan.read_region((c,r),0,(patch_size[0],p)),dtype=np.uint8)[...,0:3]
            elif  c+patch_size[1] > orig_h and r+patch_size[0] > orig_w:
                p = orig_h-c
                pp = orig_w-r
                img = np.array(scan.read_region((c,r),0,(p,pp)),dtype=np.uint8)[...,0:3]
            else:    
                img = np.array(scan.read_region((c,r),0,(patch_size[0],patch_size[1])),dtype=np.uint8)[...,0:3]
                local=0
                scipy.misc.imsave(os.path.join(save_image_dir,str(count)+'_'+str(local)+'.jpg'),img)
                print('Saved ' + str(count)+'_'+str(local) + ' images ' + str(big_images) + 'th image on processing ')
                img = np.rot90(img, 1)
                local+=1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), img)
                print('Saved ' + str(count) +'_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                img = np.rot90(img, 1)
                local+=1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), img)
                print('Saved ' + str(count) +'_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                img = np.rot90(img, 1)
                local+=1
                scipy.misc.imsave(os.path.join(save_image_dir, str(count) + '_' + str(local) + '.jpg'), img)
                print('Saved ' + str(count) +'_' + str(local) + ' images ' + str(big_images) + 'th image on processing')
                count+=1
            #print(img.shape)
            timet = time.clock() - tlast
            #print('Expected time required ',timet)
            tlast=time.clock()

            #img_np[r:r+patch_size[0],c:c+patch_size[1]] = img
    big_images=big_images+1
    name_no_ext = os.path.splitext(f)[0]
    save_path = os.path.join(out_image, name_no_ext)
    if os.path.exists(save_path):
        os.remove(save_path)
        print('Previous array  deleted')

