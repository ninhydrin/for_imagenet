import numpy as np
import matplotlib.pyplot as plt
import pickle
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
caffe.set_device(0)
caffe.set_mode_gpu()
import os
net = caffe.Net('deploy.prototxt',
                '500B_50epoch.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load('mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(50,3,227,227)
#net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('/media/yokoshima/335ec2a2-1d0c-4fcf-b96d-f6da5a3d106a/imagenet/train/cat.jpg'))
hist=np.zeros([500,1000])
imagenet_labels_filename ="synset_words.txt"
file_list=open("files.txt")
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
except:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
abc=0
for line in file_list:
    filepath= "../imagenet/train/"+line.split()[0] 
    num=int( line.split()[1])
    if(num!=abc):
        os.remove("hist")
        pickle.dump(hist,open("hist","w"),-1)
        top = hist[abc].argmax()
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.ylim([0,1300])
        plt.text(800, 800, 'Class {0} = {1} is Max'.format(top,hist[abc][top]) , ha = 'center', va = 'top')
        plt.plot( hist[abc])

        fname="Class"+str(abc)+".png"
        plt.savefig(fname)
    print filepath,num
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(filepath))
    out = net.forward()
    hist[num]+=net.blobs["prob"].data[0].flatten()    
    abc=num

    

    #top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-10:-1]
    #hist[net.blobs['prob'].data[0].flatten().argsort().argmax()]=1
    #plt.text(top_k[-1], net.blobs['prob'].data[0].flatten()[top_k[0]], '{0} is Max'.format(top_k[0]) , ha = 'center', va = 'bottom')
pickle.dump(hist,open("histlast","w"),-1)
#print labels[top_k]

