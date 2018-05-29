#/usr/bin/env python
# Copyright (c) 2018, Ruohan Gao
# All rights reserved.

import os
import argparse
import random
import numpy as np
from scipy import misc
import math
import skimage.measure
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import PIL
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flowImgInputDir', type=str, required=True)
    parser.add_argument('--rgbImgDir', type=str, required=True)
    parser.add_argument('--blockSize', type=int, default=8)
    parser.add_argument('--arrowImgOutDir', type=str)
    args = parser.parse_args()
    flowImgs = sorted(os.listdir(args.flowImgInputDir))
    if not os.path.isdir(args.arrowImgOutDir):
        os.mkdir(args.arrowImgOutDir)

    #set plots
    fig,ax = plt.subplots(1,1) 
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    nz = mcolors.Normalize(0.0, 2*np.pi)

    for img_index,img_name in enumerate(flowImgs):
    	img_name = img_name.strip()
        print 'Visualizing flow for ', img_name
    	predicted_flow = misc.imread(os.path.join(args.flowImgInputDir, img_name))
        predicted_xy = np.empty([predicted_flow.shape[0], predicted_flow.shape[1], 2])

        #flow are normalized and encoded in an image
        #convert back to original scale sin,cos,mgt
        predicted_mgt = predicted_flow[:,:,2] / 255.0 * 30
        predicted_sin = predicted_flow[:,:,0] / 255.0 * 2 - 1
        predicted_cos = predicted_flow[:,:,1] / 255.0 * 2 - 1

        #convert back to flow
        predicted_xy[:,:,0] = predicted_cos * predicted_mgt
        #minus sign because y-axis is reversed for displaying
        predicted_xy[:,:,1] = -predicted_sin * predicted_mgt

        #block-wise average flow prediction
        downsampled_predicted_xy = skimage.measure.block_reduce(predicted_xy, (args.blockSize,args.blockSize,1), np.mean)
        downsampled_mgt = np.sqrt(np.square(downsampled_predicted_xy[:,:,0]) + np.square(downsampled_predicted_xy[:,:,1]))

        im = Image.open(os.path.join(args.rgbImgDir, img_name))
        im = im.point(lambda p: p)
        im.save(os.path.join(args.arrowImgOutDir, img_name))
        input_image = misc.imresize(misc.imread(os.path.join(args.arrowImgOutDir, img_name)), predicted_flow.shape)

        #find a suitable threshold for visulization
        flatten_mgt = downsampled_mgt.flatten()
        largeNIndex = heapq.nlargest(60, range(len(flatten_mgt)), flatten_mgt.take)
        threshold = flatten_mgt[largeNIndex[-1]]
        #plot arrows
        input_image = input_image
        ax.imshow(input_image)
        # quiver statistics
        X = []
        Y = []
        U = []
        V = []
        for i in range(input_image.shape[0]/args.blockSize):
        	for j in range(input_image.shape[1]/args.blockSize):
        		if downsampled_mgt[i,j] >= threshold:
				Y.append(i * args.blockSize + args.blockSize / 2)
				X.append(j * args.blockSize + args.blockSize / 2)
				U.append(downsampled_predicted_xy[i,j,0])
				V.append(downsampled_predicted_xy[i,j,1])

	#negate to make the color agree with the color wheel
	angle = np.arctan2(np.negative(V),np.negative(U)) + np.pi
	ax.quiver(X,Y,U,V, angles='uv', units='xy', color=cm.hsv(nz(angle)), pivot='mid')
	plt.savefig(os.path.join(args.arrowImgOutDir, img_name), bbox_inches='tight', pad_inches=0)
	plt.cla()
    plt.close()

if __name__ == "__main__":
    main()
