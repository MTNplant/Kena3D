from pyhdf.SD import SD, SDC
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skimage import transform
import cv2
import h5py
import sys
from datetime import datetime

print(datetime.now())

args = sys.argv
folder = args[1]
mmvoxel = args[2]
mmvoxel = mmvoxel[0]+"."+mmvoxel[1:]
mmvoxel = float(mmvoxel)

files = sorted([f for f in os.listdir(folder) if f.endswith(".hdf")])

for n in range(len(files)):
    print(f"Now processing {files[n]} ({n}/{len(files)})")
    hdf = SD(folder+"/"+files[n],SDC.READ)
    data = hdf.select("Not specified")
    data_3d = data.get()
    width = data_3d.shape[0]
    height = data_3d.shape[1]
    depth = data_3d.shape[2]

    mag = mmvoxel/0.0151
    new_width = int(width*mag)
    new_height = int(height*mag)
    new_depth = int(depth*mag)

    img_3d = np.zeros([width,new_height,new_depth])

    for w in range(width):
        img = cv2.resize(data_3d[w,:,:],dsize=(new_depth,new_height))
        img_3d[w,:,:] = img

    del data_3d
    print(f"End first processing: {img_3d.shape}")

    img_3d2 = np.zeros([new_width,new_height,new_depth])

    for h in range(new_height):
        img = cv2.resize(img_3d[:,h,:],dsize=(new_depth,new_width))
        img_3d2[:,h,:] = img

    del img_3d
    print(f"End second processing: {img_3d2.shape}")
        
    data_3d = img_3d2

    del img_3d2

    width = new_width
    height = new_height
    depth = new_depth

    plt.figure(figsize=[5,10])

    plt.subplot(3,1,2)
    plt.imshow(data_3d[:,:,data_3d.shape[2]//2])

    plt.subplot(3,1,1)
    plt.imshow(data_3d[data_3d.shape[0]//2,:,:])

    plt.subplot(3,1,3)
    plt.imshow(data_3d[:,data_3d.shape[1]//2,:])
    plt.savefig(f"{folder}/{files[n][:-4]}_raw.png",dpi=120,bbox_inches="tight")
    plt.close()
    
    min_val = data_3d[:,:,:].min()
    
    ywhere = np.where(data_3d[data_3d.shape[0]//2,:,:data_3d.shape[2]//2]>min_val)
    yw_x,yw_y = ywhere
    yw_x_min,yw_y_min = yw_x[yw_y==yw_y.min()][0],yw_y[yw_y==yw_y.min()][0]

    xwhere = np.where(data_3d[data_3d.shape[0]//2,:data_3d.shape[1]//2,:]>min_val)
    xw_x,xw_y = xwhere
    xw_x_min,xw_y_min = xw_x[xw_x==xw_x.min()][0],xw_y[xw_x==xw_x.min()][0]

    if (yw_y_min-xw_y_min)!=0:
        rot = np.arctan((yw_x_min-xw_x_min)/(yw_y_min-xw_y_min))

        angle1 = np.rad2deg(rot)
        if angle1<-45:
            angle1 = 90 + angle1
            rot = np.deg2rad(angle1)
        else:
            pass

        print(f"angle1: {angle1}")
        x=data_3d.shape[1]
        y=data_3d.shape[2]
        
        thres = (data_3d[data_3d.shape[0]//2,:,:]>min_val).astype(int)
        cx,cy = np.mean(np.sum(thres,axis=0)),np.mean(np.sum(thres,axis=0))
        #print(cx,cy)
        center = (cx, cy)
        #trans = cv2.getRotationMatrix2D(center, angle1, 1.0)
        
        img_3d = []
        for i in range(data_3d.shape[0]):
            img2 = transform.rotate(data_3d[i,:,:],angle1,resize=True)
            #img2 = cv2.warpAffine(data_3d[i,:,:], trans, (x,y))
            img_3d.append(img2)
        img_3d = np.array(img_3d)
        img_3d[img_3d<min_val] = min_val
    else:
        img_3d = data_3d
    del data_3d 

    ywhere = np.where(img_3d[:img_3d.shape[0]//2,img_3d.shape[1]//2,:]>min_val)
    yw_x,yw_y = ywhere
    yw_x_min,yw_y_min = yw_x[yw_y==yw_y.min()][0],yw_y[yw_y==yw_y.min()][0]

    xwhere = np.where(img_3d[:,img_3d.shape[1]//2,:img_3d.shape[2]//2]>min_val)
    xw_x,xw_y = xwhere
    xw_x_min,xw_y_min = xw_x[xw_x==xw_x.min()][0],xw_y[xw_x==xw_x.min()][0]

    if (yw_y_min-xw_y_min)!=0:
        rot = np.arctan((yw_x_min-xw_x_min)/(yw_y_min-xw_y_min))

        angle2 = np.rad2deg(rot)
        if angle2<-45:
            angle2 = 90+angle2
            rot = np.deg2rad(angle2)
        else:
            pass
        
        print(f"angle2: {angle2}")

        x=img_3d.shape[2]
        y=img_3d.shape[0]
        
        thres = (img_3d[:,img_3d.shape[1]//2,:]>min_val).astype(int)
        cx,cy = np.mean(np.sum(thres,axis=0)),np.mean(np.sum(thres,axis=0))
        #print(cx,cy)
        center = (cx, cy)
        #trans = cv2.getRotationMatrix2D(center, angle2, 1.0)
        
        img_3d2 = []
        for i in range(img_3d.shape[1]):
            #img2 = cv2.warpAffine(img_3d[:,i,:], trans, (x,y))
            img2 = transform.rotate(img_3d[:,i,:],angle2,resize=True)
            img_3d2.append(img2)
        img_3d2 = np.array(img_3d2).transpose((1,0,2))
        img_3d2[img_3d2<min_val] = min_val
    else:
        img_3d2 = img_3d
    del img_3d

    ywhere = np.where(img_3d2[:,:img_3d2.shape[1]//2,img_3d2.shape[2]//2]>min_val)
    yw_x,yw_y = ywhere
    yw_x_min,yw_y_min = yw_x[yw_y==yw_y.min()][0],yw_y[yw_y==yw_y.min()][0]

    xwhere = np.where(img_3d2[:img_3d2.shape[0]//2,:,img_3d2.shape[2]//2]>min_val)
    xw_x,xw_y = xwhere
    xw_x_min,xw_y_min = xw_x[xw_x==xw_x.min()][0],xw_y[xw_x==xw_x.min()][0]

    if (yw_y_min-xw_y_min)!=0:
        rot = np.arctan((yw_x_min-xw_x_min)/(yw_y_min-xw_y_min))

        angle3 = np.rad2deg(rot)
        
        if angle3<-45:
            angle3 = 90+angle3
            rot = np.deg2rad(angle3)
        else:
            pass

        print(f"angle3: {angle3}")
        
        x=img_3d2.shape[1]
        y=img_3d2.shape[0]
        
        thres = (img_3d2[:,:,img_3d2.shape[2]//2]>min_val).astype(int)
        cx,cy = np.mean(np.sum(thres,axis=0)),np.mean(np.sum(thres,axis=0))
        #print(cx,cy)
        center = (cx, cy)
        
        #trans = cv2.getRotationMatrix2D(center, angle3, 1.0)
        
        img_3d3 = []
        for i in range(img_3d2.shape[2]):
            #img2 = cv2.warpAffine(img_3d2[:,:,i], trans, (x,y))
            img2 = transform.rotate(img_3d2[:,:,i],angle3,resize=True)
            img_3d3.append(img2)
        img_3d3 = np.array(img_3d3).transpose((1,2,0))
        img_3d3[img_3d3<min_val] = min_val
    else:
        img_3d3 = img_3d2
    del img_3d2
        
    x_range = img_3d3[:,:,img_3d3.shape[2]//2].max(axis=0)
    x_range = np.where(x_range!=min_val)
    y_range = img_3d3[img_3d3.shape[0]//2,:,:].max(axis=0)
    y_range = np.where(y_range!=min_val)
    z_range = img_3d3[:,img_3d3.shape[1]//2,:].max(axis=1)
    z_range = np.where(z_range!=min_val)
    
    x_min,x_max = x_range[0][0],x_range[0][-1]+1
    y_min,y_max = y_range[0][0],y_range[0][-1]+1
    z_min,z_max = z_range[0][0],z_range[0][-1]+1
    
    print(x_min,x_max,y_min,y_max,z_min,z_max)
    
    img_3d3 = img_3d3[z_min:z_max,x_min:x_max,y_min:y_max]
    print(img_3d3.shape)

    plt.figure(figsize=[5,10])
    
    plt.subplot(3,1,2)
    plt.imshow(img_3d3[:,:,img_3d3.shape[2]//2])

    plt.subplot(3,1,1)
    plt.imshow(img_3d3[img_3d3.shape[0]//2,:,:])

    plt.subplot(3,1,3)
    plt.imshow(img_3d3[:,img_3d3.shape[1]//2,:])
    plt.savefig(f"{folder}/{files[n][:-4]}_processed.png",dpi=120,bbox_inches="tight")
    plt.close()

    with h5py.File(f'{folder}/{files[n][:-4]}_00151.h5', 'w') as f:
        f.create_dataset('img_3d', data=img_3d3)
