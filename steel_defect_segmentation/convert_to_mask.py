import os
import cv2
import numpy as np
import pandas as pd

data_dir = '/Users/brain/Desktop/DL/data/severstal-steel-defect-detection'

df = pd.read_csv(os.path.join(data_dir, "prep_data.csv"))


def rle_to_mask(rle):
    # CONVERT RLE TO MASK
    if (pd.isnull(rle)) | (rle == ''):
        return np.zeros((128, 800), dtype=np.uint8)

    height = 256
    width = 1600
    mask = np.zeros(width*height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]-1
    lengths = array[1::2]
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape((height, width), order='F')[::2, ::2]


for i, cols in df.iterrows():
    indices = df.index
    f = df["ImageId"][i].split(".")[0]
    y = np.empty((df.shape[0], 128, 800, 4), dtype=np.uint8)
    for j in range(4):
        y[i,:,:,j] = rle_to_mask(df['Defect_'+str(j+1)].iloc[indices[i]])
        cv2.imwrite(f'{data_dir}/train_masks/{f}_mask{j+1}.png', y[i, :, :, j])

print("process complete")
