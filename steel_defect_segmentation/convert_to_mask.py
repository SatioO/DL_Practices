import os
import cv2
import numpy as np
import pandas as pd

DATA_DIR = '/Users/brain/Desktop/DL/data/severstal-steel-defect-detection'

df = pd.read_csv(os.path.join(DATA_DIR, "prep_data.csv"))


def rle_to_mask(rle, img_height=256, img_width=1600):
    if (pd.isnull(rle)) | (rle == ''):
        return np.zeros((img_height, img_width), dtype=np.uint8)

    mask = np.zeros((img_height * img_width), dtype=np.uint8)
    arr = np.array(list(map(int, rle.split(' '))))

    starts = arr[0::2]
    lengths = arr[1::2]

    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1

    return mask.reshape((img_height, img_width))


for i, cols in df.iterrows():
    img_height = 256
    img_width = 1600
    f = df["ImageId"][i].split(".")[0]
    y = np.empty((df.shape[0], img_height, img_width, 4), dtype=np.uint8)
    for j in range(4):
        y[i, :, :, j] = rle_to_mask(
            cols["Defect_" + str(j + 1)], img_height, img_width)
        cv2.imwrite(f'{DATA_DIR}/train_masks/{f}_mask{j+1}.png', y[i, :, :, j])

print("process complete")
