#!/usr/bin/env python

import h5py
import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm


splits = ["train", "test"]

for path in splits:
    digit = h5py.File(path+"/digitStruct.mat", "r")
    tot = len(digit.get("digitStruct/name"))

    df = pd.DataFrame(columns=["name", "label", "top",
                               "left", "height", "width", "imgheight", "imgwidth"])

    i = 0
    print(path, ":")
    for k, (n, b) in tqdm(enumerate(zip(digit.get("digitStruct/name"), digit.get("digitStruct/bbox"))), total=tot):
        nm = np.array(digit[n[0]]).reshape(-1)
        name = "".join(map(chr, nm))
        imgheight, imgwidth = cv.imread("train/"+name).shape[:2]
        box = digit[b[0]]
        for j, lab in enumerate(box["label"]):
            try:
                label = int(lab[0])
                top = int(box["top"][j, 0])
                left = int(box["left"][j, 0])
                height = int(box["height"][j, 0])
                width = int(box["width"][j, 0])
            except:
                label = int(digit[lab[0]][0, 0])
                top = int(digit[box["top"][j, 0]][0, 0])
                left = int(digit[box["left"][j, 0]][0, 0])
                height = int(digit[box["height"][j, 0]][0, 0])
                width = int(digit[box["width"][j, 0]][0, 0])
            if label == 10:
                label = 0
            df.loc[i] = {"name": name,
                         "label": label,
                         "top": top,
                         "left": left,
                         "height": height,
                         "width": width,
                         "imgheight": imgheight,
                         "imgwidth": imgwidth}
            i += 1

    df.to_csv(path+".csv", index=False)
