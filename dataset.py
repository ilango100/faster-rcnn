#!/usr/bin/env python

from tqdm import tqdm
import os
import math
import requests
import tarfile
import h5py
import numpy as np
import pandas as pd
import cv2 as cv


splits = ["train", "test"]
baseurl = "http://ufldl.stanford.edu/housenumbers/"
chunk_size = 1048576  # 1 MB

for file in splits:
    fname = file+".tar.gz"
    if os.path.exists(fname):
        print(fname, "already exists")
        continue
    fd = open(fname, "wb")
    resp = requests.get(baseurl+fname, stream=True)
    if resp.status_code != 200:
        raise RuntimeError("Error downloading dataset: " + resp.status_code)
    l = int(resp.headers.get("Content-Length"))
    l = math.ceil(l/chunk_size)
    for chunk in tqdm(resp.iter_content(chunk_size=chunk_size), desc=fname, total=l):
        fd.write(chunk)
    fd.close()

for file in splits:
    if os.path.exists(file):
        print(file, "already exists. No need of extraction")
        continue
    fname = file+".tar.gz"
    fd = tarfile.open(fname, "r")
    for memb in tqdm(fd.getmembers(), desc=fname + " extraction"):
        fd.extract(memb)
    fd.close()

for file in splits:
    if os.path.exists(file+".csv"):
        print(file+".csv", "already exists. No need of conversion.")
        continue
    digit = h5py.File(file+"/digitStruct.mat", "r")
    tot = len(digit.get("digitStruct/name"))

    df = pd.DataFrame(columns=["name", "label", "top",
                               "left", "height", "width", "imgheight", "imgwidth"])

    i = 0
    for k, (n, b) in tqdm(enumerate(zip(digit.get("digitStruct/name"), digit.get("digitStruct/bbox"))), desc=file+" conversion", total=tot):
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

    df["x"] = df.left+df.width/2
    df["y"] = df.top+df.height/2
    df.to_csv(file+".csv", index=False)
