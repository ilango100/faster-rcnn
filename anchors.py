import warnings
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from copy import deepcopy


class Anchor:

    showWarning = True
    anchors = [(10, 20), (10, 30), (10, 40), (20, 20), (20, 40), (20, 60), (40, 40), (40, 60), (40, 80), (40, 100), (40, 120), (60, 60), (60, 80),
               (60, 100), (60, 120), (60, 140), (80, 100), (80, 120), (80, 140), (80, 160), (100, 120), (100, 140), (120, 120), (120, 150), (120, 200)]

    def __init__(self, x, y, w, h, ww, hh):
        self.valid = Anchor.isvalid(x, y, w, h, ww, hh)
        if Anchor.showWarning and not self.valid:
            warnings.warn("Invalid coordinates, Beware of implications! (%d,%d,%d,%d,%d,%d)" % (
                x, y, w, h, ww, hh), RuntimeWarning)
            Anchor.showWarning = False
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        if self.valid:
            self.id = Anchor.anchors.index((w, h))

    @staticmethod
    def isvalid(x, y, w, h, ww, hh):
        if (w, h) not in Anchor.anchors:
            return False
        l = x-w//2
        t = y-h//2
        if l < 0 or l >= ww:
            return False
        if t < 0 or t >= hh:
            return False
        r = max(l+w, w)
        b = max(t+h, h)
        if r > ww or b > hh:
            return False
        return True

    @staticmethod
    def from_ltrb(l, t, r, b, ww, hh):
        x = (l+r)//2
        y = (t+b)//2
        w = int(abs(r-l))
        h = int(abs(b-t))
        return Anchor(x, y, w, h, ww, hh)

    @staticmethod
    def from_ltwh(l, t, w, h, ww, hh):
        return Anchor(l+w//2, t+h//2, w, h, ww, hh)

    def offset(self, dx, dy, dw, dh):
        "Offsets the anchor. Doesnot create a copy."
        self.x += dx
        self.y += dy
        self.w += dw
        self.h += dh
        return anc

    def iou(self, a):
        if not isinstance(a, Anchor):
            return 0

        l, t, r, b = self.ltrb()
        ll, tt, rr, bb = a.ltrb()

        i = (min(r, rr) - max(l, ll))
        if i < 0:
            return 0.0
        i *= (min(b, bb) - max(t, tt))
        if i < 0:
            return 0.0
        u = self.h*self.w+a.h*a.w-i
        return i/u

    def ioudelta(self, other):
        """
        Calculates iou and delta x,y,w,h values for other as (x,y,id) tuple.
        """
        l, t, r, b = self.ltrb()
        x, y, id = other
        w, h = Anchor.anchors[id]
        w2 = w / 2
        h2 = h / 2
        ll, tt, rr, bb = x-w2, y-h2, x+w2, y+h2

        i = (min(r, rr) - max(l, ll))
        if i < 0:
            return [0.0]*5
        i *= (min(b, bb) - max(t, tt))
        if i < 0:
            return [0.0]*5
        u = self.h*self.w+h*w-i
        iou = i/u

        return [iou, self.x-x, self.y-y, self.w-w, self.h-h]

    def ltrb(self):
        ww = self.w//2
        hh = self.h//2
        return self.x-ww, self.y-hh, self.x+ww, self.y+hh

    def draw(self, pl="b"):
        l, t, r, b = self.ltrb()
        plt.plot([l, r, r, l, l], [t, t, b, b, t], pl)

    def to_tuple(self):
        if self.valid:
            return (self.x, self.y, self.id)
        else:
            return (self.x, self.y, self.w, self.h)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
        # return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h

    def __repr__(self):
        return str(self.to_tuple())

    @staticmethod
    def gen_anchors(ww, hh):
        anchs = list({
            Anchor(x, y, w, h, ww, hh)
            for x in ww
            for y in hh
            for w, h in Anchor.anchors
        })
        return [x for x in anchs if x.valid]

    @staticmethod
    def gen_anchor_tuples(xx, yy, ww, hh):
        return [
            (x, y, i)
            for x in xx
            for y in yy
            for i, (w, h) in enumerate(Anchor.anchors)
            if Anchor.isvalid(x, y, w, h, ww, hh)
        ]


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser(description="Extract anchors for dataset")
    args.add_argument("r", help="Feature extractor scale ratio",
                      action="store_const", const=8)
    args = args.parse_args()

    # Set the feature extractor ratio here. In my basenet case, it is 8
    r = args.r

    splits = ["train", "test"]
    # Additionally, "label" column will also be added.
    cols = ["iou", "dx", "dy", "dw", "dh"]

    for split in splits:
        if os.path.exists("anchors-"+split):
            print("anchors-"+split, "already exists. No need of anchors extraction")
            continue
        os.makedirs("anchors-"+split)

        df = pd.read_csv(split+".csv")

        for name in tqdm(df.name.unique(), desc=split):
            ww = df.loc[df.name == name, "imgwidth"].max()
            hh = df.loc[df.name == name, "imgheight"].max()
            if hh < 21:
                continue
            xx = range(r//2-1, ww, r)
            yy = range(r//2-1, hh, r)
            ancs = Anchor.gen_anchor_tuples(xx, yy, ww, hh)

            idf = pd.DataFrame(
                [[0.0, 0.0, 0.0, 0.0, 0.0]],
                index=pd.MultiIndex.from_tuples(ancs, 0, ["x", "y", "i"]),
                columns=cols
            )
            idf["label"] = -1

            for x, y, w, h, label in df.loc[df.name == name, ["x", "y", "width", "height", "label"]].values:
                anc = Anchor(x, y, w, h, ww, hh)
                ioudelta = idf.index.to_frame().apply(anc.ioudelta, axis=1, result_type="expand")
                ioudelta.columns = cols
                replace = ioudelta["iou"] > idf["iou"]
                idf.loc[replace, cols] = ioudelta[replace]
                idf.loc[replace, "label"] = label

            idx = idf.index.to_frame(index=False)
            idx.x = (idx.x-3)//8
            idx.y = (idx.y-3)//8
            idf.index = pd.MultiIndex.from_frame(idx)
            idf.label = idf.label.astype(int)

            idf.to_csv("anchors-"+split+"/"+name+".csv")
