import warnings
import matplotlib.pyplot as plt


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
    def from_ltrb(l, t, r, b, ww, hh):
        x = (l+r)//2
        y = (t+b)//2
        w = int(abs(r-l))
        h = int(abs(b-t))
        return Anchor(x, y, w, h, ww, hh)

    @staticmethod
    def from_ltwh(l, t, w, h, ww, hh):
        return Anchor(l+w//2, t+h//2, w, h, ww, hh)

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

    def iou_tuple(self, a):
        l, t, r, b = self.ltrb()
        x, y, id = a
        w, h = Anchor.anchors[id]
        w2 = w / 2
        h2 = h / 2
        ll, tt, rr, bb = x-w2, y-h2, x+w2, y+h2

        i = (min(r, rr) - max(l, ll))
        if i < 0:
            return 0.0
        i *= (min(b, bb) - max(t, tt))
        if i < 0:
            return 0.0
        u = self.h*self.w+h*w-i
        return i/u

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

    def ioudelta(self, other):
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

    def __repr__(self):
        return str(self.to_tuple())

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

    # def ismatching(self, anchors, thresh=0.7):
    #     "Finds if any of the given anchors matches the current bbox."
    #     for a in self.gen_anchors(self.ww, self.hh):
    #         if self.iou(a) > thresh:
    #             return True
    #     return False

    # def maxiou(self, anchors):
    #     mx = i = 0.0
    #     for a in self.gen_anchors(self.ww, self.hh):
    #         i = self.iou(a)
    #         if i > mx:
    #             mx = i
    #     return mx
