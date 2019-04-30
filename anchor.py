import warnings
import matplotlib.pyplot as plt


class Anchor:

    showWarning = True
    anchors = [(10, 20), (10, 30), (10, 40), (20, 20), (20, 40), (20, 60), (40, 40), (40, 60), (40, 80), (40, 100), (40, 120), (60, 60), (60, 80),
               (60, 100), (60, 120), (60, 140), (80, 100), (80, 120), (80, 140), (80, 160), (100, 120), (100, 140), (120, 120), (120, 150), (120, 200)]

    def __init__(self, x, y, w, h, ww, hh):
        l = max(x-w//2, 0)
        t = max(y-h//2, 0)
        w = min(w, ww-l-1)
        h = min(h, hh-t-1)
        if Anchor.showWarning and (w <= 0 or h <= 0):
            #             raise Exception("Got w,h =",(w,h))
            warnings.warn("Got w=%d, h=%d, Beware of implications!" % (w, h),)
            Anchor.showWarning = False
        self.x = l+w//2
        self.y = t+h//2
        self.w = w
        self.h = h
        self.ww = ww
        self.hh = hh

    @classmethod
    def from_ltrb(cls, l, t, r, b, ww, hh):
        x = (l+r)//2
        y = (t+b)//2
        w = int(abs(r-l))
        h = int(abs(b-t))
        return Anchor(x, y, w, h, ww, hh)

    @classmethod
    def from_ltwh(cls, l, t, w, h, ww, hh):
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
        a = Anchor(*a, self.ww, self.hh)
        return self.iou(a)

    def ltrb(self):
        ww = self.w//2
        hh = self.h//2
        return self.x-ww, self.y-hh, self.x+ww, self.y+hh

    def draw(self, pl="b"):
        l, t, r, b = self.ltrb()
        plt.plot([l, r, r, l, l], [t, t, b, b, t], pl)

    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h, self.ww, self.hh))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.w == other.w and self.h == other.h and self.ww == other.ww and self.hh == other.hh

    def to_tuple(self):
        return (self.x, self.y, self.w, self.h)

    @staticmethod
    def gen_anchors(ww, hh, anchors):
        anchs = list({
            Anchor(x, y, w, h, ww, hh)
            for x in range(ww)
            for y in range(hh)
            for w, h in anchors
        })
        return [x for x in anchs if x.w > 0 and x.h > 0]

    @staticmethod
    def gen_anchors_tuples(ww, hh, anchors):
        return [x.to_tuple() for x in Anchor.gen_anchors(ww, hh, anchors)]

    @staticmethod
    def gen_anchors_scaled(ww, hh, anchors, r=1/8):
        anchors = [(x*r, y*r) for x, y in anchors]
        print(anchors)

    def ismatching(self, anchors, thresh=0.7):
        "Finds if any of the given anchors matches the current bbox."
        for a in self.gen_anchors(self.ww, self.hh, anchors):
            if self.iou(a) > thresh:
                return True
        return False

    def maxiou(self, anchors):
        mx = i = 0.0
        for a in self.gen_anchors(self.ww, self.hh, anchors):
            i = self.iou(a)
            if i > mx:
                mx = i
        return mx

    def __repr__(self):
        return "(%d,%d,%d,%d,%d,%d)" % (self.x, self.y, self.w, self.h, self.ww, self.hh)
