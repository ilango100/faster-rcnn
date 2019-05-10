import numpy as np
from anchors import Anchor


# Process the output from rpn to extract the anchors and probabilities
def get_anchors(probs, deltas, prob_thresh=0.6, r=8):
    ww, hh = probs.shape[:2]*r
    sel = probs >= prob_thresh

    # Take indices of selection and create anchors
    idxs = np.transpose(np.nonzero(sel))
    anchors = [Anchor(x*r, y*r, *Anchor.anchors[i], ww, hh)
               for y, x, i in idxs]

    # Drop invalid anchors
    anchors = [anc for anc in anchors if anc.isvalid]

    # Offset the anchors
    for anc in anchors:
        anc.offset(*deltas[anc.y, anc.x, anc.id*4:(anc.id+1)*4])
    return np.array(anchors), probs[sel]


def nms(anchors, probs, iou_thresh=0.7):
    # Sort the arrays according to descending probability
    ordr = np.argsort(-probs)
    anchors, probs = anchors[ordr], probs[ordr]

    # Find indexes to be dropped
    dropidxs = []
    for i, anc in enumerate(anchors):
        for j, ancc in enumerate(anchors[:i]):
            if anc.iou(ancc) > iou_thresh:
                dropidxs.append(j)

    # Drop the anchors
    anchors = np.delete(anchors, dropidxs)

    return anchors
