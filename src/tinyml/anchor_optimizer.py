"""K-Means anchor optimizer using IoU-distance for YOLO-FastestV2 on AZN dataset."""
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AnchorSet:
    anchors: List[Tuple[float, float]]
    avg_iou: float

def _iou(box, anchor):
    inter = min(box[0], anchor[0]) * min(box[1], anchor[1])
    union = box[0]*box[1] + anchor[0]*anchor[1] - inter
    return inter / max(union, 1e-9)

def kmeans_anchors(wh_pairs, k=6, num_iters=300, seed=42):
    random.seed(seed)
    boxes = list(wh_pairs)
    n = len(boxes)
    if n < k:
        raise ValueError("Need at least k={} boxes, got {}.".format(k, n))
    centroids = [random.choice(boxes)]
    for _ in range(k-1):
        dists = [min(1.0-_iou(b,c) for c in centroids) for b in boxes]
        total = sum(dists); r = random.uniform(0, total); cum = 0.0
        for b, d in zip(boxes, dists):
            cum += d
            if cum >= r: centroids.append(b); break
    for _ in range(num_iters):
        clusters = [[] for _ in range(k)]
        for box in boxes:
            clusters[max(range(k), key=lambda j: _iou(box, centroids[j]))].append(box)
        new_c = [(sum(b[0] for b in c)/len(c), sum(b[1] for b in c)/len(c)) if c else random.choice(boxes) for c in clusters]
        if new_c == centroids: break
        centroids = new_c
    avg_iou = sum(max(_iou(b,c) for c in centroids) for b in boxes) / max(n, 1)
    return AnchorSet(anchors=sorted(centroids, key=lambda a: a[0]*a[1]), avg_iou=avg_iou)
