"""Complete IoU (CIoU) regression loss for YOLO-FastestV2 training."""
import math

def _box_iou(pred, gt):
    px, py, pw, ph = pred; gx, gy, gw, gh = gt
    px1,py1,px2,py2 = px-pw/2,py-ph/2,px+pw/2,py+ph/2
    gx1,gy1,gx2,gy2 = gx-gw/2,gy-gh/2,gx+gw/2,gy+gh/2
    inter = max(0.0, min(px2,gx2)-max(px1,gx1)) * max(0.0, min(py2,gy2)-max(py1,gy1))
    union = pw*ph + gw*gh - inter
    iou = inter / max(union, 1e-9)
    c2 = (max(px2,gx2)-min(px1,gx1))**2 + (max(py2,gy2)-min(py1,gy1))**2 + 1e-9
    rho2 = (px-gx)**2 + (py-gy)**2
    v = (4/math.pi**2) * (math.atan(gw/max(gh,1e-9)) - math.atan(pw/max(ph,1e-9)))**2
    alpha = v / (1-iou+v+1e-9) if (1-iou+v) > 1e-9 else 0.0
    return iou - (rho2/c2) - alpha*v

def ciou_loss(pred, gt):
    return 1.0 - _box_iou(pred, gt)
