import numpy as np
import cv2

def four_point_transform(image, pts, target_size=None):
    (tl, tr, br, bl) = pts

    if target_size is None:
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
    else:
        maxWidth, maxHeight = target_size

    if maxWidth < 10 or maxHeight < 10:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight),
        flags=cv2.INTER_CUBIC
    )
    return warped