import numpy as np
import skimage.morphology
import skimage.measure
import edt


def segmentation_to_scribble_points(segmentation, points=20):
    """
    return two array filled with positive and negative points coordinate for the given segmentation.
    """
    # convert segmentation into positive and negative distance map
    dt_pos = edt.edt(segmentation)

    dt_neg = -edt.edt(1-segmentation)
    dt_neg = dt_neg - dt_neg.min()
    dt_neg[segmentation == 1] = 0

    # flatten and sort distance map
    coord = np.arange(segmentation.size).flatten()
    dt_pos_flat = dt_pos.flatten()
    argsort = np.argsort(dt_pos_flat)
    coord_flat_pos_sorted = np.flip(np.take_along_axis(coord, argsort, axis=0))

    coord = np.arange(segmentation.size).flatten()
    dt_neg_flat = dt_neg.flatten()
    argsort = np.argsort(dt_neg_flat)
    coord_flat_neg_sorted = np.flip(np.take_along_axis(coord, argsort, axis=0))

    # create points
    pos_points = []
    neg_points = []
    seg_size = np.sum(segmentation)
    background_size = segmentation.size - np.sum(segmentation)
    for _ in range(points):
        a = abs(int(np.random.normal(loc=0.0, scale=seg_size//4)))
        pos_points.append(np.unravel_index(coord_flat_pos_sorted[a], segmentation.shape))

        a = abs(int(np.random.normal(loc=0.0, scale=background_size//4)))
        neg_points.append(np.unravel_index(coord_flat_neg_sorted[a], segmentation.shape))

    return pos_points, neg_points


def segmentation_to_scribble_skeleton(segmentation):
    """
    return an array filled with scribble (skeleton) for the given segmentation.
    """
    scribble = skimage.morphology.skeletonize_3d(segmentation)

    return scribble
