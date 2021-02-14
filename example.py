import random
import numpy as np
import scipy.ndimage.morphology
import NumPyDraw.view as view
import NumPyDraw.npd3d as npd3d
import NumPyRandomShapes3D.nprs as nprs
import sts

# fixed seed
random.seed(42)

array = np.zeros((256, 256, 256), dtype=np.uint8)
center = 64, 128, 128
radius = 50, 100, 100
rot = np.deg2rad(20), np.deg2rad(-10), np.deg2rad(10)
array[npd3d.spheroid_coordinate(array.shape, center, radius, rot)] = 1
segmentation = nprs.elastic_deformation(array)

# scribble points
pos, neg = sts.segmentation_to_scribble_points(segmentation)
scribble_points_pos = np.zeros(segmentation.shape)
scribble_points_neg = np.zeros(segmentation.shape)
for a in pos:
    scribble_points_pos[a] = 1
for a in neg:
    scribble_points_neg[a] = 1
scribble_points_pos = scipy.ndimage.morphology.binary_dilation(scribble_points_pos,
                                                               scipy.ndimage.morphology.generate_binary_structure(3, 2),
                                                               iterations=3)
scribble_points_neg = scipy.ndimage.morphology.binary_dilation(scribble_points_neg,
                                                               scipy.ndimage.morphology.generate_binary_structure(3, 2),
                                                               iterations=3)
scribble_points = (np.copy(segmentation)+1)/3
scribble_points[scribble_points_pos == 1] = 1
scribble_points[scribble_points_neg == 1] = 0

# scribble skeleton
scribble_skeleton_ = sts.segmentation_to_scribble_skeleton(segmentation)
scribble_skeleton_ = scipy.ndimage.morphology.binary_dilation(scribble_skeleton_,
                                                              scipy.ndimage.morphology.generate_binary_structure(3, 2),
                                                              iterations=2)
scribble_skeleton = (np.copy(segmentation)+1)/3
scribble_skeleton[scribble_skeleton_ == 1] = 1

view.show_stack(segmentation)
view.show_stack(scribble_points, vmin=0, vmax=1)
view.show_stack(scribble_skeleton, vmin=0, vmax=1)

view.gif_stack(segmentation, "example/example.gif")
view.gif_stack(scribble_points, "example/example_points.gif", vmin=0, vmax=1)
view.gif_stack(scribble_skeleton,  "example/example_skeleton.gif", vmin=0, vmax=1)

# 0-255 RGB tif stack for view using Fiji 3D Viewer
'''
import tifffile

segmentation_tif = np.repeat(np.expand_dims(segmentation*255, -1), 3, -1)
tifffile.imwrite("example/example.tif", segmentation_tif)
'''
