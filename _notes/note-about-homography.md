---
title: 'About homography'
date: 2024-05-28
permalink: /study-note/2024/05/about-homography/
tags:
  - study note
---

Recently I came across an image homography problem during the work. I have studied homography long ago on my sophomore image processing class, but honestly, I have never really thorougly understand the implementation details of the algorithm. I think this is a good time to write them down as a note on my website.

Descriptions
======

What is homography? For those who don't know computer grpahics, this word might sounds sophisticated. In fact, homography is a very understandable thing. Just think of a case when you have a multi-camera system and have taken multiple images. You may want to combine these images taken from difference perspectives to create something like a panorama or a all-around picture. In order to map these images to a single canvas while following the real-world geometry, you will need a special transformation for each image, which is called homography. 

As for the problem I encountered, I was given camera calibration matrices to produce a top-down-view (or bird-eye-view) picture from camera images. 

Math Formulas
======

I actually don't like the description about homography on wikipedia as they make things too complicated. The best mathematical explanation about homography I found is on [this website](https://towardsdatascience.com/estimating-a-homography-matrix-522c70ec4b2c).

To begin with, we need to know the calibration parameters of our cameras. Since we have multiple cameras taking pictures, we need to know their relative positions, focal lengths, and photo image sizes. Color is a totally different problem, so we just assume all the cameras have set to use the same color setting. 

Calibration parameters of a camera come in two parts: the extrinsic parameters, and the intrinsic parameters. They can be both written in matrix form:

$$
\displaylines{
  C_{ext} = 
  \begin{array}{c|c}
    R & T \\
  \end{array} 
  = 
  \[
    \left[
      \begin{array}{ccc|c}
        r_{11} & r_{12} & r_{13} & T_{x} \\
        r_{21} & r_{22} & r_{23} & T_{y} \\
        r_{31} & r_{32} & r_{33} & T_{z} \\
        0 & 0 & 0 & 1 \\
      \end{array}
    \right]
  \] \\\
  C_{int} = 
  \begin{bmatrix}
    f_{x} & 0 \frac{w}{2} \\
    0 & f_{y} & \frac{h}{2} \\
    0 & 0 & 1 \\
  \end{bmatrix} \\\
}
$$

The extrinsic matrix \\(C_{ext}\\) transforms any point \\((x, y, z)\\) in world to camera coordinate. The \\(R\\) and \\(T\\) are rotation matrix and translation matrix. The intrinsic matrix \\(C_{int}\\) transforms any point in camera coordinate to a location on the image. 

Therefore, intuitively, we should be able to transform any point in world to somewhere on the image like this:
 
$$
\textbf{p}_{i} = C_{int} C_{ext} \textbf{p}_{w}
$$

However, depending on what kind of result we want, we need to adjust the matrices to make linear algebra works. In my case, I want to get a bird-eye-view image, so I need to transform pixel from different camera images to a <b>zero-height</b> plane in world, which is the ground. Hence, the transform will become:

$$
\begin{bmatrix}
  kx_{i} \\
  ky_{i} \\
  k \\
\end{bmatrix}
=
\begin{bmatrix}
  f_{x} & 0 \frac{w}{2} \\
  0 & f_{y} & \frac{h}{2} \\
  0 & 0 & 1 \\
\end{bmatrix}
\[
  \left[
    \begin{array}{ccc|c}
      r_{11} & r_{12} & r_{13} & T_{x} \\
      r_{21} & r_{22} & r_{23} & T_{y} \\
      r_{31} & r_{32} & r_{33} & T_{z} \\
    \end{array}
  \right]
\]
\begin{bmatrix}
  x_{w} \\
  y_{w} \\
  0 \\
  1 \\
\end{bmatrix}
$$

The last row of extrinsic matrix is dropped since we are not doing an affine transform. The resulting point in image is scaled by some arbitrary "depth" value `k`. By normalizing the `x` , `y` with `k`, we can natrually project the pixels on the image plane by distance.


Code and Implementation
======

A function in the library `cameratransform` called [Camera.getTopViewOfImage()](https://cameratransform.readthedocs.io/en/stable/_modules/cameratransform/camera.html#Camera.getTopViewOfImage) inspired my implementation. I followed the coding style of the original function, but made it more independent and easier to reuse in other projects.

```python
import cv2
import numpy as np

def convertBEV(images, extrinsics, intrinsics, output_size, bev_area):
    """
    Args:
      - images: array of N camera images, shape: (N, C, H, W)
      - extrinsics: extrinsic matrix, shape: (N, 4, 4)
      - intrinsics: intrinsic matrix, shape: (N, 3, 3)
      - output_size: size of the resulting bev_image, (H, W)
      - bev_area: the area that we want to include in our bev image, [x_min, x_max, y_min, y_max], in meters
    Outputs:
      - the resulting bev image, shape: (C, H, W)
    """

    canvas = np.zeros((3, output_size[0], output_size[1]))

    image_num = images.shape[0]
    # check dimension match
    assert extrinsics.shape[0] == image_num
    assert intrinsics.shape[0] == image_num

    # iterate cameras
    for i in range(image_num):
        # load image
        image_buffer = images[i, :, :, :].copy()
        # map color to [0, 1]
        min_val = image_buffer.min()
        max_val = image_buffer.max()
        # avoid divide by zero
        if min_val != max_val:
            (image_buffer - min_val) / (max_val - min_val)
        image_buffer = image_buffer.astype(np.float32)
        image_buffer = image_buffer.transpose(1, 2, 0)  # (H, W, C)
        # color mapping
        image_buffer = cv2.cvtColor(image_buffer, cv2.RGB2BGR)

        # compute image2BEV pixel mapping
        x, y = getMapping(extrinsics[i, :, :], intrinsics[i, :, :], bev_area, output_size)

        # remap the image
        image = cv2.remap(image_buffer, x, y
                          interpolation=cv2.INTER_NEAREST,
                          borderValue=[0, 0, 0, 0])

        # overlay on canvas
        canvas = cv2.addWeighted(canvas, 1, image, 0.5) # alpha blending

    return canvas


def getMapping(camera2ego, camera2image, bev_area=None, output_size=None, Z=0):
    """
    Get the pixel location mapping from the input image to the output bev image.

    Args:
      - camera2ego: extrinsic, shape: (N, 4, 4)
      - camera2image: intrinsic, shape: (N, 3, 3)
      - bev_area: the area that we want to include in our bev image, [x_min, x_max, y_min, y_max], in meters
      - output_size: size of the resulting bev_image, (H, W)
      - Z: "height" of the plane on which to project
    """

    # get transformation
    ego2camera = np.linalg.inv(camera2ego)
    rotation = ego2camera[:3, :2]                   # (3, 2), get only x, y
    translation = ego2camera[:3, 3].reshape(3, 1)   # (3, 1), get x, y, z
    transform = np.hstack((rotation, translation))  # (3, 3)

    # get meters per pixel
    x_scale = (bev_area[1] - bev_area[0]) / output_size[0]
    y_scale = (bev_area[3] - bev_area[2]) / output_size[1]

    # generate mesh grid, representing sampling points in world
    mesh = np.array(np.meshgrid(
        np.arange(bev_area[0], bev_area[1], x_scale),
        np.arange(bev_area[3], bev_area[2], y_scale)
    ))
    # reshape
    mesh_points = mesh.reshape(2, mesh.shape[1] * mesh.shape[2]).T                # (H*W, 2)
    mesh_points = np.hstack((mesh_points, Z*np.ones((mesh_points.shape[0], 1))))  # (H*W, 3)

    # map points from world to image
    mesh_points = mesh_points @ transform.T @ camera2image.T
    
    # normalize x,y position by z value
    mesh_points[:, 0] = mesh_points[:, 0] / mesh_points[:, 2]
    mesh_points[:, 1] = mesh_points[:, 1] / mesh_points[:, 2]
    # filter points behind image (z <= 0)
    mesh_points[mesh_points[:, 2] <= 0.0] = np.nan
    mesh_points = mesh_points[:, :2]  # (H*W, 2)

    # reshape
    mapping = mesh_points.T.reshape(mesh.shape).astype(np.float32)[:, ::-1, :]  # (2, H, W)

    return mapping


```