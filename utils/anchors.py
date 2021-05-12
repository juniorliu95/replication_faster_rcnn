"""functions and attributes of anchors"""

import numpy as np

def generate_anchor_base(base_size=16, ratios=[0.5, 1., 2.], anchor_scales=[8, 16, 32]):
    """
    The basic anchor definition. Using top-left and bottom-right point.
    Inputs:
        base_size: The base of the size of the anchors on the image. base_size * anchor_scales is the side length of the anchor.
        ratios: The ratios between h and w of the anchors.
        anchor_scales: times of base_size.
    Output:
        K standard anchors to be put on the selected points of the image. 
        Anchors are defined by the top-left and bottom right points (x1, y1, x2, y2).
    """

    anchor_base = np.zeros([len(ratios)*len(anchor_scales), 4]).astype(np.float32)

    for r_ind, ratio in enumerate(ratios):
        for s_ind, anchor_scale in enumerate(anchor_scales):
            anchor_index = r_ind*len(anchor_scales) + s_ind

            h = base_size * anchor_scale * np.sqrt(ratio)
            w = base_size * anchor_scale * np.sqrt(1./ratio)

            anchor_base[anchor_index, 0] = - h/2 # x1
            anchor_base[anchor_index, 1] = - w/2 # y1
            anchor_base[anchor_index, 2] = h/2 # x2 
            anchor_base[anchor_index, 3] = w/2 # y2

    return anchor_base

def generate_anchors(anchor_base, feat_stride, width, height):
    """
    get all the anchors on the image
    Inputs:
        anchor_base: the standard anchors for each point.
        feat_stride: the ratio between image size and conv size.
        height, width: the number of anchors on each side of the image.
    Outputs:
        all the anchors on the image. [K*conv_H*conv_W, 4]
    """

    # generate the center of all the anchors
    # notice here in definition of meshgrid, x_pos increase in the horizontal direction
    x_pos = np.arange(0, feat_stride*width, feat_stride)
    y_pos = np.arange(0, feat_stride*height, feat_stride)
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)
    
    # xy: [Na, 4]
    xy = np.stack((x_pos.flatten(), y_pos.flatten(),
                      x_pos.flatten(), y_pos.flatten(),), axis=1)

    K = anchor_base.shape[0] # number of anchors for each point
    Na = xy.shape[0] # number of points

    # generate the x1, y1, x2, y2 for all anchors
    anchor = anchor_base[None,:,:] + xy[:,None,:] # braodcast to [Na, K, 4]
    anchor = anchor.reshape((K * Na, 4)).astype(np.float32)

    return anchor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ratios = [0.5, 1., 2.] # h/m
    anchor_scales = [8, 16, 32] # 128, 256, 512

    anchor_base = generate_anchor_base(base_size=16, ratios=ratios, anchor_scales=anchor_scales)
    anchor = generate_anchors(anchor_base, 10, 10, 10)

    x = (anchor[:,0]+anchor[:,2])/2
    y = (anchor[:,1]+anchor[:,3])/2
    plt.scatter(x, y)
    plt.savefig("anchor_points.png")
    
