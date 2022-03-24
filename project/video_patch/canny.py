# Implement Canny Edge Detection from Scratch with Pytorch

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image

import pdb


# transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))


def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)

    # (Pdb) gaussian_1D
    # array([-1.,  0.,  1.])
    # (Pdb) x
    # array([[-1.,  0.,  1.],
    #        [-1.,  0.,  1.],
    #        [-1.,  0.,  1.]])
    # (Pdb) y
    # array([[-1., -1., -1.],
    #        [ 0.,  0.,  0.],
    #        [ 1.,  1.,  1.]])

    distance = (x**2 + y**2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-((distance - mu) ** 2) / (2 * sigma**2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma**2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    x, y = np.meshgrid(range, range)

    # (Pdb) range
    # array([-1.,  0.,  1.])
    # (Pdb) x
    # array([[-1.,  0.,  1.],
    #        [-1.,  0.,  1.],
    #        [-1.,  0.,  1.]])
    # (Pdb) y
    # array([[-1., -1., -1.],
    #        [ 0.,  0.,  0.],
    #        [ 1.,  1.,  1.]])

    sobel_2D_numerator = x
    sobel_2D_denominator = x**2 + y**2
    # (Pdb) sobel_2D_denominator
    # array([[2., 1., 2.],
    #        [1., 0., 1.],
    #        [2., 1., 2.]])

    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    # (Pdb) sobel_2D
    # array([[-0.5,  0. ,  0.5],
    #        [-1. ,  0. ,  1. ],
    #        [-0.5,  0. ,  0.5]])

    return sobel_2D


def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3  # actual size of the directional kernel
    # increase for a while to avoid interpolation when rotating
    k_increased = k_thin + 2

    # get 0° angle directional kernel
    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1 :] = -1

    # (Pdb) thin_kernel_0
    # array([[ 0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  1., -1., -1.],
    #        [ 0.,  0.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  0.]])

    # rotate the 0° angle directional kernel to get the other ones
    thin_kernels = []
    for angle in range(start, end, step):
        (h, w) = thin_kernel_0.shape
        # get the center to not rotate around the (0, 0) coord point
        center = (w // 2, h // 2)
        # apply rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        # (Pdb) rotation_matrix
        # array([[ 1.,  0.,  0.],
        #    [-0.,  1.,  0.]])

        kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

        # get the k=3 kerne
        kernel_angle = kernel_angle_increased[1:-1, 1:-1]

        # (Pdb) kernel_angle
        # array([[ 0.,  0.,  0.],
        #        [ 0.,  1., -1.],
        #        [ 0.,  0.,  0.]])
        # (Pdb) abs(kernel_angle) == 1
        # array([[False, False, False],
        #        [False,  True,  True],
        #        [False, False, False]])

        is_diag = abs(kernel_angle) == 1  # because of the interpolation
        kernel_angle = kernel_angle * is_diag  # because of the interpolation
        thin_kernels.append(kernel_angle)

    # (Pdb) for k in thin_kernels: print(k)
    # [[ 0.  0.  0.]
    #  [ 0.  1. -1.]
    #  [ 0.  0.  0.]]
    # [[ 0. -0. -1.]
    #  [ 0.  1. -0.]
    #  [ 0.  0.  0.]]
    # [[ 0. -1.  0.]
    #  [ 0.  1.  0.]
    #  [ 0.  0.  0.]]
    # [[-1. -0.  0.]
    #  [-0.  1.  0.]
    #  [ 0.  0.  0.]]
    # [[ 0.  0.  0.]
    #  [-1.  1.  0.]
    #  [ 0.  0.  0.]]
    # [[ 0.  0.  0.]
    #  [-0.  1.  0.]
    #  [-1. -0.  0.]]
    # [[ 0.  0.  0.]
    #  [ 0.  1.  0.]
    #  [ 0. -1.  0.]]
    # [[ 0.  0.  0.]
    #  [ 0.  1. -0.]
    #  [ 0. -0. -1.]]
    return thin_kernels


class CannyFilter(nn.Module):
    def __init__(self, k_gaussian=3, mu=0, sigma=1, k_sobel=3):
        super(CannyFilter, self).__init__()

        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=k_gaussian, padding=k_gaussian // 2, bias=False
        )
        self.gaussian_filter.weight.requires_grad = False
        self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)

        # sobel
        sobel_2D = get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False
        )
        self.sobel_filter_x.weight.requires_grad = False
        self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)

        self.sobel_filter_y = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=k_sobel, padding=k_sobel // 2, bias=False
        )
        self.sobel_filter_y.weight.requires_grad = False
        self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)

        # thin
        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=thin_kernels[0].shape,
            padding=thin_kernels[0].shape[-1] // 2,
            bias=False,
        )
        self.directional_filter.weight.requires_grad = False
        self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)

        # hysteresis
        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.hysteresis.weight.requires_grad = False
        self.hysteresis.weight[:] = torch.from_numpy(hysteresis)

    def forward(self, image, low_threshold=None, high_threshold=None, hysteresis=True):
        # set the setps tensors
        B, C, H, W = image.shape
        blurred = torch.zeros((B, C, H, W)).to(image.device)
        grad_x = torch.zeros((B, 1, H, W)).to(image.device)
        grad_y = torch.zeros((B, 1, H, W)).to(image.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(image.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(image.device)

        # gaussian
        for c in range(C):
            blurred[:, c : c + 1] = self.gaussian_filter(image[:, c : c + 1])

            grad_x = grad_x + self.sobel_filter_x(blurred[:, c : c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c : c + 1])

        # thick edges
        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
        grad_orientation = torch.atan(grad_y / grad_x)

        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45
        # grad_orientation.size()
        # torch.Size([1, 1, 256, 256])

        # thin edges
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # directional.size()
        # torch.Size([1, 8, 256, 256])

        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # thresholds
        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        thin_edges = thin_edges.float()

        # return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges
        return thin_edges


if __name__ == "__main__":
    model = T.GaussianBlur(kernel_size=3) # Pytorch 1.7.0
    # model = CannyFilter()
    model = model.cuda()
    model.eval()

    print(model)

    image = Image.open("/tmp/lena.png")
    input_tensor = T.ToTensor()(image).unsqueeze(0)
    input_tensor = input_tensor.cuda()

    with torch.no_grad():
        # output_tensor = model(input_tensor, low_threshold=0.15, high_threshold=0.30)
        output_tensor = model(input_tensor)

    output_image = T.ToPILImage()(output_tensor.squeeze(0))

    output_image.show()
