from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import os
import cv2
import copy
import numpy as np

# import scipy.io as sio
from utils.common_utils import (
    interp,
    BackwardFlowConsistCheck,
    ForwardFlowConsistCheck,
    ConsistCheck,
)

# , get_KeySourceFrame_flowNN

import pdb


def get_flowNN(args, video, mask, forward_flow, backward_flow):

    # video:      image_height x image_width x 3 x num_frame
    # mask:       image_height x image_width x num_frame
    # forward_flow: image_height x image_width x 2 x (num_frame - 1)
    # backward_flow: image_height x image_width x 2 x (num_frame - 1)

    # video.shape -- (512, 960, 3, 10), [0.0, 1.0]

    # mask.shape -- (512, 960, 10)
    # (Pdb) mask.shape, mask.min(), mask.max()
    # ((512, 960, 10), False, True)

    # (Pdb) forward_flow.shape, forward_flow.min(), forward_flow.max()
    # ((512, 960, 2, 9), -1.8590815, 18.602379)

    # pp args.Nonlocal --False
    num_candidate = 2
    image_height, image_width, num_frame = mask.shape
    num_mask_one_pixel = np.sum(mask)

    # |--------------------|
    # |       y            |
    # |   x   *            |
    # |                    |
    # |--------------------|

    tmask = np.where(mask == 1)
    sub = np.concatenate(
        (
            tmask[0].reshape(-1, 1),
            tmask[1].reshape(-1, 1),
            tmask[2].reshape(-1, 1),
        ),
        axis=1,
    )
    # sub.shape -- (403971, 3) ==> 82.19%d

    video_mask_index = np.ones((image_height, image_width, num_frame)) * -1
    # video_mask_index[r, c, t] gives the index of the missing pixel@[r, c, t] in sub
    for idx in range(len(sub)):  # len(sub) - 403971
        video_mask_index[sub[idx, 0], sub[idx, 1], sub[idx, 2]] = idx
    # video_mask_index.shape -- (512, 960, 10)

    # mask_one_flow: num_mask_one_pixel x 3 x 2
    mask_one_flow = np.ones((num_mask_one_pixel, 3, 2)) * 99999

    # video_flow_neighbor:  image_height x image_width x num_frame x 2
    # First channel stores backward flow neighbor, second channel stores forward flow neighbor.
    video_flow_neighbor = np.ones((image_height, image_width, num_frame, 2)) * 99999
    video_flow_neighbor[mask, :] = 0
    # mask.shape -- (512, 960, 10)
    # video_flow_neighbor[mask == True, :]  ==> 0.0
    # video_flow_neighbor[mask == False, :] ==> 99999.0

    consistency_map = np.zeros((image_height, image_width, num_frame, num_candidate))
    consistency_uv = np.zeros((image_height, image_width, 2, 2, num_frame))

    # 1. Forward Pass (backward flow propagation)
    print("Forward Pass......")
    nb_index = 0  # neighbor index
    for t in range(1, num_frame):
        # frame t
        hole_pixel_range = sub[:, 2] == t
        #  hole_pixel_range.shape --(403971,), array([False, ..., False])

        # Hole pixel location at frame t, i.e. [r, c, t]
        hole_pixel_position = sub[hole_pixel_range, :]
        # hole_pixel_position.shape -- (40097, 3), dtype -- dtype('int64')
        hole_pixel_rows = hole_pixel_position[:, 0]
        hole_pixel_cols = hole_pixel_position[:, 1]

        # Calculate the backward flow neighbor. Should be located at frame t-1
        backward_flow_position = copy.deepcopy(hole_pixel_position).astype(np.float32)

        backward_flow_horizont = backward_flow[:, :, 0, t - 1]
        backward_flow_vertical = backward_flow[:, :, 1, t - 1]  # t --> t-1
        # backward_flow_horizont.shape -- (512, 960)

        forward_flow_horizont = forward_flow[:, :, 0, t - 1]
        forward_flow_vertical = forward_flow[:, :, 1, t - 1]  # t-1 --> t

        backward_flow_position[:, 0] += backward_flow_vertical[hole_pixel_rows, hole_pixel_cols]
        backward_flow_position[:, 1] += backward_flow_horizont[hole_pixel_rows, hole_pixel_cols]
        backward_flow_position[:, 2] -= 1 # frame == t - 1

        # Round the backward flow neighbor location
        flow_neighbor_position = np.round(copy.deepcopy(backward_flow_position)).astype(np.int32) # (40097, 3)

        # Chen: I should combine the following two operations together
        # Check the backward/forward consistency
        consist_range = BackwardFlowConsistCheck(
            backward_flow_position,
            forward_flow_vertical,
            forward_flow_horizont,
            hole_pixel_position,
            args.consistencyThres,
        )

        # Check out-of-boundary
        # Last column and last row does not have valid gradient
        valid_neighbor_range = np.logical_and(
            np.logical_and(
                flow_neighbor_position[:, 0] >= 0,
                flow_neighbor_position[:, 0] <= image_height - 1,
            ),
            np.logical_and(
                flow_neighbor_position[:, 1] >= 0,
                flow_neighbor_position[:, 1] <= image_width - 1,
            ),
        )
        # (Pdb) valid_neighbor_range -- array([ True,  True,  True, ...,  True,  True,  True])
        # (Pdb) valid_neighbor_range.shape -- (40097,)

        # flow_neighbor_position
        # array([[113, 386,   0],
        #        ...,
        #        [395, 397,   0]], dtype=int32)
        # (Pdb) flow_neighbor_position.shape -- (40097, 3)
     
        # Only work with pixels that are not out-of-boundary
        hole_pixel_position = hole_pixel_position[valid_neighbor_range, :]
        # (Pdb) hole_pixel_position.shape -- (40097, 3)

        backward_flow_position = backward_flow_position[valid_neighbor_range, :]

        flow_neighbor_position = flow_neighbor_position[valid_neighbor_range, :]
        flow_neighbor_rows = flow_neighbor_position[:, 0]
        flow_neighbor_cols = flow_neighbor_position[:, 1]

        consist_range = consist_range[valid_neighbor_range]
        # (Pdb) consist_range -- array([ True,  True,  True, ...,  True,  True,  True])
        # (Pdb) consist_range.shape -- (40097,)

        # Case 1: If mask[round(r'), round(c'), t-1] == 0,
        #         the backward flow neighbor of [r, c, t] is known.
        #         [r', c', t-1] is the backward flow neighbor.

        # known_range: Among all backward flow neighbors, which pixel is known.
        known_range = mask[flow_neighbor_rows, flow_neighbor_cols, t - 1] == 0

        known_consist_range = np.logical_and(known_range, consist_range)

        # save backward flow neighbor backward_flow_position in mask_one_flow
        mask_one_flow[
            video_mask_index[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], t].astype(np.int32),
            :,
            nb_index,
        ] = backward_flow_position[known_consist_range, :]

        # mark [r, c, t] in video_flow_neighbor as 1
        video_flow_neighbor[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            t,
            nb_index,
        ] = 1

        # pdb.set_trace()

        # video_flow_neighbor[:, :, :, 0]
        # 0: Backward flow neighbor can not be reached
        # 1: Backward flow neighbor can be reached
        # -1: Pixels that do not need to be completed

        bf_uv = ConsistCheck(forward_flow[:, :, :, t - 1], backward_flow[:, :, :, t - 1])

        consistency_uv[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            nb_index,
            0,
            t,
        ] = np.abs(bf_uv[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], 0])
        consistency_uv[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            nb_index,
            1,
            t,
        ] = np.abs(bf_uv[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], 1])

        # Case 2: If mask[round(r'), round(c'), t-1] == 1,
        #  the pixel@[round(r'), round(c'), t-1] is also occluded.
        #  We further check if we already assign a backward flow neighbor for the backward flow neighbor
        #  If video_flow_neighbor[round(r'), round(c'), t-1] == 0,
        #   this is isolated pixel. Do nothing.
        #  If video_flow_neighbor[round(r'), round(c'), t-1] == 1,
        #   we can borrow the value and refine it.

        unknown_range = np.invert(known_range)

        # If we already assign a backward flow neighbor@[round(r'), round(c'), t-1]
        HaveNNInd = video_flow_neighbor[flow_neighbor_rows, flow_neighbor_cols, t - 1, nb_index] == 1

        # Unknown & consist_range & HaveNNInd
        Valid_ = np.logical_and.reduce((unknown_range, HaveNNInd, consist_range))

        refineVec = np.concatenate(
            (
                (backward_flow_position[:, 0] - flow_neighbor_rows).reshape(-1, 1),
                (backward_flow_position[:, 1] - flow_neighbor_cols).reshape(-1, 1),
                np.zeros((backward_flow_position[:, 0].shape[0])).reshape(-1, 1),
            ),
            1,
        )

        # pdb.set_trace()

        # Check if the transitive backward flow neighbor of [r, c, t] is known.
        # Sometimes after refinement, it is no longer known.
        flowNN_tmp = copy.deepcopy(
            mask_one_flow[
                video_mask_index[flow_neighbor_rows, flow_neighbor_cols, t - 1].astype(np.int32),
                :,
                nb_index,
            ]
            + refineVec[:, :]
        )
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        # Check out-of-boundary. flowNN_tmp may be out-of-boundary
        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0, flowNN_tmp[:, 0] <= image_height - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0, flowNN_tmp[:, 1] <= image_width - 1),
        )

        # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
        # in the next line. It won't affect anything as ValidPos_ is saved already
        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0], flowNN_tmp[:, 1], flowNN_tmp[:, 2]] == 0

        # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        # We save the transitive backward flow neighbor backward_flow_position in mask_one_flow
        mask_one_flow[
            video_mask_index[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], t].astype(np.int32),
            :,
            nb_index,
        ] = (
            mask_one_flow[
                video_mask_index[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    t - 1,
                ].astype(np.int32),
                :,
                nb_index,
            ]
            + refineVec[Valid, :]
        )

        # mark [r, c, t] in video_flow_neighbor as 1
        video_flow_neighbor[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], t, nb_index] = 1

        consistency_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], nb_index, 0, t] = np.maximum(
            np.abs(bf_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], 0]),
            np.abs(
                consistency_uv[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    nb_index,
                    0,
                    t - 1,
                ]
            ),
        )
        consistency_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], nb_index, 1, t] = np.maximum(
            np.abs(bf_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], 1]),
            np.abs(
                consistency_uv[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    nb_index,
                    1,
                    t - 1,
                ]
            ),
        )

        consistency_map[:, :, t, nb_index] = (
            consistency_uv[:, :, nb_index, 0, t] ** 2 + consistency_uv[:, :, nb_index, 1, t] ** 2
        ) ** 0.5

        print(
            "Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}".format(
                t,
                np.sum(video_flow_neighbor[:, :, t, nb_index] == 1),
                np.sum(video_flow_neighbor[:, :, t, nb_index] == 0),
                np.sum(video_flow_neighbor[:, :, t, nb_index] != 99999),
            )
        )

    # 2. Backward Pass (forward flow propagation)
    print("Backward Pass......")
    nb_index = 1  # FN:1
    for t in range(num_frame - 2, -1, -1):

        # Bool indicator of missing pixels at frame t
        hole_pixel_range = sub[:, 2] == t

        # Hole pixel location at frame t, i.e. [r, c, t]
        hole_pixel_position = sub[hole_pixel_range, :]
        hole_pixel_rows = hole_pixel_position[:, 0]
        hole_pixel_cols = hole_pixel_position[:, 1]

        # Calculate the forward flow neighbor. Should be located at frame t+1
        forward_flow_position = copy.deepcopy(hole_pixel_position).astype(np.float32)

        forward_flow_horizont = forward_flow[:, :, 0, t]
        forward_flow_vertical = forward_flow[:, :, 1, t]  # t --> t+1

        backward_flow_horizont = backward_flow[:, :, 0, t]
        backward_flow_vertical = backward_flow[:, :, 1, t]  # t+1 --> t

        forward_flow_position[:, 0] += forward_flow_vertical[hole_pixel_rows, hole_pixel_cols]
        forward_flow_position[:, 1] += forward_flow_horizont[hole_pixel_rows, hole_pixel_cols]
        forward_flow_position[:, 2] += 1 # frame = t + 1

        # Round the forward flow neighbor location
        flow_neighbor_position = np.round(copy.deepcopy(forward_flow_position)).astype(np.int32)

        # Check the forawrd/backward consistency
        consist_range = ForwardFlowConsistCheck(
            forward_flow_position,
            backward_flow_vertical,
            backward_flow_horizont,
            hole_pixel_position,
            args.consistencyThres,
        )

        fb_uv = ConsistCheck(backward_flow[:, :, :, t], forward_flow[:, :, :, t])

        # Check out-of-boundary
        # Last column and last row does not have valid gradient
        valid_neighbor_range = np.logical_and(
            np.logical_and(
                flow_neighbor_position[:, 0] >= 0,
                flow_neighbor_position[:, 0] <= image_height - 1,
            ),
            np.logical_and(
                flow_neighbor_position[:, 1] >= 0,
                flow_neighbor_position[:, 1] <= image_width - 1,
            ),
        )

        # Only work with pixels that are not out-of-boundary
        hole_pixel_position = hole_pixel_position[valid_neighbor_range, :]
        forward_flow_position = forward_flow_position[valid_neighbor_range, :]
        flow_neighbor_position = flow_neighbor_position[valid_neighbor_range, :]
        flow_neighbor_rows = flow_neighbor_position[:, 0]
        flow_neighbor_cols = flow_neighbor_position[:, 1]

        consist_range = consist_range[valid_neighbor_range]

        # Case 1:
        known_range = mask[flow_neighbor_rows, flow_neighbor_cols, t + 1] == 0

        known_consist_range = np.logical_and(known_range, consist_range)
        mask_one_flow[
            video_mask_index[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], t].astype(np.int32),
            :,
            nb_index,
        ] = forward_flow_position[known_consist_range, :]

        video_flow_neighbor[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            t,
            nb_index,
        ] = 1

        consistency_uv[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            nb_index,
            0,
            t,
        ] = np.abs(fb_uv[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], 0])
        consistency_uv[
            hole_pixel_position[known_consist_range, 0],
            hole_pixel_position[known_consist_range, 1],
            nb_index,
            1,
            t,
        ] = np.abs(fb_uv[hole_pixel_position[known_consist_range, 0], hole_pixel_position[known_consist_range, 1], 1])

        # Case 2:
        unknown_range = np.invert(known_range)
        HaveNNInd = video_flow_neighbor[flow_neighbor_rows, flow_neighbor_cols, t + 1, nb_index] == 1

        # Unknown & consist_range & HaveNNInd
        Valid_ = np.logical_and.reduce((unknown_range, HaveNNInd, consist_range))

        refineVec = np.concatenate(
            (
                (forward_flow_position[:, 0] - flow_neighbor_rows).reshape(-1, 1),
                (forward_flow_position[:, 1] - flow_neighbor_cols).reshape(-1, 1),
                np.zeros((forward_flow_position[:, 0].shape[0])).reshape(-1, 1),
            ),
            1,
        )

        # Check if the transitive backward flow neighbor of [r, c, t] is known.
        # Sometimes after refinement, it is no longer known.
        flowNN_tmp = copy.deepcopy(
            mask_one_flow[
                video_mask_index[flow_neighbor_rows, flow_neighbor_cols, t + 1].astype(np.int32),
                :,
                nb_index,
            ]
            + refineVec[:, :]
        )
        flowNN_tmp = np.round(flowNN_tmp).astype(np.int32)

        # Check out-of-boundary. flowNN_tmp may be out-of-boundary
        ValidPos_ = np.logical_and(
            np.logical_and(flowNN_tmp[:, 0] >= 0, flowNN_tmp[:, 0] <= image_height - 1),
            np.logical_and(flowNN_tmp[:, 1] >= 0, flowNN_tmp[:, 1] <= image_width - 1),
        )

        # Change the out-of-boundary value to 0, in order to run mask[y,x,t]
        # in the next line. It won't affect anything as ValidPos_ is saved already
        flowNN_tmp[np.invert(ValidPos_), :] = 0
        ValidNN = mask[flowNN_tmp[:, 0], flowNN_tmp[:, 1], flowNN_tmp[:, 2]] == 0

        # Valid = np.logical_and.reduce((Valid_, ValidNN, ValidPos_))
        Valid = np.logical_and.reduce((Valid_, ValidPos_))

        # We save the transitive backward flow neighbor backward_flow_position in mask_one_flow
        mask_one_flow[
            video_mask_index[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], t].astype(np.int32),
            :,
            nb_index,
        ] = (
            mask_one_flow[
                video_mask_index[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    t + 1,
                ].astype(np.int32),
                :,
                nb_index,
            ]
            + refineVec[Valid, :]
        )

        # mark [r, c, t] in video_flow_neighbor as 1
        video_flow_neighbor[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], t, nb_index] = 1

        consistency_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], nb_index, 0, t] = np.maximum(
            np.abs(fb_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], 0]),
            np.abs(
                consistency_uv[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    nb_index,
                    0,
                    t + 1,
                ]
            ),
        )
        consistency_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], nb_index, 1, t] = np.maximum(
            np.abs(fb_uv[hole_pixel_position[Valid, 0], hole_pixel_position[Valid, 1], 1]),
            np.abs(
                consistency_uv[
                    flow_neighbor_position[Valid, 0],
                    flow_neighbor_position[Valid, 1],
                    nb_index,
                    1,
                    t + 1,
                ]
            ),
        )

        consistency_map[:, :, t, nb_index] = (
            consistency_uv[:, :, nb_index, 0, t] ** 2 + consistency_uv[:, :, nb_index, 1, t] ** 2
        ) ** 0.5

        print(
            "Frame {0:3d}: {1:8d} + {2:8d} = {3:8d}".format(
                t,
                np.sum(video_flow_neighbor[:, :, t, nb_index] == 1),
                np.sum(video_flow_neighbor[:, :, t, nb_index] == 0),
                np.sum(video_flow_neighbor[:, :, t, nb_index] != 99999),
            )
        )

    # Interpolation
    videoBN = copy.deepcopy(video)
    videoFN = copy.deepcopy(video)

    for t in range(num_frame):
        # Index of missing pixel whose backward flow neighbor is from frame t
        SourceFmInd = np.where(mask_one_flow[:, 2, 0] == t)

        print("{0:8d} pixels are from source Frame {1:3d}".format(len(SourceFmInd[0]), t))
        # The location of the missing pixel whose backward flow neighbor is
        # from frame t mask_one_flow[SourceFmInd, 0, 0], mask_one_flow[SourceFmInd, 1, 0]

        if len(SourceFmInd[0]) != 0:

            # |--------------------|
            # |       y            |
            # |   x   *            |
            # |                    |
            # |--------------------|
            # sub: num_mask_one_pixel x 3 [r, c, t]
            # img: [r, c]
            # interp(img, c, r)

            videoBN[
                sub[SourceFmInd[0], :][:, 0],
                sub[SourceFmInd[0], :][:, 1],
                :,
                sub[SourceFmInd[0], :][:, 2],
            ] = interp(
                videoBN[:, :, :, t],
                mask_one_flow[SourceFmInd, 1, 0].reshape(-1),
                mask_one_flow[SourceFmInd, 0, 0].reshape(-1),
            )

            assert ((sub[SourceFmInd[0], :][:, 2] - t) <= 0).sum() == 0

    for t in range(num_frame - 1, -1, -1):
        # Index of missing pixel whose forward flow neighbor is from frame t
        SourceFmInd = np.where(mask_one_flow[:, 2, 1] == t)
        print("{0:8d} pixels are from source Frame {1:3d}".format(len(SourceFmInd[0]), t))
        if len(SourceFmInd[0]) != 0:
            videoFN[
                sub[SourceFmInd[0], :][:, 0],
                sub[SourceFmInd[0], :][:, 1],
                :,
                sub[SourceFmInd[0], :][:, 2],
            ] = interp(
                videoFN[:, :, :, t],
                mask_one_flow[SourceFmInd, 1, 1].reshape(-1),
                mask_one_flow[SourceFmInd, 0, 1].reshape(-1),
            )

            assert ((t - sub[SourceFmInd[0], :][:, 2]) <= 0).sum() == 0

    # New mask
    mask_tofill = np.zeros((image_height, image_width, num_frame)).astype(np.bool_)

    # pdb.set_trace()

    for t in range(num_frame):
        HaveNN = np.zeros((image_height, image_width, num_candidate))

        HaveNN[:, :, 0] = video_flow_neighbor[:, :, t, 0] == 1
        HaveNN[:, :, 1] = video_flow_neighbor[:, :, t, 1] == 1

        NotHaveNN = np.logical_and(
            np.invert(HaveNN.astype(np.bool_)),
            np.repeat(np.expand_dims((mask[:, :, t]), 2), num_candidate, axis=2),
        )

        HaveNN_sum = np.logical_or.reduce((HaveNN[:, :, 0], HaveNN[:, :, 1]))

        videoCandidate = np.zeros((image_height, image_width, 3, num_candidate))
        videoCandidate[:, :, :, 0] = videoBN[:, :, :, t]
        videoCandidate[:, :, :, 1] = videoFN[:, :, :, t]

        # args.alpha -- 0.1
        consistency_map[:, :, t, :] = np.exp(-consistency_map[:, :, t, :] / args.alpha)

        consistency_map[NotHaveNN[:, :, 0], t, 0] = 0
        consistency_map[NotHaveNN[:, :, 1], t, 1] = 0

        weights = (consistency_map[HaveNN_sum, t, :] * HaveNN[HaveNN_sum, :]) / (
            (consistency_map[HaveNN_sum, t, :] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True)
        )

        # Fix the numerical issue. 0 / 0
        fix = np.where((consistency_map[HaveNN_sum, t, :] * HaveNN[HaveNN_sum, :]).sum(axis=1, keepdims=True) == 0)[0]
        weights[fix, :] = HaveNN[HaveNN_sum, :][fix, :] / HaveNN[HaveNN_sum, :][fix, :].sum(axis=1, keepdims=True)

        # Fuse RGB channel independently
        video[HaveNN_sum, 0, t] = np.sum(np.multiply(videoCandidate[HaveNN_sum, 0, :], weights), axis=1)
        video[HaveNN_sum, 1, t] = np.sum(np.multiply(videoCandidate[HaveNN_sum, 1, :], weights), axis=1)
        video[HaveNN_sum, 2, t] = np.sum(np.multiply(videoCandidate[HaveNN_sum, 2, :], weights), axis=1)

        mask_tofill[np.logical_and(np.invert(HaveNN_sum), mask[:, :, t]), t] = True

    return video, mask_tofill
