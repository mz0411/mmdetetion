# model settings
_base_=['./cascade_rcnn_swin_fpn.py',]
model = dict(
    neck=dict(
        _delete_=True,
        type='PAFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        _delete_=True,
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.11, 0.11]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    train_cfg = dict(        
        rpn=dict(
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                # mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                # mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='CombinedSampler',
                    num=512,
                    pos_fraction=0.25,
                    add_gt_as_proposals=True,
                    pos_sampler=dict(type='InstanceBalancedPosSampler'),
                    neg_sampler=dict(
                        type='IoUBalancedNegSampler',
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3)),
                # mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
        test_cfg = dict(
            rpn=dict(
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_per_img=1000,
                # nms_pre=2000,
                # nms_post=2000,
                # max_per_img=2000,
                # nms=dict(type='nms', iou_threshold=0.85),
                # nms=dict(type='nms', iou_threshold=0.95),
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                # score_thr=0.005,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=1000)
                # mask_thr_binary=0.5)
                ))
