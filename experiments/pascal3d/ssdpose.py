#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import caffe
from caffe.model_libs import *
from google.protobuf import text_format

import math
import os
import shutil
import stat
import argparse
import subprocess
import os.path as osp

caffe_root = os.environ['CAFFEROOT']

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 512, 3, 1, 2)

    for i in xrange(7, 9):
      from_layer = out_layer
      out_layer = "conv{}_1".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1)

      from_layer = out_layer
      out_layer = "conv{}_2".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    # Add global pooling layer.
    name = net.keys()[-1]
    net.pool6 = L.Pooling(net[name], pool=P.Pooling.AVE, global_pooling=True)

    return net

def makeBatchSampler():
    batch_sampler = [
    {
            'sampler': {
                    },
            'max_trials': 1,
            'max_sample': 1,
    },
    {
            'sampler': {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    },
            'sample_constraint': {
                    'min_jaccard_overlap': 0.7,
                    },
            'max_trials': 50,
            'max_sample': 1,
    },
    {
            'sampler': {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    },
            'sample_constraint': {
                    'min_jaccard_overlap': 0.9,
                    },
            'max_trials': 50,
            'max_sample': 1,
    },
    {
            'sampler': {
                    'min_scale': 0.3,
                    'max_scale': 1.0,
                    'min_aspect_ratio': 0.5,
                    'max_aspect_ratio': 2.0,
                    },
            'sample_constraint': {
                    'max_jaccard_overlap': 1.0,
                    },
            'max_trials': 50,
            'max_sample': 1,
    }
    ]
    return batch_sampler


def run_main():
    share_pose = True
    num_classes = 12
    num_poses = 8
    pose_weight = 1.0
    max_iter = 30000
    num_test = 2969
    step_size = 11000

    run_soon = False
    resume_training = True

    # Solver parameters.
    gpus = "0"
    gpulist = gpus.split(",")
    num_gpus = len(gpulist)

    # The database file for training data. Created by data/VOC0712/create_data.sh
    train_data = 'data/pascal3d/lmdb/train_lmdb/'
    val_data = 'data/pascal3d/lmdb/val_lmdb/'
    test_data = 'data/pascal3d/lmdb/test_lmdb/'

    # Specify the batch sampler.
    resize_width = 300
    resize_height = 300
    resize = "{}x{}".format(resize_width, resize_height)
    batch_sampler = makeBatchSampler()

    train_transform_param = {
            'mirror': True,
            'mean_value': [104, 117, 123],
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                }
            }
    test_transform_param = {
            'mean_value': [104, 117, 123],
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    'interp_mode': [P.Resize.LINEAR],
                    },
            }

    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    use_batchnorm = False
    # Use different initial learning rate.
    if use_batchnorm:
        base_lr = 0.00004 * 1000
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = 0.00004

    # Modify the job name if you want.
    if share_pose:
        job_name = "SSD_share_pose"
    else:
        #job_name = "SSD_seperate_pose_{}_{}_bins_{}".format(resize, num_poses, idx)
        job_name = "SSD_seperate_pose"
    # The name of the model. Modify it if you want.
    model_name = "VGG_Pascal3D_{}".format(job_name)

    save_dir = "models/VGGNet/Pascal3D/{}".format(job_name)
    snapshot_dir = "models/VGGNet/Pascal3D/{}".format(job_name)
    job_dir = "jobs/VGGNet/Pascal3D/{}".format(job_name)
    output_result_dir = "data/pascal3d/results/{}/Main".format(job_name)

    # model definition files.
    train_net_file = "{}/train.prototxt".format(save_dir)
    val_net_file = "{}/val.prototxt".format(save_dir)
    test_net_file = "{}/test.prototxt".format(save_dir)
    deploy_net_file = "{}/deploy.prototxt".format(save_dir)
    train_solver_file = "{}/solver.prototxt".format(save_dir)
    test_solver_file = "{}/test_solver.prototxt".format(save_dir)
    snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
    job_file = "{}/{}.sh".format(job_dir, model_name)

    # The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
    pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
    # Stores LabelMapItem.
    label_map_file = "data/pascal3d/labelmap_3D.prototxt"

    #
    # MultiBoxLoss parameters.
    #
    share_location = True
    background_label_id = 0
    train_on_diff_gt = True
    normalization_mode = P.Loss.VALID
    code_type = P.PriorBox.CENTER_SIZE
    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    multibox_pose_loss_param = {
        'loc_loss_type': caffe_pb2.LocLossType.Value('LocLossType_SMOOTH_L1'),
        'conf_loss_type':caffe_pb2.ConfLossType.Value('ConfLossType_SOFTMAX'),
        'loc_weight': loc_weight,
        'pose_weight': pose_weight,
        'num_classes': num_classes,
        'num_poses': num_poses,
        'share_location': share_location,
        'share_pose': share_pose,
        'match_type':caffe_pb2.MatchType.Value('MatchType_PER_PREDICTION'),
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': background_label_id,
        'use_difficult_gt': train_on_diff_gt,
        'do_neg_mining': True,
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': code_type,
        }
    loss_param = {
        'normalization': normalization_mode,
        }

    # parameters for generating priors.
    # minimum dimension of input image
    # TODO ASK WEI
    min_dim = 300
    # conv4_3 ==> 38 x 38
    # fc7 ==> 19 x 19
    # conv6_2 ==> 10 x 10
    # conv7_2 ==> 5 x 5
    # conv8_2 ==> 3 x 3
    # pool6 ==> 1 x 1
    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'pool6']
    # in percent %
    min_ratio = 20
    max_ratio = 95
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
      min_sizes.append(min_dim * ratio / 100.)
      max_sizes.append(min_dim * (ratio + step) / 100.)
    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [[]] + max_sizes
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]
    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
      prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
      prior_variance = [0.1]
    flip = True
    clip = True

    # Divide the mini-batch to different GPUs.
    batch_size = 32
    accum_batch_size = 32
    iter_size = accum_batch_size / batch_size
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = batch_size
    if num_gpus > 0:
      batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
      iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
      solver_mode = P.Solver.GPU
      device_id = int(gpulist[0])

    if normalization_mode == P.Loss.BATCH_SIZE:
      base_lr /= iter_size
    elif normalization_mode == P.Loss.NONE:
      base_lr /= batch_size_per_device * iter_size
    elif normalization_mode == P.Loss.VALID:
      base_lr *= 25. / loc_weight / iter_size
    elif normalization_mode == P.Loss.FULL:
      # Roughly there are 2000 prior bboxes per image.
      # TODO(weiliu89): Estimate the exact # of priors.
      base_lr *= 2000. / iter_size

    # Which layers to freeze (no backward) during training.
    freeze_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2']

    # Evaluate on whole test set.
    #num_test_image = args.get_opts('num_val')
    num_test_image = num_test
    test_batch_size = 1
    test_iter = num_test_image / test_batch_size

    train_solver_param = {
        # Train parameters
        'base_lr': base_lr,
        'weight_decay': 0.0005,
        'lr_policy': "step",
        'stepsize': step_size,
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': iter_size,
        'max_iter': max_iter,
        'snapshot': 2000,
        'display': 10,
        'average_loss': 10,
        'type': "SGD",
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': [500],
        'test_interval': 1000,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': False,
        }
    test_solver_param = {
        # Train parameters
        'base_lr': base_lr,
        'weight_decay': 0.0005,
        'lr_policy': "step",
        'stepsize': step_size,
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': iter_size,
        'max_iter': max_iter,
        'snapshot': 2000,
        'display': 10,
        'average_loss': 10,
        'type': "SGD",
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        # Test parameters
        'test_iter': [num_test],
        'test_interval': 1,
        'eval_type': "detection",
        'ap_version': "11point",
        'test_initialization': False,
        }
    # parameters for generating detection output.
    det_out_param = {
        'num_classes': num_classes,
        'num_poses': num_poses,
        'share_location': share_location,
        'share_pose': share_pose,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
        'save_output_param': {
            'output_directory': output_result_dir,
            'output_name_prefix': "comp4_det_test_",
            'output_format': "VOC",
            'label_map_file': label_map_file,
            'num_test_image': num_test_image,
            },
        'keep_top_k': 200,
        'confidence_threshold': 0.01,
        'code_type': code_type,
        }

    # parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': num_classes,
        'num_poses': num_poses,
        'background_label_id': background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        }

    ### Hopefully you don't need to change the following ###
    # Check file.
    check_if_exist(train_data)
    check_if_exist(val_data)
    check_if_exist(test_data)
    check_if_exist(label_map_file)
    check_if_exist(pretrain_model)
    make_if_not_exist(save_dir)
    make_if_not_exist(job_dir)
    make_if_not_exist(snapshot_dir)

    #
    # Create train net.
    #
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
            train=True, output_label=True, label_map_file=label_map_file,
            transform_param=train_transform_param, batch_sampler=batch_sampler)
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
    AddExtraLayers(net, use_batchnorm)
    mbox_layers = CreateMultiBoxPoseHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations,
            num_classes=num_classes, num_poses=num_poses, 
            share_location=share_location, share_pose=share_pose,
            flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1)

    #
    # Create the MultiBoxLossLayer.
    #
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxPoseLoss(*mbox_layers, multibox_pose_loss_param=multibox_pose_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, True, False, False])

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    print('write train_net_file: {}'.format(train_net_file))

    #
    # Create val net.
    #
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(val_data, batch_size=test_batch_size,
            train=False, output_label=True, label_map_file=label_map_file,
            transform_param=test_transform_param)
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
    AddExtraLayers(net, use_batchnorm)
    mbox_layers = CreateMultiBoxPoseHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations,
            num_classes=num_classes, num_poses=num_poses, 
            share_location=share_location, share_pose=share_pose,
            flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1)
    conf_name = "mbox_conf"
    if multibox_pose_loss_param["conf_loss_type"] == caffe_pb2.ConfLossType.Value('ConfLossType_SOFTMAX'):
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_pose_loss_param["conf_loss_type"] == caffe_pb2.ConfLossType.Value('ConfLossType_LOGISTIC'):
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]

    # Only consider Softmax right now 
    pose_name = "mbox_pose"
    reshape_name = "{}_reshape".format(pose_name)
    net[reshape_name] = L.Reshape(net[pose_name], shape=dict(dim=[0, -1, num_poses]))
    softmax_name = "{}_softmax".format(pose_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(pose_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[2] = net[flatten_name]

    net.detection_out = L.DetectionPoseOutput(*mbox_layers,
        detection_pose_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluatePose(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(val_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    print('write val_net_file: {}'.format(val_net_file))

    #
    # Create test net.
    #
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
            train=False, output_label=True, label_map_file=label_map_file,
            transform_param=test_transform_param)
    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False, freeze_layers=freeze_layers)
    AddExtraLayers(net, use_batchnorm)
    mbox_layers = CreateMultiBoxPoseHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations,
            num_classes=num_classes, num_poses=num_poses, 
            share_location=share_location, share_pose=share_pose,
            flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1)
    conf_name = "mbox_conf"
    if multibox_pose_loss_param["conf_loss_type"] == caffe_pb2.ConfLossType.Value('ConfLossType_SOFTMAX'):
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_pose_loss_param["conf_loss_type"] == caffe_pb2.ConfLossType.Value('ConfLossType_LOGISTIC'):
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]
    # Only consider Softmax right now 
    pose_name = "mbox_pose"
    reshape_name = "{}_reshape".format(pose_name)
    net[reshape_name] = L.Reshape(net[pose_name], shape=dict(dim=[0, -1, num_poses]))
    softmax_name = "{}_softmax".format(pose_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(pose_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[2] = net[flatten_name]
    net.detection_out = L.DetectionPoseOutput(*mbox_layers,
        detection_pose_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluatePose(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)
    print('write test_net_file: {}'.format(test_net_file))

    #
    # Create deploy net.
    #
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
        print(net_param, file=f)
    print('write deploy_net_file: {}'.format(deploy_net_file))

    #
    # Create solver.
    #
    train_solver = caffe_pb2.SolverParameter(
            train_net=train_net_file,
            test_net=[val_net_file],
            snapshot_prefix=snapshot_prefix,
            **train_solver_param)
    test_solver = caffe_pb2.SolverParameter(
            train_net=train_net_file,
            test_net=[test_net_file],
            snapshot_prefix=snapshot_prefix,
            **test_solver_param)
    with open(train_solver_file, 'w') as f:
        print(train_solver, file=f)
    print('write train_solver_file: {}'.format(train_solver_file))
    with open(test_solver_file, 'w') as f:
        print(test_solver, file=f)
    print('write test_solver_file: {}'.format(test_solver_file))

    _max_iter = 0
    # Find most recent snapshot.
    for file in os.listdir(snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        it = int(basename.split("{}_iter_".format(model_name))[1])
        if it > _max_iter:
          _max_iter = it
    train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
    if resume_training:
      if _max_iter > 0:
        train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, _max_iter)

    #
    # Create job file.
    #
    log_file = '{}/{}.log'.format(job_dir, model_name)
    if osp.isfile(log_file):
        opt = '-a'
    else:
        opt = ''
    with open(job_file, 'w') as f:
      f.write('$CAFFEROOT/build/tools/caffe train \\\n')
      f.write('\t--solver="{}" \\\n'.format(train_solver_file))
      f.write('\t'+train_src_param)
      if train_solver_param['solver_mode'] == P.Solver.GPU:
        f.write('\t--gpu {} 2>&1 | tee {} {}\n'.format(gpus, opt, log_file))
      else:
        f.write('\t2>&1 | tee {}\n'.format(log_file))
    print('write job_file: {}'.format(job_file))

    # Copy the python script to job_dir.
    py_file = os.path.abspath(__file__)
    shutil.copy(py_file, job_dir)

    # Run the job.
    os.chmod(job_file, stat.S_IRWXU)
    if run_soon:
      subprocess.call(job_file, shell=True)

if __name__ == "__main__":
    run_main()
