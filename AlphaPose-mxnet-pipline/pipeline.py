import sys
import os
import time
import tqdm
import threading

import mxnet as mx
import gluoncv as gcv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sppe.models.sefastpose import FastPose_SE
from pose_utils import pose_nms

from opt import opt
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    import queue
# otherwise, import the Queue class for Python 2.7
else:
    import Queue as queue


# box conf = 0.05: 17.3FPS -> 19.3FPS
# box conf = 0.10: 20.0FPS -> 22.3FPS
# box conf = 0.20: 22.1FPS -> 24.8FPS
# box conf = 0.30: 23.0FPS -> 25.9FPS
# box conf = 0.40: 24.0FPS -> 26.6FPS
# box conf = 0.50: 24.5FPS -> 27.0FPS

ctx = mx.gpu()



class PoseEstPip:
    def __init__(self, im_names):
        print('===Initializing===')
        ## info of the input image
        self.img_dir = opt.inputpath
        self.img_list = im_names
        self.data_len = len(self.img_list)
        for i in range(len(self.img_list)):
            self.img_list[i] = self.img_list[i].rstrip('\n').rstrip('\r')
            self.img_list[i] = os.path.join(opt.inputpath, self.img_list[i])

        ## Load Detector
        self.input_size = int(opt.inp_dim)
        # model config
        print('Loading yolo3_darknet53_coco ...')
        self.net_det = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
        self.net_det.set_nms(opt.nms_thresh, post_nms=-1)
        self.person_idx = self.net_det.classes.index('person')
        print('Modifying output layers to ignore non-person classes...')
        self.reset_class()
        self.net_det.collect_params().reset_ctx(ctx)
        self.net_det.hybridize()


        ## Load Pose Estimator
        # model config
        self.pose_batch_size = opt.posebatch
        print('Loading SPPE ...')
        self.net_pos = FastPose_SE(ctx)
        self.net_pos.load_parameters('sppe/params/duc_se.params')
        self.net_pos.hybridize()
        self.net_pos.collect_params().reset_ctx(ctx)

    def __len__(self):
        return len(self.img_list)

        # ImageLoader
    def get_batch_image(self):
        time_rec = []
        tensor_size = int(opt.inp_dim)
        tic = time.time()
        tensor_batch = []
        img_batch = []
        img_size_batch = []


        # laod images
        for k in range(len(self.img_list)):
            tensor_k, img_k, img_size_k = self.load_fn(self.img_list[k], tensor_size)
            tensor_batch.append(tensor_k)
            img_batch.append(img_k)
            img_size_batch.append(img_size_k)

        tensor_batch = mx.nd.concatenate(tensor_batch, axis=0)
        img_size_batch = mx.nd.array(img_size_batch, dtype='float32')
        img_size_batch = img_size_batch.tile(reps=[1, 2])

        toc = time.time()
        time_rec.append(toc - tic)

        print('ImageLoader: %fs' % (np.mean(time_rec)))

        return tensor_batch, img_batch, img_size_batch # the whole image for process


    def load_fn(self, img_name, tensor_size):
        '''
        Load single image from the given file
        INPUT:
            img_name: string, image file name
            tensor_size: int, image size after resize
        OUTPUT:
            tensor: mx.nd, input tensor for detection
            img: mx.nd, original image in nd type
            img_size: (int, int), original image size
        '''
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        img = mx.image.imread(img_name)
        img_size = (img.shape[1], img.shape[0])

        # resize image
        tensor = gcv.data.transforms.image.resize_long(img, tensor_size, interp=9)
        tensor = mx.nd.image.to_tensor(tensor)
        tensor = mx.nd.image.normalize(tensor, mean=mean, std=std)
        tensor = tensor.expand_dims(0)

        # pad tensor
        pad_h = tensor_size - tensor.shape[2]
        pad_w = tensor_size - tensor.shape[3]
        pad_shape = (0, 0, 0, 0, 0, pad_h, 0, pad_w)
        tensor = mx.nd.pad(tensor, mode='constant',
                           constant_value=0.5, pad_width=pad_shape)

        return tensor, img, img_size



        # Detector

    def detect_fn(self, tensor_batch, img_size_batch):
        time_rec = []

        tic = time.time()

        # get prediction
        tensor_batch= mx.nd.array(tensor_batch.asnumpy()[np.newaxis,:])# add a new axis for the format of batch
        class_idxs, scores, boxes = self.net_det(tensor_batch.copyto(ctx))
        class_idxs = class_idxs.copyto(mx.cpu())
        scores = scores.copyto(mx.cpu())
        boxes = boxes.copyto(mx.cpu())

        toc = time.time()
        time_rec.append(toc - tic)

        print('Detector: %fs' % (np.mean(time_rec)))
        return boxes[0], class_idxs[0, :, 0], scores[0, :, 0], img_size_batch


    def reset_class(self):
        '''Modify output layers to ignore non-person classes'''
        self.net_det._clear_cached_op()
        len_per_anchor = 5 + len(self.net_det.classes)

        for output in self.net_det.yolo_outputs:
            num_anchors = output._num_anchors
            picked_channels = np.array(list(range(len_per_anchor)) * num_anchors)
            picked_channels = np.where((picked_channels < 5) |
                                       (picked_channels == 5 + self.person_idx))

            parameters = output.prediction.params
            for k in parameters:
                if 'weight' in k:
                    key_weight = k
                    init_weight = parameters[k].data()[picked_channels]
                    in_channels = parameters[k].data().shape[1]
                elif 'bias' in k:
                    key_bias = k
                    init_bias = parameters[k].data()[picked_channels]

            output.prediction = mx.gluon.nn.Conv2D(6 * num_anchors,
                                                   in_channels=in_channels,
                                                   kernel_size=1,
                                                   padding=0,
                                                   strides=1,
                                                   prefix=output.prediction.prefix)
            output.prediction.collect_params().initialize()
            output.prediction.params[key_weight].set_data(init_weight)
            output.prediction.params[key_bias].set_data(init_bias)
            output._classes = 1
            output._num_pred = 6
        self.net_det._classes = ['person']

        # DetectionProcessor

    def transform_fn(self, boxes, class_idxs, scores, img_size ):
        time_rec = []

        tic = time.time()

        # rescale coordinates
        scaling_factor = mx.nd.min(self.input_size / img_size)
        boxes /= scaling_factor

        # cilp coordinates
        boxes[:, [0, 2]] = mx.nd.clip(boxes[:, [0, 2]], 0., img_size[0].asscalar() - 1)
        boxes[:, [1, 3]] = mx.nd.clip(boxes[:, [1, 3]], 0., img_size[1].asscalar() - 1)

        # select boxes
        mask1 = (class_idxs == self.person_idx).asnumpy()
        mask2 = (scores > opt.confidence).asnumpy()
        picked_idxs = np.where((mask1 + mask2) > 1)[0]

        toc = time.time()
        time_rec.append(toc - tic)
        print('DetectionProcessor: %fs' % (np.mean(time_rec)))
        # put into queue
        if picked_idxs.shape[0] == 0:
            return None, None
        else:
            return boxes[picked_idxs], scores[picked_idxs]



    # ImageCropper
    '''Crop persons from original images'''

    def crop_process(self, img, boxes, scores):
        time_rec = []

        tic = time.time()

        if boxes is None:
            return None, img, None, None, None, None
        else:
            # crop person poses
            tensors, pt1, pt2 = self.crop_fn(img, boxes)

            toc = time.time()
            time_rec.append(toc - tic)
            print('ImageCropper: %fs' % (np.mean(time_rec)))
            # put into queue
            return tensors, img, boxes, scores, pt1, pt2



    def crop_fn(self, img, boxes):
        '''
        Crop persons based on given boxes
        INPUT:
            img: mx.nd, original image
            boxes: mx.nd, image size after resize
        OUTPUT:
            tensors: mx.nd, input tensor for pose estimation
            pt1: mx.nd, coordinates of left upper box corners
            pt2: mx.nd, coordinates of right bottom box corners
        '''
        mean = (0.485, 0.456, 0.406)
        std = (1.0, 1.0, 1.0)
        img_width, img_height = img.shape[1], img.shape[0]

        tensors = mx.nd.zeros([boxes.shape[0], 3, opt.inputResH, opt.inputResW])
        pt1 = mx.nd.zeros([boxes.shape[0], 2])
        pt2 = mx.nd.zeros([boxes.shape[0], 2])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        img = img.transpose(axes=[1, 2, 0])

        for i, box in enumerate(boxes.asnumpy()):
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            if box_width > 100:
                scale_rate = 0.2
            else:
                scale_rate = 0.3

            # crop image
            left = int(max(0, box[0] - box_width * scale_rate / 2))
            up = int(max(0, box[1] - box_height * scale_rate / 2))
            right = int(min(img_width - 1,
                            max(left + 5, box[2] + box_width * scale_rate / 2)))
            bottom = int(min(img_height - 1,
                             max(up + 5, box[3] + box_height * scale_rate / 2)))
            crop_width = right - left
            crop_height = bottom - up
            cropped_img = mx.image.fixed_crop(img, left, up, crop_width, crop_height)

            # resize image
            resize_factor = min(opt.inputResW / crop_width, opt.inputResH / crop_height)
            new_width = int(crop_width * resize_factor)
            new_height = int(crop_height * resize_factor)
            tensor = mx.image.imresize(cropped_img, new_width, new_height)
            tensor = tensor.transpose(axes=[2, 0, 1])
            tensor = tensor.reshape(1, 3, new_height, new_width)

            # pad tensor
            pad_h = opt.inputResH - new_height
            pad_w = opt.inputResW - new_width
            pad_shape = (0, 0, 0, 0, pad_h // 2, (pad_h + 1) // 2, pad_w // 2, (pad_w + 1) // 2)
            tensor = mx.nd.pad(tensor, mode='constant',
                               constant_value=0.5, pad_width=pad_shape)
            tensors[i] = tensor.reshape(3, opt.inputResH, opt.inputResW)
            pt1[i] = (left, up)
            pt2[i] = (right, bottom)

        return tensors, pt1, pt2


    # PoseEstimator
    '''Estimate person poses and transform pose coordinates'''

    def estimate_fn(self, tensors, boxes, box_scores, pt1, pt2):
        time_rec = []


        tic = time.time()

        if tensors is None:
            return None, None, None, None
        else:
            heatmaps = []
            num_poses = tensors.shape[0]
            num_batches = (num_poses + self.pose_batch_size - 1) // self.pose_batch_size
            # this batch size is the pose batch size default = 80, in our situation the num batches is always 1

            for k in range(num_batches):
                # get batch tensor
                begin_idx = k * self.pose_batch_size
                end_idx = min(begin_idx + self.pose_batch_size, num_poses)
                tensor_batch = tensors[begin_idx:end_idx]
                # get prediction
                heatmap_batch = self.net_pos(tensor_batch.copyto(ctx))
                heatmap_batch = heatmap_batch[:, :17, :, :]
                heatmaps.append(heatmap_batch.copyto(mx.cpu()))

                # coordinate transformation
                heatmaps = mx.nd.concatenate(heatmaps, axis=0)
                pose_hms, pose_coords, pose_scores = self.estimate_transform_fn(heatmaps, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

                toc = time.time()
                time_rec.append(toc - tic)
                print('PoseEstimator: %fs' % (np.mean(time_rec)))
                # put into queue
                return boxes, box_scores, pose_coords, pose_scores





    def estimate_transform_fn(self, hms, pt1, pt2, inp_h, inp_w, res_h, res_w):
        '''
        Transform pose heatmaps to coordinates
        INPUT:
            hms: mx.nd, pose heatmaps
            pt1: mx.nd, coordinates of left upper box corners
            pt2: mx.nd, coordinates of right bottom box corners
            inp_h: int, input tensor height
            inp_w: int, input tensot width
            res_h: int, output heatmap height
            res_w: int, output heatmap width
        OUTPUT:
            preds: mx.nd, pose coordinates in box frames
            preds_tf: mx.nd, pose coordinates in image frames
            maxval: mx.nd, pose scores
        '''
        pt1 = pt1.expand_dims(axis=1)
        pt2 = pt2.expand_dims(axis=1)

        # get keypoint coordinates
        idxs = mx.nd.argmax(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
        maxval = mx.nd.max(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
        preds = idxs.tile(reps=[1, 1, 2])
        preds[:, :, 0] %= hms.shape[3]
        preds[:, :, 1] /= hms.shape[3]

        # get pred masks
        pred_mask = (maxval > 0).tile(reps=[1, 1, 2])
        preds *= pred_mask

        # coordinate transformation
        box_size = pt2 - pt1
        len_h = mx.nd.maximum(box_size[:, :, 1:2], box_size[:, :, 0:1] * inp_h / inp_w)
        len_w = len_h * inp_w / inp_h
        canvas_size = mx.nd.concatenate([len_w, len_h], axis=2)
        offsets = pt1 - mx.nd.maximum(0, canvas_size / 2 - box_size / 2)
        preds_tf = preds * len_h / res_h + offsets

        return preds, preds_tf, maxval


    # PoseProcessor
    '''Pose NMS'''


    def PoseProcessor(self, img, boxes, box_scores, pose_coords, pose_scores, img_name):
        time_rec = []
        # im_names_desc = tqdm.tqdm(range(self.data_len))

        tic = time.time()

        if boxes is None:
            print('No pose!')
        else:
            # pose nms
            final_result, boxes, box_scores = pose_nms(boxes.asnumpy(),
                                                       box_scores.asnumpy(),
                                                       pose_coords.asnumpy(), pose_scores.asnumpy())

            toc = time.time()
            time_rec.append(toc - tic)

            # put into queue
            if opt.save_img:
                gcv.utils.viz.plot_bbox(img, boxes, box_scores, thresh=0.1)
                plt.xlim([0, img.shape[1] - 1])
                plt.ylim([0, img.shape[0] - 1])
                plt.gca().invert_yaxis()
                for result in final_result:
                    pts = result['keypoints']
                    mask = (result['kp_score'][:, 0] > 0.1)
                    plt.scatter(pts[:, 0][mask], pts[:, 1][mask], s=20)
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(os.path.join('examples/res', img_name.split('/')[-1]))

            #self.Q.put((img, final_result, boxes, box_scores, img_name))
        print('PoseProcessor: %fs' % (np.mean(time_rec)))
