import os
import os.path as osp
from opt import opt
from pipeline import PoseEstPip
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# os.environ['MXNET_CPU_WORKER_NTHREADS'] = '2'

# output path
vis_dir = 'examples/res'
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

# input image path
img_list = os.listdir('./examples/demo')
img_list = [osp.join('./examples/demo', item.strip('\n')) for item in img_list]


PoseEstPip = PoseEstPip(img_list)
# load image
tensor_batch, img_batch, img_size_batch = PoseEstPip.get_batch_image()
for i in range(PoseEstPip.__len__()):
    print('Processing image %s' %(PoseEstPip.img_list[i]))
    # Detector
    boxes, class_idxs, scores, img_size = PoseEstPip.detect_fn(tensor_batch[i], img_size_batch[i])
    # DetectionProcessor
    boxes, scores = PoseEstPip.transform_fn(boxes, class_idxs, scores, img_size)
    # ImageCropper
    tensors, img, boxes, box_scores, pt1, pt2 = PoseEstPip.crop_process(img_batch[i], boxes, scores)
    # PoseEstimator
    boxes, box_scores, pose_coords, pose_scores = PoseEstPip.estimate_fn(tensors, boxes, box_scores, pt1, pt2)

    ## 1 show keypoint
    # Pose NMS
    final_result = PoseEstPip.PoseProcessor(img, boxes, box_scores, pose_coords, pose_scores, PoseEstPip.img_list[i])

    ## 2 show frame
    # link frame

    result = {
        'imgname': PoseEstPip.img_list[i],
        'result': final_result
    }
    # draw frame
    PoseEstPip.vis_frame(img, result)







