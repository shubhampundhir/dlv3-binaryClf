import sys 
from torch.utils.data import dataset
from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np
import pickle 
import cv2

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])

    # Train Options

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=256)

    parser.add_argument("--ckpt", default='/home/shubhamp/Downloads/binarymodified_test_freezebackbone/checkpoints_binaryClf15k/CITY_768x768/best_deeplabv3plus_resnet101_cityscapes_os8.pth', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def read_video_frames(video_path,sample_interval=10):
    # Make the directory to save the frames.
    video_name =os.path.basename(video_path).split('.mp4')[0]
    os.makedirs(video_name,exist_ok=True)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * sample_interval)
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        # If there are no more frames, break the loop
        if not ret:
            break
        # Append the frame to the list if it's the desired interval
        if frame_count % interval_frames == 0:
            frames.append(frame)
            cv2.imwrite(os.path.join(video_name,str(frame_count)+'.jpg'),frame)
        frame_count += 1
    # Release the video capture object
    cap.release()
    return frames

def binaryClf_prediction_emarg(img,img_path=None):
    opts = get_argparser().parse_args()
    opts.num_classes = 2
    # opts.ckpt = 'shubamclf.pth'
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: %s" % device)
    
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=2, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)
    
    if os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    if opts.crop_val:
        transform = T.Compose([
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    else:
        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
  
    with torch.no_grad():
        model = model.eval()
        if img_path is not None:
            img_canvas = cv2.imread(img_path)
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            print(img_name)
            img_ = cv2.imread(img_path)
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.fromarray(img)
            img = img.resize((256, 256))

        img = transform(img).unsqueeze(0) # To tensor of NCHW
        img = img.to(device)

        # Getting the pred 
        pred_= torch.sigmoid(model(img)).cpu().numpy()[0]      
        output = pred_[0]
        logit = np.round(pred_[0],3)
        THRESHOLD = 0.4
        # Adding the tint and saving it to another folder 
        if output > THRESHOLD:
            output = 1 
        else:
            output = 0 
        
        # tint_color = (0, 0, 255) if output == 0 else (0, 255, 0) 
        # tinted_frame = cv2.addWeighted(img_canvas, 0.75, np.full_like(img_canvas, tint_color), 0.25, 0)
      
    # return pred_,colorized_preds
    # return output,tinted_frame
    
    return [output,max(logit,1-logit)]

def binaryClf_prediction_emarg_images_list(image_list):
    # Iterate over the images and get the list of scores 
    results = []
    for i,frame in enumerate(image_list):
        res = binaryClf_prediction_emarg(frame,None)
        results.append(res)
    return results

def images_to_video(input_folder, output_video, fps):
    image_files = [img for img in os.listdir(input_folder) if img.endswith(".jpg")]
    # image_files.sort()  # Ensure the images are in the correct order
    image_files.sort(key=lambda x: os.path.getmtime(os.path.join(input_folder, x)))

    # Get the first image to determine the size
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, layers = first_image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_file in image_files:
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


''' for binaryClf_prediction_emarg function'''
# if __name__ == '__main__':
#     videoPath = "/home/shubhamp/Downloads/binarymodified_test_freezebackbone/video_1.mp4"
#     read_video_frames(videoPath, sample_interval=10)
#     folderPath = '/home/shubhamp/Downloads/binarymodified_test_freezebackbone/video_1'
#     os.makedirs('./video_1new', exist_ok=True)
#     outFolder = "/home/shubhamp/Downloads/binarymodified_test_freezebackbone/video_1new"
#     os.makedirs(outFolder, exist_ok=True)

#     # video_fps = get_video_fps("/home/shubhamp/Downloads/binarymodified_test_freezebackbone/video_1.mp4")
#     # print("Original Video FPS:", video_fps)

#     image_files = [img for img in os.listdir(folderPath) if img.endswith(".jpg")]
#     image_files.sort(key=lambda x: os.path.getmtime(os.path.join(folderPath, x)))

#     for file in image_files:
#         img_path = os.path.join(folderPath, file)
#         ext = os.path.basename(img_path).split('.')[-1]
#         img_name = os.path.basename(img_path)[:-len(ext)-1]
#         out, tinted_frame = binaryClf_prediction_emarg(img=None, img_path=img_path)
#         print('{} = {}'.format(img_name, out))
#         output_path = os.path.join(outFolder, img_name + '.jpg')
#         print("Saving to:", output_path)
#         cv2.imwrite(output_path, tinted_frame)

#     output_video = '/home/shubhamp/Downloads/binarymodified_test_freezebackbone/OUT1_' + os.path.basename(videoPath)
#     fps = 5
#     images_to_video('/home/shubhamp/Downloads/binarymodified_test_freezebackbone/video_1new', output_video, fps)
#     print('~Completed')

