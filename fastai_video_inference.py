'''
FASTAI MORD Model - Video Based Inference
'''

import sys
import os
import cv2
import csv 
import numpy as np 
import argparse
from fastai import *
from fastai.vision import *
from fastai.vision.all import *

# IIIT Model Import
from video_inference import binaryClf_prediction_emarg_images_list

def get_argparser():
    parser = argparse.ArgumentParser()
    # Inference Options
    parser.add_argument("--modelName", type=str,default='MoRD')
    parser.add_argument("--videoName", type=str,default='/home/shubhamp/Downloads/binarymodified_test_freezebackbone/selected_frames_video_og.mp4')
    parser.add_argument("--checkpoint", type=str,default='/home/shubhamp/Downloads/binarymodified_test_freezebackbone/S-SRI-U_imagenet_train-416-final.pkl')
    parser.add_argument("--imageFolderPath", type=str,default=None)
    parser.add_argument("--csvFileName", type=str,default=None )
    parser.add_argument("--inputFPS", type=int,default=1)
    parser.add_argument("--outputFPS", type=int,default=1)
    args = parser.parse_args()
    return args

def mord_model_inference_images_list(args, image_files):
    # Load the model
    try:
        if os.path.exists(args.checkpoint):
            mord_model = load_learner(args.checkpoint)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    except Exception as ex:
        print(f"Error in FastAI Model Loading: {ex}")
        return []

    test_dl = mord_model.dls.test_dl(image_files)
    preds, _ = mord_model.get_preds(dl=test_dl)
    class_preds, _, decoded = mord_model.get_preds(dl=test_dl, with_decoded=True)
    decoded = np.asarray(decoded).tolist()
    preds = np.array(preds).tolist()
    result_dictionary = [list(t) for t in zip(decoded, preds)]
    return result_dictionary

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


def write_video(image_files, output_video_name,output_fps):
    # Get the first image to determine the size
    first_image = image_files[0]
    height, width, layers = first_image.shape
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video = cv2.VideoWriter(output_video_name, fourcc, output_fps, (width, height))
    for image_file in image_files:
        video.write(image_file)
    cv2.destroyAllWindows()
    video.release()

def annotate_frame(image,modelName,content,top_left=[0,0],box_length=20,box_width=160,tintColor=False):
    canvas = image.copy()
    h,w,_ = image.shape
    # Content 
    output,max_score = content
    box_color = (0, 0, 255) if output == 0 else (0, 255, 0)

    # Draw Rectangular Box.
    # top_left = (text_start[0]+np.int32(box_length/2),text_start[1]+ np.int32(box_length/2))  # Example coordinates, adjust as needed
    text_start = (5+top_left[0],top_left[1]+np.int32(box_length/2)+5)
    bottom_right = (top_left[0] + box_width, top_left[1] + box_length)
    canvas=cv2.rectangle(canvas, top_left ,bottom_right , box_color, thickness=cv2.FILLED)

    # Write text.
    # Font Selection.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    font_scale = 1
    font_line_type = cv2.LINE_AA

    # Write inside the box.
    verdict = 'GOOD' if output == 1 else 'POOR'
    text_content = "{} {} {}".format(str(modelName),str(max_score),verdict)
    print('TEXT : {}'.format(text_content))
    canvas = cv2.putText(canvas,text_content,text_start,font,0.5,(255,0,0),thickness=font_thickness) 

    # # If tinted is required.
    # if output is not None and tintColor is True:  
    #     tint_color = (0, 0, 255) if output == 0 else (0, 255, 0) 
    #     canvas= cv2.addWeighted(canvas, 0.75, np.full_like(canvas, tint_color), 0.25, 0)        
    
    return canvas


'''
You need to provide scores_dict
to this function.
'''
def video_inference(args):
    # Check if video exists , and create a list of images based on input fps.
    try:
        if os.path.exists(args.videoName):
            video_frames = read_video_frames(args.videoName,sample_interval=args.inputFPS)
            # video_frames = [video_frames[0],video_frames[1]]
            print(len(video_frames))
        else:
            print('Input Video is not available')
    except Exception as e:
        print('Error in video creation : {}'.format(e))
    
    # MoRD Model Inference
    mord_scores = mord_model_inference_images_list(args,video_frames)
    iiit_scores = binaryClf_prediction_emarg_images_list(video_frames)

    # Annotate each video frame with details.
    annotated_frames = []
    for i,frame in enumerate(video_frames):
        H,W,_ = frame.shape
        # Inference score of MoRD.
        mord_content = [mord_scores[i][0], round(max(mord_scores[i][1][1],mord_scores[i][1][0]),2)]
        iiit_content = [iiit_scores[i][0], round(iiit_scores[i][1],2)]
        # print('IIIT Content : {}'.format(iiit_content))
        aframe_ = annotate_frame(frame,args.modelName,mord_content,top_left=[5,5])
        aframe_ = annotate_frame(aframe_,' IIIT ',iiit_content,top_left=[5,30])
        annotated_frames.append(aframe_)
    # Combine the frames into video.
    write_video(annotated_frames, '/home/shubhamp/Downloads/binarymodified_test_freezebackbone/out_video_OG1.mp4', args.outputFPS)
    print('~Video Inference : {} completed'.format(args.videoName))
    
if __name__ == '__main__':
    args = get_argparser()
    print(args.inputFPS)
    video_inference(args)
    print('~ COMPLETED')
