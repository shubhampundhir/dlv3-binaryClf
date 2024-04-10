import os
import argparse
import cv2
import numpy as np
import torch
from torch.nn import DataParallel
from torchvision import transforms

import network

def get_argparser():
    parser = argparse.ArgumentParser()

    # Inference Options
    parser.add_argument("--video_path", type=str, default='/home/shubhamp/Downloads/videos/RoadVideosEmarg/RoadVideos/video_1.mp4', help="path to input video")
    parser.add_argument("--output_video_path", type=str, default='/home/shubhamp/Downloads/binarymodified_test_freezebackbone/predicted_video.mp4', help="path to output video")
    parser.add_argument("--checkpoint_path", type=str, default='/home/shubhamp/Downloads/binarymodified_test_freezebackbone/checkpoints_binary_freezedbackbone/CITY_768x768/best_deeplabv3plus_mobilenet_cityscapes_os16.pth', help="path to the trained checkpoint")

    return parser

def read_video_frames(video_path, sample_interval=10):
    video_name = os.path.basename(video_path).split('.mp4')[0]
    os.makedirs(video_name, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
    frames = []
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * sample_interval)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval_frames == 0:
            frames.append(frame)
            cv2.imwrite(os.path.join(video_name, str(frame_count) + '.jpg'), frame)
        frame_count += 1
    cap.release()
    return frames

def binary_classification_inference(video_path, output_video_path, checkpoint_path):
    print("Starting binary classification inference...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    torch.backends.cudnn.benchmark = True  # For potential speed-up

    # Load model
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=2, output_stride=16)
    model = DataParallel(model).to(device)
    print("Model loaded.")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Print loaded checkpoint keys
    # print("Checkpoint Keys:", checkpoint.keys())

    # Load model state dictionary
    model_state_dict = checkpoint["model_state"]

    # Load the model state dict with strict=False to ignore missing keys
    model.load_state_dict(model_state_dict, strict=False)
    print("Model checkpoint loaded.")

    # Print model state keys
    # print("Model State Keys:", model.state_dict().keys())

    model.eval()

    # Transformation for input frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        # Setup video capture and writer
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        print("Video Dimensions (width, height):", width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'MJPG'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply the same transformation used during training
            input_frame = transform(frame).unsqueeze(0).to(device)

            # Perform inference
            outputs = model(input_frame)
            preds = torch.sigmoid(outputs)
            print("Predictions:", preds)

            # Dynamic Thresholding (Adjust the threshold as needed)
            threshold = 0.5
            binary_preds = (preds > threshold).float()
            binary_preds = binary_preds.cpu().numpy()
            print("Binary Predictions:", binary_preds)

            # Add tint to signify class
            tint_color = np.array((0, 255, 0) if binary_preds[0][0] == 1 else (0, 0, 255), dtype=frame.dtype)
            # print("Tint Color:", tint_color)

            # Blend frames using NumPy operations
            tinted_frame = frame * 0.5 + tint_color * 0.5

            # Merge arrays using cv2.merge for compatibility
            vis_frame = np.concatenate([frame, tinted_frame], axis=-1).astype(np.uint8)
            # print("Tinted Frame:", tinted_frame)
            # print("Vis Frame:", vis_frame)

            # Write the processed frame to the output video
            out.write(vis_frame)

    cap.release()
    out.release()
    print("Inference completed!")

def main():
    opts = get_argparser().parse_args()

    # Inference on the video
    binary_classification_inference(opts.video_path, opts.output_video_path, opts.checkpoint_path)

if __name__ == '__main__':
    main()
