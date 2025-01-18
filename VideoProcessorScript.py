import cv2
import os
import csv
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generatorimport import SAM2AutomaticMaskGenerator


###sam setup
# selecting the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# Load the model
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

# Video to frame conversion
def process_videos_in_folder(input_folder, output_folder, frame_count=150, output_size=256):
    """

    Args:
        input_folder (str): path of input directory
        output_folder (str): path of output directory
        frame_count (int, optional): Defaults to 150 frames ie. 5 seconds
        output_size (int, optional): Compresion of the largest crop possible from the video size. Defaults to 256x256 compression once cropped
    """
    # Ensure the output folder exists otherwise creates it
    os.makedirs(output_folder, exist_ok=True)

    # Process each video in the input folder
    video_index = 48
    for video_filename in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_filename)

        # Skip non-video files
        if not video_filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            print(f"Skipping non-video file: {video_filename}")
            continue

        # Create a numbered subfolder for the video in the output folder
        video_output_folder = os.path.join(output_folder, str(video_index))
        os.makedirs(video_output_folder, exist_ok=True)
        video_index += 1

        print(f"Processing video: {video_filename}")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            continue

        # Read all frames sequentially
        frames = []
        total_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            total_frames += 1

        cap.release()
        print(f"Total frames read from {video_filename}: {total_frames}")

        # Extract the last `frame_count` frames
        if total_frames < frame_count:
            print(f"Warning: Video {video_filename} has fewer than {frame_count} frames. Extracting all available frames.")
            frames_to_save = frames
        else:
            frames_to_save = frames[-frame_count:]

        # CSV file setup
        csv_file_path = os.path.join(video_output_folder, 'classification.csv')
        csv_data = []

        # Save, crop, and resize the extracted frames
        for idx, frame in enumerate(frames_to_save, start=1):
            height, width, _ = frame.shape

            # Determines the largest square possible
            square_size = min(height, width)

            # Calculates the crop coordinates for the middle square
            crop_x = (width - square_size) // 2
            crop_y = (height - square_size) // 2
            cropped_frame = frame[crop_y:crop_y + square_size, crop_x:crop_x + square_size]

            # Resize the cropped frame to output_size x output_size
            resized_frame = cv2.resize(cropped_frame, (output_size, output_size), interpolation=cv2.INTER_AREA)

            # Save the resized frame as a JPEG file
            frame_filename = f"frame_{idx}.jpg"
            frame_path = os.path.join(video_output_folder, frame_filename)
            cv2.imwrite(frame_path, resized_frame)
            print(f"Saved {frame_filename} in {video_output_folder}")
            
            # Segmentation of frame
            sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
            mask_generator = SAM2AutomaticMaskGenerator(sam2)
            
            masks = mask_generator.generate(frame_path)
            
            
            
            # Save segmentation mask as a JPEG file
            
            # mask_filename = f"mask_{idx}.jpg"
            # mask_path = os.path.join(video_output_folder, mask_filename)
            # cv2.imwrite(mask_path, masks)
            # print(f"Saved {mask_filename} in {video_output_folder}")

            # Add entry to CSV
            csv_data.append([frame_filename, 0])

        # Write CSV file
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_data)
        print(f"CSV file saved as {csv_file_path}")

    print(f"Processing complete. Frames and CSV files saved in {output_folder}")


input_folder = input("Enter the path for your desired input directory")
output_folder = input("Enter the path for your desired output directory")
process_videos_in_folder(input_folder, output_folder)
