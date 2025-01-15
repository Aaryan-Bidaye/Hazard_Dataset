import cv2
import os
import csv

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
