import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extracts frames from a video at the specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Folder to save extracted frames.
        frame_rate (int): Number of frames to extract per second.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ Video not found at: {video_path}")

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("❌ Couldn't read FPS from video. Is the file valid?")

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every nth frame
        if int(count % (fps // frame_rate)) == 0:
            frame_filename = f"frame_{saved:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"✅ Extracted {saved} frames from {video_path} into {output_folder}")