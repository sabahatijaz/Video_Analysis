import cv2
import numpy as np
import torch
import glob
import os
from tqdm import tqdm
from PIL import Image
from decouple import config
from transformers import CLIPImageProcessor, CLIPModel
from image_similarity_measures.quality_metrics import ssim

# Load the CLIP model
model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

def OpenAIClipSimilarityCheck(imageA, imageB):
    """
    Calculate the similarity score between two images using the CLIP model.

    Args:
        imageA (str): Path to the first image.
        imageB (str): Path to the second image.

    Returns:
        float: Similarity score between -1 and 1.
    """
    def load_and_preprocess_image(image_path):
        # Load the image from the specified path
        image = Image.open(image_path)

        # Apply the CLIP preprocessing to the image
        image = preprocess(image, return_tensors="pt")

        # Return the preprocessed image
        return image

    # Load and preprocess the two images for CLIP
    image_a = load_and_preprocess_image(imageA)["pixel_values"]
    image_b = load_and_preprocess_image(imageB)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()

def final_Scan(Folder_path):
    """
    Scan a folder for duplicate images based on CLIP similarity and remove duplicates.

    Args:
        Folder_path (str): Path to the folder containing images.
    """
    img_dir = Folder_path
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    thresh = config("Video_Similarity_Threshold")

    with tqdm(total=len(files) * (len(files) - 1) // 2) as pbar:
        for i in range(len(files) - 1):
            for j in range(i + 1, len(files) - 1, 1):
                if j < len(files) and len(files) > 12:
                    OpenAI_check = OpenAIClipSimilarityCheck(files[i], files[j])
                    if OpenAI_check > thresh:
                        os.remove(files[j])
                        del files[j]
                    pbar.update(1)

def get_video_duration(video_path):
    """
    Get the total duration of a video in seconds.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Total duration of the video in seconds.
    """
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    # Get the total number of frames and frames per second (fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the total duration in seconds
    duration = frame_count / fps

    # Release the video capture object
    cap.release()

    return duration, fps

def is_valid(image):
    """
    Check if an image is valid based on its saturation channel.

    Args:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        bool: True if the image is valid, False if it is considered noise.
    """
    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.5
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.055
    return s_perc > s_thr

def FrameCapture(Video_path, Folder_Path):
    """
    Extract frames from a video, filter valid frames, and save them to a folder.

    Args:
        Video_path (str): Path to the video file.
        Folder_Path (str): Path to the folder to save frames.
    """
    total_duration, fps = get_video_duration(Video_path)
    frames_interval = int(fps)
    vidObj = cv2.VideoCapture(Video_path)

    if not vidObj.isOpened():
        print("Error: Could not open video file.")
        exit()

    count = 0
    success = 1
    frame_count = 0
    ssim_threshold=config("SSIM_Video")

    while success:
        success, image = vidObj.read()

        if success == 1:
            if frame_count % frames_interval == 0:
                noise1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                res = is_valid(noise1)
                if res:
                    img_dir = Folder_Path
                    data_path = os.path.join(img_dir, '*g')
                    files = glob.glob(data_path)
                    match = 0
                    for f1 in files:
                        in_img1 = cv2.imread(f1)
                        out_ssim = ssim(in_img1, image)
                        if out_ssim > ssim_threshold:
                            match = 1
                            break
                    if match == 0:
                        Path = img_dir
                        cv2.imwrite(os.path.join(Path, "frame%d.jpg" % count), image)

            frame_count += 1
        count += 1

if __name__ == '__main__':
    Video_Path = 'videos/random.mp4'
    Folder_Path = 'UnblurrNUnique'
    FrameCapture(Video_Path, Folder_Path)
    final_Scan(Folder_Path=Folder_Path)
