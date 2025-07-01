import os
import glob
import natsort
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def render_video_from_images(image_folder: str, output_path: str, fps: int = 24, pattern: str = 'view*_pred.png'):
    """Creates a video from a sequence of images."""
    print(f"Rendering video from '{image_folder}'...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    search_pattern = os.path.join(image_folder, pattern)
    found_files = glob.glob(search_pattern)
    
    if not found_files:
        print(f"Error: No images found matching pattern '{pattern}' in '{image_folder}'.")
        return

    sorted_files = natsort.natsorted(found_files)
    print(f"Found {len(sorted_files)} matching images.")

    try:
        clip = ImageSequenceClip(sorted_files, fps=fps)
        clip.write_videofile(output_path, codec='libx264', logger='bar')
        print(f"Video successfully saved to: {output_path}")
    except Exception as e:
        print(f"Failed to render video: {e}")