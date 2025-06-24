import subprocess
from pathlib import Path
import os

def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 1):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_stem = video_path.stem  # video1.MOV → video1
    output_pattern = output_dir / f"{filename_stem}_%04d.jpg"

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        str(output_pattern)
    ]
    subprocess.run(command, check=True)
    print(f"✅ Extracted from {video_path.name}")

def extract_from_directory(input_dir: str, output_dir: str, fps: int = 1):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for video_file in input_dir.glob("*.MOV"):
        extract_frames_from_video(video_file, output_dir, fps)

if __name__ == "__main__":
    extract_from_directory("data/videos", "data/raw_frames", fps=1) 