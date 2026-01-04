"""
Script to crop the last 2/3 columns from GIF files and save them.
Processes .gif files and saves cropped versions.
"""
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image


def crop_last_two_thirds_pil(image):
    """
    Crop the last 2/3 columns from a PIL Image (keeping all rows).
    
    Args:
        image: PIL Image object
    
    Returns:
        Cropped PIL Image with last 2/3 columns
    """
    width, height = image.size
    start_col = width // 3  # Start from 1/3 of width
    # Crop: (left, top, right, bottom)
    bbox = (start_col, 0, width, height)
    cropped = image.crop(bbox)
    return cropped


def process_single_file(input_path, output_path=None, overwrite=False):
    """
    Process a single .gif file.
    
    Args:
        input_path: Path to input .gif file
        output_path: Path to save cropped GIF (if None, auto-generate)
        overwrite: Whether to overwrite existing files
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return False
    
    if not input_path.suffix.lower() == '.gif':
        print(f"Warning: File is not .gif format: {input_path}")
        return False
    
    # Load GIF
    try:
        gif = Image.open(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return False
    
    # Extract all frames
    frames = []
    try:
        while True:
            frames.append(gif.copy())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    
    if len(frames) == 0:
        print(f"Error: No frames found in {input_path}")
        return False
    
    # Crop each frame
    cropped_frames = []
    for frame in frames:
        cropped_frame = crop_last_two_thirds_pil(frame)
        cropped_frames.append(cropped_frame)
    
    # Determine output path
    if output_path is None:
        # Save in same directory with '_cropped' suffix
        output_path = input_path.parent / f"{input_path.stem}_cropped.gif"
    else:
        output_path = Path(output_path)
    
    # Check if output exists
    if output_path.exists() and not overwrite:
        print(f"Warning: Output file exists (use --overwrite to replace): {output_path}")
        return False
    
    # Save cropped GIF
    try:
        # Save first frame with all others appended
        if len(cropped_frames) > 0:
            cropped_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=cropped_frames[1:],
                loop=gif.info.get('loop', 0),
                duration=gif.info.get('duration', 100)  # Default 100ms if not specified
            )
            print(f"Saved cropped GIF: {output_path} ({len(cropped_frames)} frames, size: {cropped_frames[0].size})")
            return True
        else:
            print(f"Error: No frames to save")
            return False
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return False


def process_directory(input_dir, output_dir=None, overwrite=False, recursive=False):
    """
    Process all .gif files in a directory.
    
    Args:
        input_dir: Directory containing .gif files
        output_dir: Directory to save cropped GIFs (if None, save in same directory)
        overwrite: Whether to overwrite existing files
        recursive: Whether to process subdirectories recursively
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Find all .gif files
    if recursive:
        gif_files = list(input_dir.rglob("*.gif"))
    else:
        gif_files = list(input_dir.glob("*.gif"))
    
    if len(gif_files) == 0:
        print(f"No .gif files found in {input_dir}")
        return
    
    print(f"Found {len(gif_files)} .gif files")
    
    # Process each file
    success_count = 0
    for gif_file in tqdm(gif_files, desc="Processing files"):
        if output_dir is None:
            # Save in same directory
            output_path = None
        else:
            # Maintain relative path structure
            rel_path = gif_file.relative_to(input_dir)
            output_path = Path(output_dir) / rel_path.parent / f"{rel_path.stem}_cropped.gif"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if process_single_file(gif_file, output_path, overwrite):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(gif_files)} files successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Crop the last 2/3 columns from GIF files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input file or directory path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file or directory path (if None, saves in same location with '_cropped' suffix)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process subdirectories recursively (only for directory input)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        process_single_file(input_path, args.output, args.overwrite)
    elif input_path.is_dir():
        # Process directory
        process_directory(input_path, args.output, args.overwrite, args.recursive)
    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == "__main__":
    main()

