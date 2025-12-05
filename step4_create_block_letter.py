"""
Step 4: Generate a block-style letter image for the statistics meme.
Creates a synthetic black-and-white image containing a bold, centered letter
(such as "S" by default), rendered as a white background (1.0) with a black (0.0)
block letter for use as an example or template in meme or image processing workflows.
"""


import numpy as np

def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Generates a 2D array representing a centered, block-style letter (default "S")
    with black (0.0) letter on a white (1.0) background.

    Parameters
    ----------
    height : int
        Height of the output image in pixels.
    width : int
        Width of the output image in pixels.
    letter : str
        The letter to render (default "S").
    font_size_ratio : float
        Fraction of the minimum(height, width) to use for the font size (default 0.9).

    Returns
    -------
    arr : np.ndarray
        Image as 2D array (height, width) with values in [0, 1]
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create a white background
    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)

    # Try multiple fonts for block/bold letter
    font_paths = [
        # DejaVu Sans Bold is often available with PIL
        "DejaVuSans-Bold.ttf",
        # Common locations for Arial or Liberation fonts
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "arialbd.ttf",
    ]
    font = None
    # Calculate font size
    font_size = int(min(height, width) * font_size_ratio)

    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            continue
    if font is None:
        # If truetype not available, fallback to default
        font = ImageFont.load_default()
        # The default font will be small; stretch later

    # Get text size and position for centering
    try:
        bbox = draw.textbbox((0, 0), letter, font=font, anchor=None)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        # For older PIL versions
        text_w, text_h = draw.textsize(letter, font=font)

    text_x = (width - text_w) // 2
    text_y = (height - text_h) // 2

    # Draw black letter ("S") on white background
    draw.text((text_x, text_y), letter, fill=0, font=font)

    # Convert to numpy array in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Ensure 2D, and squeeze if not
    if arr.ndim > 2:
        arr = arr.squeeze()

    # Clip range just in case
    arr = np.clip(arr, 0.0, 1.0)

    return arr

