"""
Step 5: Apply the block letter mask to the stippled image to simulate selection bias.
This function removes stipples in masked areas (block letter) to visualize a biased estimate.
"""

import numpy as np

def create_masked_stipple(stipple_img: np.ndarray, mask_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply the given mask image to the stippled image to simulate selection bias.

    Parameters
    ----------
    stipple_img : np.ndarray
        2D numpy array of the stippled image, values in [0, 1].
    mask_img : np.ndarray
        2D numpy array of the mask (block letter), values in [0, 1], 
        where 0.0 = black (mask area), 1.0 = white (keep area).
    threshold : float, optional
        Threshold for determining masked region. Pixels in mask_img < threshold are considered "mask" (to be removed/stipple cleared). Default 0.5.

    Returns
    -------
    np.ndarray
        2D numpy array of the same shape as stipple_img, masked: white where removed, original stippling elsewhere.
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError("stipple_img and mask_img must have the same shape.")

    # Where mask is dark (below threshold), set to white (1.0). Else keep stipple.
    masked = np.where(mask_img < threshold, 1.0, stipple_img)
    # Ensure result stays in [0, 1]
    masked = np.clip(masked, 0, 1)
    return masked
