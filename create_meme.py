"""
Step 6: Assemble the four panels into a professional-looking meme.
This function combines the original image, stippled image, block letter mask,
and masked stippled image into a single image with four panels.
""" 


import numpy as np

def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "pink"
) -> None:
    """
    Assemble a four-panel meme showing selection bias.

    Parameters
    ----------
    original_img : np.ndarray
        Grayscale reality image (height, width), values in [0, 1].
    stipple_img : np.ndarray
        Stippled model image (height, width), values in [0, 1].
    block_letter_img : np.ndarray
        Block letter mask image (height, width), values in [0, 1].
    masked_stipple_img : np.ndarray
        Masked stippled estimate (height, width), values in [0, 1].
    output_path : str
        Path to save the resulting PNG meme.
    dpi : int, optional
        Dots-per-inch for saving (default 150).
    background_color : str, optional
        Background color for the figure (default "white").
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # Prepare images as a list
    images = [original_img, stipple_img, block_letter_img, masked_stipple_img]
    titles = ["Reality", "Your Model", "Selection Bias", "Estimate"]

    # Ensure all images are 2D and have same size (use first image's size)
    h, w = images[0].shape
    processed_images = []
    for img in images:
        # Remove singleton color channels if present
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        # Resize if needed
        if img.shape != (h, w):
            # Only import here as needed
            from PIL import Image
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            pil_img = pil_img.resize((w, h), resample=Image.BICUBIC)
            img = np.array(pil_img) / 255.0
        processed_images.append(img)

    # Set up figure
    fig_width = 14  # inches (~3.5 per panel for 4 panels)
    fig_height = 4.5
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = gridspec.GridSpec(1, 4, wspace=0.15, left=0.03, right=0.98)
    fig.patch.set_facecolor(background_color)

    # Panel style
    for i, (img, title) in enumerate(zip(processed_images, titles)):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=16)
        ax.axis('off')
        # Optional: add thin border around each panel
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        # Fine-tune subplot margins for professional look
        ax.set_xticks([])
        ax.set_yticks([])

    # Optional meme title - not required per the template, so not included

    # Save file (bbox_inches tight for zero whitespace)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=background_color, pad_inches=0.05)
    plt.close(fig)
