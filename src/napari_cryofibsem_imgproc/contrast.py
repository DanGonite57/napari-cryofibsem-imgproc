import concurrent.futures

import numpy as np
from magicgui import magic_factory
from napari.layers import Image
from napari_plugin_engine import napari_hook_implementation
from skimage.exposure import equalize_adapthist

from .utils import restore_original_type


def process_slice(slice_data, tile_row, tile_col, clip_limit):
    img_clahe = equalize_adapthist(slice_data,
                                   kernel_size=(tile_row, tile_col),
                                   clip_limit=clip_limit,
                                   nbins=256)

    # Converts and normalizes range to original datatype
    return restore_original_type(img_clahe, slice_data.dtype)


@magic_factory(
    call_button="Enhance contrast",
    image={"label": "Input Image"},
    clip_limit={"label": "Clip Limit", "choices": ["0.005", "0.007", "0.008", "0.009", "0.010", "0.011", "0.012", "0.013", "0.014", "0.015"]},
    tile_row={"label": "Number of Rows"},
    tile_col={"label": "Number of Columns"}
)
def contrast(
        image: Image,
        clip_limit: str = "0.007",
        tile_row: int = 50,
        tile_col: int = 50
) -> Image:
    """
    This widget enhances the contrast of the chosen image according to the Contrast-Limited Adaptive
    Histogram Equalization algorithm. It utilizes the equalize_adapthist function from scikit-image.

    Parameters
    ----------
    Image : "Image"
        Image to be processed

    Clip Limit : str
        Contrast limit for localized changes in the histogram

    Number of Rows : int
        Number of rows into which the image will be divided
        First argument in kernel size parameter is 1/X of the height of the image

    Number of Columns : int
        Nnumber of columns into which the image will be divided
        First argument in kernel size parameter is 1/X of the width of the image

    Returns
    -------
        napari Image layer containing the contrast-enhanced image
    """
    if image is None:  # Handles null cases
        print("Please select an image layer.")
        return

    clip_limit_float = float(clip_limit)  # Converts string of clip limit value into a float

    if len(image.data.shape) > 2:
        stack = image.data
        processed_slices = []
        slice_order = []  # To keep track of slice order

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {executor.submit(process_slice, stack[slice_idx], tile_row, tile_col, clip_limit_float):
                                   slice_idx for slice_idx in range(stack.shape[0])}
            for future in concurrent.futures.as_completed(future_to_slice):
                slice_idx = future_to_slice[future]
                slice_order.append(slice_idx)
                processed_slices.append(future.result())

        # Sort processed slices based on original order
        processed_slices = [x for _, x in sorted(zip(slice_order, processed_slices))]
        processed_stack = np.stack(processed_slices)

    else:
        processed_stack = process_slice(image.data, tile_row, tile_col, clip_limit_float)

    image_name = f"CoEn_clip{clip_limit_float}_row{tile_row}_col{tile_col}"

    print(f"\nImage or Stack contrast enhanced successfully!\n{image_name} added to Layer List.")

    # Returns the processed stack with the parameters in the name
    return Image(processed_stack, name=image_name)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return contrast
