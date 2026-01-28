
import os
import pyvips
import numpy as np
import random
from PIL import Image
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Pyvips Tile Image Processing from Jirka Borovec
def extract_prune_tiles(
    path_img: str, folder: str, size: int = 2048, scale: float = 0.25,
    drop_thr: float = 0.6, max_samples: int = 30
) -> str:
    print(f"processing: {path_img}")
    name, _ = os.path.splitext(os.path.basename(path_img))
    folder = os.path.join(folder, name)
    os.makedirs(folder, exist_ok=True)
    tiles, _ = extract_image_tiles(
        path_img, folder, size=size, scale=scale,
        drop_thr=drop_thr, max_samples=max_samples)
    return folder


def extract_image_tiles(
    p_img, folder, size: int = 2048, scale: float = 0.5,
    drop_thr: float = 0.6, white_thr: int = 240, max_samples: int = 50
) -> list:
    name, _ = os.path.splitext(os.path.basename(p_img))
    im = pyvips.Image.new_from_file(p_img)
    w = h = size
    # https://stackoverflow.com/a/47581978/4521646
    idxs = [(y, y + h, x, x + w) for y in range(0, im.height, h) for x in range(0, im.width, w)]
    # random subsample
    max_samples = max_samples if isinstance(max_samples, int) else int(len(idxs) * max_samples)
    random.shuffle(idxs)
    files = []
    for y, y_, x, x_ in idxs:
        # https://libvips.github.io/pyvips/vimage.html#pyvips.Image.crop
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]
        if tile.shape[:2] != (h, w):
            tile_ = tile
            tile_size = (h, w) if tile.ndim == 2 else (h, w, tile.shape[2])
            tile = np.zeros(tile_size, dtype=tile.dtype)
            tile[:tile_.shape[0], :tile_.shape[1], ...] = tile_
        black_bg = np.sum(tile, axis=2) == 0
        tile[black_bg, :] = 255
        mask_bg = np.mean(tile, axis=2) > white_thr
        if np.sum(mask_bg) >= (np.prod(mask_bg.shape) * drop_thr):
            #print(f"skip almost empty tile: {k:06}_{int(x_ / w)}-{int(y_ / h)}")
            continue
        p_img = os.path.join(folder, f"{int(x_ / w)}-{int(y_ / h)}.png")
        # print(tile.shape, tile.dtype, tile.min(), tile.max())
        new_size = int(size * scale), int(size * scale)
        Image.fromarray(tile).resize(new_size, Image.LANCZOS).save(p_img)
        files.append(p_img)
        # need to set counter check as some empty tiles could be skipped earlier
        if len(files) >= max_samples:
            break
    return files, idxs


class TilesFolderDataset(Dataset):

    def __init__(
        self,
        folder: str,
        image_ext: str =  '.png',
        transforms = None
    ):
        assert os.path.isdir(folder)
        self.transforms = transforms
        self.imgs = glob.glob(os.path.join(folder, "*" + image_ext))

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.imgs[idx]
        assert os.path.isfile(img_path), f"missing: {img_path}"
        img = np.array(Image.open(img_path))[..., :3]
        # filter background
        mask = np.sum(img, axis=2) == 0
        img[mask, :] = 255
        if np.max(img) < 1.5:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        #print(f"img dim: {img.shape}")
        return img

    def __len__(self) -> int:
        return len(self.imgs)


if __name__ == '__main__':
    DATASET_FOLDER = "/kaggle/input/UBC-OCEAN/"
    IMAGES_FOLDER = "./test_tiles"
    TEST_TILES_FOLDER = "/kaggle/temp/test_tiles"

    os.environ['VIPS_CONCURRENCY'] = '4'
    os.environ['VIPS_DISC_THRESHOLD'] = '15gb'

    # Create the test tiles folder
    os.makedirs(TEST_TILES_FOLDER, exist_ok=True)

    # Load the tiles
    ls = sorted(glob.glob(os.path.join(DATASET_FOLDER, "test_images", '*.png')))
    print(f"found images: {len(ls)}")
    folder_tiles = extract_prune_tiles(ls[0], IMAGES_FOLDER, size=2048, scale=0.25)
    dataset = TilesFolderDataset(folder_tiles)
    print(f"found tiles: {len(dataset)}")