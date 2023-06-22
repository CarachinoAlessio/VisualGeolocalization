import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import logging


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25, args=None):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        if args:
            self.night_test = args.night_test  # if args.night_test else False
            self.night_brightness = args.night_brightness
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)

        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"), recursive=True))

        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(
            float)

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)

        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        if hasattr(self, "night_test") and self.night_test and index < len(self.database_paths):
            pil_img = transforms.functional.adjust_brightness(pil_img, brightness_factor=self.night_brightness)

        normalized_img = self.base_transform(pil_img)
        return normalized_img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"

    def get_positives(self):
        return self.positives_per_query


class DataAugTestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25, test_method=None, resize=False):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)

        # data augmentation stuff
        self.test_method = test_method

        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"), recursive=True))

        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(
            float)

        # set db image size (height, width)
        if resize:
            self.db_image_size = Image.open(self.database_paths[0]).size
            self.db_image_size = (self.db_image_size[1], self.db_image_size[0])
        else:
            self.db_image_size = None

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)

        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)

        if self.db_image_size:
            if self.test_method == "database":
                normalized_img = transforms.functional.resize(normalized_img, self.db_image_size)
            else:
                normalized_img = self._test_query_transform(normalized_img)

        return normalized_img, index

    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "central_crop":
            scale = max(self.db_image_size[0] / H, self.db_image_size[1] / W)
            proc_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            proc_img = transforms.functional.center_crop(proc_img, self.db_image_size)
            assert proc_img.shape[1:] == torch.Size(
                self.db_image_size), f"proc_img.shape: {proc_img.shape}, db_image_size: {self.db_image_size}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.db_image_size)
            proc_img = transforms.functional.resize(img, shorter_side)
            proc_img = torch.stack(transforms.functional.five_crop(proc_img, shorter_side))
            assert proc_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{proc_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        elif self.test_method == "five_custom":
            custom_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(),
                transforms.CenterCrop(self.db_image_size),
            ])
            # like center crop
            scale = max(self.db_image_size[0] / H, self.db_image_size[1] / W)
            proc_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            proc_img = transforms.functional.center_crop(proc_img, self.db_image_size)

            # five crops approach
            proc_img = torch.stack([
                proc_img,
                *[custom_transforms(proc_img) for _ in range(4)]
            ])
        else:
            # single query
            proc_img = transforms.functional.resize(img, min(self.db_image_size))
        return proc_img

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"

    def get_positives(self):
        return self.positives_per_query