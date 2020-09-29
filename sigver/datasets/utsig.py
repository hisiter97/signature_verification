import os

import cv2
from skimage.io import imread
from datasets.base import IterableDataset
from skimage import img_as_ubyte


class UTSigDataset(IterableDataset):
    """ Helper class to load the UTSig dataset
    """

    def __init__(self, path='./data/UTSig', extension='tif'):
        self.path = path
        self.users = [int(user) for user in sorted(os.listdir(os.path.join(self.path, 'Genuine')))]
        self.extension = extension

    @property
    def genuine_per_user(self):
        return 27

    @property
    def skilled_per_user(self):
        return 6

    @property
    def simple_per_user(self):
        return 66

    @property
    def opposite_per_user(self):
        return 3

    @property
    def maxsize(self):
        return 952, 1360

    @property
    def canvas_size(self):
        return 400, 574

    @property
    def resize_size(self):
        return 350, 350

    def get_user_list(self):
        return self.users

    def iter_genuine(self, user):
        """ Iterate over genuine signatures for the given user"""

        user_genuine_folder = os.path.join(self.path, 'Genuine', '{:d}'.format(user))
        user_genuine_files = sorted(os.listdir(user_genuine_folder))
        for f in user_genuine_files:
            full_path = os.path.join(user_genuine_folder, f)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield img, f

    # FAKE
    def iter_forgery(self, user):
        """ Iterate over skilled forgeries for the given user"""

        user_forgery_folder = os.path.join(self.path, 'Forgery', 'Skilled', '{:d}'.format(user))
        user_forgery_files = sorted(os.listdir(user_forgery_folder))
        for f in user_forgery_files:
            full_path = os.path.join(user_forgery_folder, f)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield img, f

    def iter_simple_forgery(self, user):
        user_simple_forgery_folder = os.path.join(self.path, 'Forgery', 'Simple', '{:d}'.format(user))
        user_simple_forgery_files = sorted(os.listdir(user_simple_forgery_folder))
        for f in user_simple_forgery_files:
            full_path = os.path.join(user_simple_forgery_folder, f)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield img, f

    def iter_opposite_forgery(self, user):
        user_opposite_forgery_folder = os.path.join(self.path, 'Forgery', 'Opposite Hand', '{:d}'.format(user))
        user_opposite_forgery_files = sorted(os.listdir(user_opposite_forgery_folder))
        for f in user_opposite_forgery_files:
            full_path = os.path.join(user_opposite_forgery_folder, f)
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yield img, f


    def get_signature(self, user, img_idx, forgery):
        """ Returns a particular signature (given by user id, img id and
            whether or not it is a forgery
        """

        if forgery:
            prefix = 'cf'
        else:
            prefix = 'c'
        filename = '{}-{:03d}-{:02d}.{}'.format(prefix, user, img_idx,
                                                self.extension)
        full_path = os.path.join(self.path, '{:03d}'.format(user), filename)
        return img_as_ubyte(imread(full_path, as_gray=True))

