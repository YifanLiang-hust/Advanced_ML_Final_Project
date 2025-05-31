import os
import pickle
import math
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):
    '''
    A dataset class for ImageNet, inheriting from DatasetBase.
    Attributes:
        dataset_dir (str): The directory name for the dataset.
        image_dir (str): The directory path for images.
        preprocessed (str): The file path for preprocessed data.
        split_fewshot_dir (str): The directory path for few-shot split data.
    Methods:
        __init__(cfg):
            Initializes the ImageNet dataset with the given configuration.
        read_classnames(text_file):
            Reads class names from a text file and returns a dictionary of folder names to class names.
        read_data(classnames, split_dir):
            Reads data from the given split directory and returns a list of Datum objects.
    '''

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, "ID", "ImageNet")
        self.image_dir = os.path.join(self.dataset_dir)
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            print(f"Loading preprocessed data from {self.preprocessed}")
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            print(f"Preprocessed data not found. Preprocessing data...")
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        print(f"NUM_SHOTS: {num_shots}")
        if num_shots >= 1:
            seed = cfg.SEED
            print(f"SEED: {seed}")
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        print(f"SUBSAMPLE: {subsample}")
        train, test = self.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                # classname = " ".join(line[2:])
                classnames[folder] = classname
        return classnames

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        '''
        Divide classes into two groups: base classes and new classes.
        This function takes a list of datasets and divides the classes into two groups.
        The first group represents base classes while the second group represents new classes.
        The division is based on the alphabetical order of the class labels.
            args: A list of datasets, e.g., train, val, and test.
            subsample (str): Specifies which classes to subsample. 
                             Must be one of ["all", "base", "new"].
                             "all" returns the original datasets.
                             "base" returns datasets with only the base classes.
                             "new" returns datasets with only the new classes.
        Returns:
            list: A list of datasets with the classes subsampled according to the specified mode.
        '''
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
