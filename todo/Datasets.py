import os
from pathlib import Path

# type hints
from typing import List, Union, Dict, Any, Tuple, Optional, Literal

# plotting
import matplotlib.pyplot as plt

# calculations
import numpy as np
from sklearn.utils import shuffle

# load data
import cebra
from cebra import CEBRA

# own
from Helper import (
    check_needed_keys,
    get_str_from_dict,
    global_logger,
    force_1_dim_larger,
    force_equal_dimensions,
    encode_categorical,
    is_list_of_ndarrays,
    group_by_binned_data,
    bin_array,
    uncode_categories,
    save_file_present,
    make_list_ifnot,
    add_missing_keys,
    encode_categorical,
    do_critical,
    logical_and_extend,
    DataFilter,
    check_obj_equivalence,
    clean_filename,
)
from Visualizer import Vizualizer
from SignalProcessing import may_butter_lowpass_filter
from Setups import Femtonics, Thorlabs, Inscopix, Environment
from calculations import _compute_paired_pvalue, _compute_unpaired_pvalue
from Setups import (
    Active_Avoidance,
    Treadmill_Setup,
    Trackball_Setup,
    Openfield_Setup,
    Wheel_Setup,
)


class Dataset:
    needed_keys = ["setup"]
    labels_describe_space = ["position", "velocity", "acceleration"]

    def __init__(
        self,
        key,
        metadata,
        path=None,
        data=None,
        raw_data_object=None,
        root_dir=None,
        task_id=None,
    ):
        """Initialize the Dataset class.

        Parameters
        ----------
        key : str
            The key of the dataset, e.g. "position", "velocity", etc.
        metadata : dict
            Metadata of the dataset, containing information about the setup, preprocessing, etc.
        path : Path, optional
            The path to the dataset file. If not provided, it will be set to None.
        data : np.ndarray, optional
            The data of the dataset. If not provided, it will be set to None.
        raw_data_object : Dataset, optional
            A raw data object from which the dataset can be created. If not provided, it will be set to None.
        root_dir : Path, optional
            The root directory of the project. If not provided, it will be set to None.
        task_id : str, optional
            The task ID for the dataset. If not provided, it will be set to None.

        Attributes
        ----------
        data_filter : dict
            A dictionary to filter the data based on different types of data.
            The keys are the data types and the values are the filter criteria.
            currently supports:
            - "behavior": for filtering behavior data
                - "range": Tuple for filtering based on a range of values
        """
        # initialize plotting parameters
        super().__init__()
        self.root_dir: Path = root_dir
        self.path: Path = path
        self.data_dir = None
        self.unfiltered_data: np.ndarray = data
        self.unfiltered_binned_data: np.ndarray = None
        self.data_filter: Dict[
            Literal["range",],
            Any,
        ] = None
        self.data_filter: DataFilter = None
        self.key = key
        self.task_id = task_id
        self.raw_data_object = raw_data_object
        self.metadata = metadata
        self.fps = None if "fps" not in self.metadata.keys() else self.metadata["fps"]
        self.fps = float(self.fps) if self.fps else None
        check_needed_keys(metadata, Dataset.needed_keys)
        self.setup = self.get_setup(self.metadata["setup"])
        self.plot_attributes = Vizualizer.default_plot_attributes()
        self.category_map = None
        self.refine_metadata()

    def get_setup(self, setup_name, preprocessing_name, method_name):
        raise NotImplementedError(
            f"ERROR: Function get_setup is not defined for {self.__class__}."
        )

    def load(
        self,
        path=None,
        save=True,
        plot=True,
        data_filter: Union[
            Dict[
                Literal[
                    "position",
                    "distance",
                    "moving",
                    "velocity",
                    "acceleration",
                    "stimulus",
                ],
                Any,
            ]
        ] = None,
        regenerate=False,
        regenerate_plot=False,
    ):
        if not type(self.unfiltered_data) == np.ndarray:
            self.path = path if path else self.path
            self.data_dir = self.path.parent
            # if no raw data exists, regeneration is not possible

            # Check if the file exists, and if it does, load the data, else create the dataset
            if path.exists() and not regenerate:
                global_logger.info(f"Loading {self.path}")
                # ... and similarly load the .h5 file, providing the columns to keep
                # continuous_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["continuous1", "continuous2", "continuous3"])
                # discrete_label = cebra.load_data(file="auxiliary_behavior_data.h5", key="auxiliary_variables", columns=["discrete"]).flatten()
                self.unfiltered_data = cebra.load_data(file=self.path)
                if self.unfiltered_data.ndim == 2:
                    # format of data should be [num_time_points, num_cells]
                    self.unfiltered_data = force_1_dim_larger(data=self.unfiltered_data)
            else:
                global_logger.warning(
                    f"Generating data based on raw_data_object for {self.key} in {self.path}."
                )
                self.unfiltered_data = self.create_dataset(
                    self.raw_data_object, save=save
                )
                if save:
                    np.save(self.path, self.unfiltered_data)
        self.unfiltered_data = self.correct_data(self.unfiltered_data)
        if data_filter is not None:
            self.data_filter = data_filter
        self.refine_metadata()
        if plot:
            self.plot(regenerate_plot=regenerate_plot)
        self.unfiltered_binned_data, self.category_map = self.bin_data(self.data)
        return self.data

    def refine_metadata(self):
        self.define_plot_attributes()

    def define_plot_attributes(self):
        raise NotImplementedError(
            f"Function define_plot_attributes is not defined for {self.__class__}."
        )

    def create_dataset(self, raw_data_object=None, save=True):
        raw_data_object = raw_data_object or self.raw_data_object
        if self.raw_data_object:
            global_logger.debug(
                f"Creating {self.key} dataset based on raw data from {raw_data_object.key}."
            )
            data = self.process_raw_data(save=save)
            return data
        else:
            global_logger.warning(
                f"No raw data given. Creation not possible. Skipping."
            )

    def process_raw_data(self, save=True):
        global_logger.critical(
            f"Function for creating {self.key} dataset from raw data is not defined."
        )
        raise NotImplementedError(
            f"ERROR: Function for creating {self.key} dataset from raw data is not defined."
        )

    @property
    def data(self):
        """Returns the data of the dataset, applying the data filter if set."""
        return self.data_filter.filter(self.unfiltered_data, self.key)

    @property
    def binned_data(self):
        """Returns the binned data of the dataset, applying the data filter if set."""
        return self.data_filter.filter(self.unfiltered_binned_data, self.key)

    def correct_data(self, data):
        data_2d = Dataset.force_2d(data)
        return data_2d

    def bin_data(self, data, bin_size=None):
        """
        binning is not applied to this type of data
        """
        return data, None

    def refine_plot_attributes(
        self,
        title=None,
        ylable=None,
        xlimits=None,
        save_path=None,
    ):
        title = self.plot_attributes["title"] or title
        if title is None:
            if self.path is not None:
                title = self.path.stem
            else:
                title = self.key

        self.plot_attributes["title"] = f"{title} data"
        self.plot_attributes["title"] = (
            self.plot_attributes["title"]
            if self.plot_attributes["title"][-4:] == "data"
            else self.plot_attributes["title"] + " data"
        )

        self.plot_attributes["ylable"] = (
            self.plot_attributes["ylable"] or ylable or self.key
        )

        self.plot_attributes["xlimits"] = (
            self.plot_attributes["xlimits"] or xlimits or (0, len(self.data))
        )

        descriptive_metadata_keys = [
            "area",
            "stimulus_type",
            "method",
            "preprocessing_software",
            "setup",
        ]

        descriptive_metadata_txt = get_str_from_dict(
            dictionary=self.metadata, keys=descriptive_metadata_keys
        )
        if descriptive_metadata_txt not in self.plot_attributes["title"]:
            self.plot_attributes["title"] += f" {descriptive_metadata_txt}"

        self.plot_attributes["title"] += " " + self.data_filter.filter_description()
        figure_title = clean_filename(self.plot_attributes["title"] + ".png")

        if self.path is not None:
            self.plot_attributes["save_path"] = (
                self.plot_attributes["save_path"]
                or save_path
                or self.root_dir.parent.joinpath("figures", figure_title)
            )

    def plot(
        self,
        figsize=None,
        title=None,
        xlable=None,
        xlimits=None,
        xticks=None,
        ylable=None,
        ylimits=None,
        yticks=None,
        seconds_interval=5,
        fps=None,
        num_ticks=50,
        save_path=None,
        regenerate_plot=None,
        show=False,
        dpi=300,
        as_pdf=False,
    ):
        self.refine_plot_attributes(
            title=title, ylable=ylable, xlimits=xlimits, save_path=save_path
        )
        if self.data_filter.filter_description() not in self.plot_attributes["title"]:
            self.plot_attributes["title"] += f" {self.data_filter.filter_description()}"
        plot_present = save_file_present(fpath=self.plot_attributes["save_path"])
        if regenerate_plot or not plot_present:
            self.plot_attributes = Vizualizer.default_plot_start(
                plot_attributes=self.plot_attributes,
                figsize=figsize,
                xlable=xlable,
                xlimits=xlimits,
                xticks=xticks,
                ylimits=ylimits,
                yticks=yticks,
                num_ticks=num_ticks,
                fps=fps,
            )
            self.plot_data()
            Vizualizer.default_plot_ending(
                plot_attributes=self.plot_attributes,
                regenerate_plot=True,
                show=show,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        else:
            Vizualizer.plot_image(
                plot_attributes=self.plot_attributes,
                show=show,
            )

    def plot_data(self):
        # dimensions =  self.data.ndim
        dimensions = len(self.metadata["environment_dimensions"])
        if dimensions == 1:
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
            )
        elif dimensions == 2:
            if self.key != "position":
                raw_data_object = self.raw_data_object
                while raw_data_object.key != "position":
                    # fetch raw data object until position data is found
                    raw_data_object = raw_data_object.raw_data_object
                    if raw_data_object is None:
                        do_critical(
                            ValueError,
                            f"No position data found for plotting 2D data representation of {self.key}.",
                        )
                position_data = raw_data_object.data
            else:
                position_data = self.data

            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=position_data,
                border_limits=self.metadata["environment_dimensions"],
                plot_attributes=self.plot_attributes,
            )

    def get_transdata(self, transformation: Literal["relative", "binned"] = None):
        return self.data

    @staticmethod
    def split_ids(
        num_ids,
        split_ratio: float = 0.8,
        method: Literal["random", "sequential"] = "random",
    ):
        """
        Split the ids into training and testing sets.
        """
        global_logger.debug(
            f"Splitting {num_ids} ids into {split_ratio:.0%} training and {1-split_ratio:.0%} testing."
        )
        ids = np.arange(num_ids)
        if method == "random":
            np.random.shuffle(ids)
        elif method == "sequential":
            pass
        else:
            do_critical(
                ValueError,
                f"Method {method} not supported for splitting. Use 'random' or 'sequential'.",
            )
        split_index = int(num_ids * split_ratio)
        ids_train = ids[:split_index]
        ids_test = ids[split_index:]
        return ids_train, ids_test

    @staticmethod
    def split(
        data,
        split_ratio: float = 0.8,
        ids_train: Optional[List[int]] = None,
        ids_test: Optional[List[int]] = None,
        method: Literal["random", "sequential"] = "sequential",
    ):
        """
        Split the data into training and testing sets.
        """
        if ids_train is None and ids_test is None:
            ids_train, ids_test = Dataset.split_ids(
                data.shape[0], split_ratio=split_ratio, method=method
            )
        elif (
            ids_train is None
            and ids_test is not None
            or ids_train is not None
            and ids_test is None
        ):
            do_critical(
                ValueError,
                f"Both ids_train and ids_test should be provided or None. Got {ids_train} and {ids_test}.",
            )
        data_train = data[ids_train]
        data_test = data[ids_test]
        return data_train, data_test

    @staticmethod
    def shuffle(data):
        global_logger.debug(f"Shuffling data.")
        return shuffle(data)

    @staticmethod
    def filter_by_idx(data, idx_to_keep=None):
        if (
            isinstance(idx_to_keep, np.ndarray)
            and len(idx_to_keep) > 0
            or isinstance(idx_to_keep, list)
            and len(idx_to_keep) > 0
        ):
            # global_logger.debug(f"Filtering data by idx_to_keep.")
            if len(idx_to_keep) > data.shape[0] or max(idx_to_keep) >= data.shape[0]:
                do_critical(
                    ValueError,
                    f"idx_to_keep has more dimensions than data. idx_to_keep: {idx_to_keep.shape}, data: {data.shape}",
                )
            if data.shape != idx_to_keep.shape:
                data_filtered = data[idx_to_keep.flatten()]
            else:
                data_filtered = data[idx_to_keep]
            return data_filtered
        else:
            global_logger.debug(f"No idx_to_keep given. Returning unfiltered data.")
            return data

    @staticmethod
    def force_2d(data: np.ndarray, transepose=True):
        data_2d = data
        if data_2d.ndim == 1:
            data_2d = data_2d.reshape(1, -1)
            if transepose:
                data_2d = data_2d.T
        elif data_2d.ndim > 2:
            raise ValueError("Data has more than 2 dimensions.")
        return data_2d

    @staticmethod
    def filter_shuffle_split(
        data: np.ndarray,
        idx_to_keep: Optional[np.ndarray] = None,
        shuffle: bool = False,
        split_ratio: Optional[float] = None,
        idx_train: Optional[List[int]] = None,
        idx_test: Optional[List[int]] = None,
        split_method: Literal["random", "sequential"] = "sequential",
        return_ids: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter, shuffle, and split the data into training and testing sets.
        """
        if idx_train is None or idx_test is None:
            global_logger.debug(
                f"No idx_train or idx_test given. Splitting data based on idx_to_keep and split_ratio of {split_ratio:.0%}."
            )
            data, idx_to_keep = force_equal_dimensions(data, idx_to_keep)
            if idx_to_keep is not None:
                idx_filtered = np.where(idx_to_keep == True)[0]
                global_logger.debug(
                    f"Filtering data by idx_to_keep. {len(idx_filtered)} indices to keep."
                )
            else:
                idx_filtered = np.arange(data.shape[0])
                global_logger.debug(
                    f"No idx_to_keep given. Keeping all {data.shape[0]} indices."
                )
            idx_shuffled = Dataset.shuffle(idx_filtered) if shuffle else idx_filtered

            # get the indices for train and test
            idx_train, idx_test = Dataset.split(
                data=idx_shuffled,
                split_ratio=split_ratio,
                method=split_method,
            )
        else:
            global_logger.debug(
                f"Using provided idx_train and idx_test. {len(idx_train)} training and {len(idx_test)} testing."
            )
            if len(idx_train) + len(idx_test) > data.shape[0]:
                do_critical(
                    ValueError,
                    f"idx_train and idx_test should be equal to the number of data points. idx_train: {len(idx_train)}, idx_test: {len(idx_test)}, data: {data.shape[0]}",
                )
            if shuffle:
                idx_train = Dataset.shuffle(idx_train)
                idx_test = Dataset.shuffle(idx_test)

        data_train = Dataset.filter_by_idx(data, idx_to_keep=idx_train)
        data_test = Dataset.filter_by_idx(data, idx_to_keep=idx_test)

        data_train = Dataset.force_2d(data_train)
        data_test = Dataset.force_2d(data_test)
        if return_ids:
            return data_train, data_test, idx_train, idx_test
        return data_train, data_test

    @staticmethod
    def manipulate_data(
        data,
        idx_to_keep=None,
        shuffle=False,
        split_ratio=None,
        idx_train: List[int] = None,
        idx_test: List[int] = None,
        split_method: Literal["random", "sequential"] = "sequential",
        as_list=False,
        return_ids=False,
    ):
        """
        Manipulates the input data by applying filtering, shuffling, splitting,
        and enforcing a 2D structure on the resulting subsets.

        Parameters:
        -----------
        data : list or any type
            The input data to be manipulated. If not a list, the data is first converted into a list.

        idx_to_keep : list or None, optional
            A list of indices to filter the data. Only data at these indices will be retained.
            If None, no filtering is applied. Default is None.

        shuffle : bool, optional
            If True, the filtered data will be shuffled before splitting.
            If False, the data remains in its original order. Default is False.

        split_ratio : float or None, optional
            The ratio used to split the data into training and testing subsets.
            If None, no splitting is applied and the entire dataset is considered training data. Default is None.

        Returns:
        --------
        data_train : list
            A list containing the training subsets of the input data.
            Each subset has been filtered, shuffled (if requested), split, and enforced to be 2D.

        data_test : list
            A list containing the testing subsets of the input data.
            Each subset has been filtered, shuffled (if requested), split, and enforced to be 2D.

        Notes:
        ------
        - The function assumes the existence of the `Dataset` class with the following methods:
            - `filter_by_idx(data, idx_to_keep)`: Filters data by keeping only the specified indices.
            - `shuffle(data)`: Shuffles the input data.
            - `split(data, split_ratio)`: Splits the data into training and testing subsets based on the `split_ratio`.
            - `force_2d(data)`: Ensures that the data has a 2D structure.
        """
        if as_list:
            if not is_list_of_ndarrays(data):
                data = [data]
            data_train = []
            data_test = []
            train_ids = []
            test_ids = []
            for i, data_i in enumerate(data):
                out = Dataset.filter_shuffle_split(
                    data=data_i,
                    idx_to_keep=idx_to_keep,
                    shuffle=shuffle,
                    split_ratio=split_ratio,
                    idx_train=idx_train,
                    idx_test=idx_test,
                    split_method=split_method,
                    return_ids=return_ids,
                )
                data_train_i, data_test_i = out[:2]
                if return_ids:
                    train_ids.append(out[2])
                    test_ids.append(out[3])
                data_train.append(data_train_i)
                data_test.append(data_test_i)
        else:
            out = Dataset.filter_shuffle_split(
                data=data,
                idx_to_keep=idx_to_keep,
                shuffle=shuffle,
                split_ratio=split_ratio,
                idx_train=idx_train,
                idx_test=idx_test,
                split_method=split_method,
                return_ids=return_ids,
            )
            data_train, data_test = out[:2]
            if return_ids:
                train_ids = out[2]
                test_ids = out[3]

        if return_ids:
            return data_train, data_test, train_ids, test_ids
        return data_train, data_test


class BehaviorDataset(Dataset):
    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.plot_attributes["fps"] = (
            self.metadata["imaging_fps"]
            if "imaging_fps" in self.metadata.keys()
            else self.metadata["fps"]
        )
        self.plot_attributes["fps"] = (
            float(self.plot_attributes["fps"]) if self.plot_attributes["fps"] else None
        )
        # default binning size is 1cm
        self.binning_size = None
        self.max_bin = None

    def get_transdata(self, transformation: Literal["relative", "binned"] = None):
        """
        Get the transdata of the data.

        Parameters
        ----------
        transformation : str
            The transformation to apply to the data.
            Options are:
                - "relative": Normalize the data to the environment dimensions from 0 to 1.
                            Self.data is expected to be in the form of [x, y] or [x, y, z].

                            Returns:
                            --------
                            relative : np.ndarray
                                The relative data.

        Returns
        -------
            transdata : np.ndarray
                The transformed data.
        """
        if transformation is None:
            data = self.data
        elif transformation == "relative":
            data = Environment.to_relative(self.data)
        elif transformation == "binned":
            data = self.binned_data
        else:
            raise ValueError(f"Transformation {transformation} not supported.")

        return data

    def get_setup(self, setup_name):
        self.setup_name = self.metadata["setup"]
        if setup_name == "active_avoidance":
            setup = Active_Avoidance(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "openfield":
            setup = Openfield_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "treadmill":
            setup = Treadmill_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "trackball":
            setup = Trackball_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "wheel":
            setup = Wheel_Setup(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Behavior setup {setup_name} not supported.")
        return setup


class NeuralDataset(Dataset):
    needed_keys = ["method", "preprocessing", "processing", "setup"]

    def __init__(
        self,
        key,
        path=None,
        data=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key=key,
            path=path,
            data=data,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        check_needed_keys(metadata, NeuralDataset.needed_keys)

    def create_dataset(self, raw_data_object=None, save=True):
        self.unfiltered_data = self.process_raw_data(save=save)
        return self.unfiltered_data

    def get_process_data(
        self, type: str = "processed", fit_to_shape: Union[None, Tuple[int, int]] = None
    ):
        """
        Get the data of the task.

        The data is processed and has 1 dimension larger to create the format of [time, features].

        Parameters
        ----------
        type : str, optional
            The type of data to get (default is "processed"). Other options could be available, depending on the function of setup.preprocess.process.data
            Options are:
                - "processed": the processed data
                - "unprocessed": the unprocessed data

        fit_to_shape : tuple, optional
            The shape to fit the data to. Default is None, which does not change the shape of the data.

        Returns
        -------
            data : np.ndarray
                The data of the task with 1 dimension larger than the number of features.
        """
        data = self.setup.preprocess.process.data(type=type)
        data = force_1_dim_larger(data)
        if fit_to_shape:
            empty_data = np.zeros(fit_to_shape)
            data, _ = force_equal_dimensions(data, empty_data)
        return data


class Data_Position(BehaviorDataset):
    """
    Dataset class for managing position data.
    Attributes:
        - data (np.ndarray): The position data.
        - binned_data (np.ndarray): The binned position data into default: 1cm bins.
        - raw_position (np.ndarray): The raw position data.
        - lap_starts (np.ndarray): The indices of the lap starts.
        - environment_dimensions (List): The dimensions of the environment.
    """

    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="position",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.max_bin = None
        self.raw_position = None
        self.lap_starts = None

    def refine_metadata(self):
        if self.metadata["environment_dimensions"] is None:
            if self.data is not None:
                global_logger.warning(
                    f"No Environmental Data is provided. Defining environment dimensions based on position data."
                )
                self.metadata["environment_dimensions"] = (
                    Environment.define_border_by_pos(self.data, map=False)
                )
        elif self.metadata["environment_dimensions"] is not None:
            self.metadata["environment_dimensions"] = make_list_ifnot(
                self.metadata["environment_dimensions"]
            )

        if not "binning_size" in self.metadata.keys():
            if self.metadata["environment_dimensions"] is not None:
                self.binning_size = self.gues_binning_size()
        else:
            self.binning_size = self.metadata["binning_size"]

        self.define_plot_attributes()

    def gues_binning_size(self):
        environment_dimensions = len(self.metadata["environment_dimensions"])
        # 1D environment
        if environment_dimensions == 1:
            return 0.01
        # 2D environment
        elif environment_dimensions == 2:
            return 0.05
        else:
            return 0.01

    def define_plot_attributes(self):
        self.plot_attributes["figsize"] = (20, 5)
        self.plot_attributes["cmap"] = "rainbow"
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 1:
                self.plot_attributes["ylable"] = "position m"
            elif len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)

    def plot_data(self):
        if self.data.shape[1] == 1:
            marker = "^" if self.lap_starts is not None else None
            Vizualizer.data_plot_1D(
                data=self.data,
                plot_attributes=self.plot_attributes,
                marker_pos=self.lap_starts,
                marker=marker,
            )
        elif self.data.shape[1] == 2:
            Vizualizer.data_plot_2D(
                data=self.data,
                position_data=self.data,
                border_limits=self.metadata.get("environment_dimensions"),
                plot_attributes=self.plot_attributes,
                color_by="time",
            )

    def animate_data(
        self,
        data=None,
        fps=None,
    ):
        """
        Animate the position data in 2D.
        """
        from Visualizer import animate_2D_positions

        raw_pos = self.process_raw_data() if data is None else data
        # reverse y axis for correct orientation
        # raw_pos[:, 1] = -raw_pos[:, 1]
        # raw_pos[:, 0] = -raw_pos[:, 0]
        # rot_pos = self.rotate_position_data(raw_pos)

        # Test the function
        print("Starting animation...")
        fps = 20
        # Convert FPS to milliseconds
        fps = self.metadata.get("fps", 1) if fps is None else fps
        interval = 1000 / fps
        task_id = self.metadata.get("task_id", "unknown_task")
        ani = animate_2D_positions(
            raw_pos,
            interval=interval,
            point_size=10,
            trail=True,
            title=f"{task_id} Position Animation - Rainbow Time Visualization",
            fps=fps,
        )
        return ani

    def correct_data(self, data):
        """
        Correct the position data by rotating it based on the cue card location.
        """
        data_2d = Dataset.force_2d(data)
        if "cue_card_location" in self.metadata.keys():
            data_2d = self.rotate_position_data(data_2d)
        return data_2d

    def rotate_position_data(self, data=None):
        """
        Rotate the position data by 90 degrees.
        This is useful for visualizing the data in a 2D environment.
        """
        if "cue_card_location" in self.metadata.keys():
            direction = self.metadata["cue_card_location"]
            global_logger.info(
                f"Rotating position data based on cue card location: {direction}"
            )
            # rotate the matrix so cue card location points to the right
            if direction == "left":
                k = 2
            elif direction == "right":
                k = 0
            elif direction == "up":
                k = 1
            elif direction == "bottom":
                k = 3
            else:
                raise ValueError(f"Unknown cue card location: {direction}")

        if data is None:
            data = self.data
        if data.shape[1] != 2:
            raise ValueError("Position data must be 2D.")
        # Create a rotation matrix for 90 degrees
        rotation_matrix = np.array([[0, -1], [1, 0]])
        # Rotate the data by multiplying with the rotation matrix
        rotated_data = data
        for _ in range(k):
            rotated_data = rotated_data @ rotation_matrix
        return rotated_data

    def define_binsize_minbins_maxbins(
        self, data=None, bin_size=None, min_bins=None, max_bins=None
    ):
        """
        Define the bin size, min bins and max bins for the position data.

        Args:
            - bin_size (float): The size of the bins in meters.
            - min_bins (int): The minimum number of bins.
            - max_bins (int): The maximum number of bins.

        Returns:
            - bin_size (float): The size of the bins in meters.
            - min_bins (int): The minimum number of bins.
            - max_bins (int): The maximum number of bins.
        """
        if data is None:
            data = self.data
        if bin_size is None:
            bin_size = self.binning_size
        dimensions = self.metadata["environment_dimensions"]
        if len(dimensions) == 1:
            if min_bins is None:
                min_bins = min_bins or 0
            if max_bins is None:
                max_bins = max_bins or dimensions
        elif len(dimensions) == 2:
            borders = Environment.define_border_by_pos(data)
            if min_bins is None:
                min_bins = borders[:, 0]
            if max_bins is None:
                max_bins = borders[:, 1]
        return bin_size, min_bins, max_bins

    def bin_data(self, data=None, bin_size=None, min_bins=None, max_bins=None):
        """
        Bin the position data into 1cm bins in 1D and 5cm^2 bins for 2D environments.
        Args:
            - data (np.ndarray): The position data.
            - bin_size (float): The size of the bins in meters.
        return: binned_data (np.ndarray): The binned position data into cm bins (default 1cm).
        """
        if data is None:
            data = self.data
        bin_size, min_bins, max_bins = self.define_binsize_minbins_maxbins(
            data=data, bin_size=bin_size, min_bins=min_bins, max_bins=max_bins
        )
        binned_data = bin_array(
            data, bin_size=bin_size, min_bin=min_bins, max_bin=max_bins
        )
        dimensions = self.metadata["environment_dimensions"]
        max_bin = np.ceil(np.array(dimensions) / np.array(bin_size)).astype(int)
        # convert to 1d if 2d
        if len(max_bin.shape) != 1:
            max_bin = max_bin[:, 1]
            raise ValueError(
                "!!!! check why !!!!. Only 1D and 2D environments are supported."
            )
        self.max_bin = max_bin
        # create category map based on max_bin
        if len(max_bin) == 1:
            # 1D environment
            category_map = {i: i for i in range(max_bin[0])}
        elif len(max_bin) == 2:
            # 2D environment
            category_map = {
                (i, j): i * max_bin[1] + j
                for i in range(max_bin[0])
                for j in range(max_bin[1])
            }
        else:
            raise ValueError(
                f"Only 1D and 2D environments are supported. Environment dimensions: {len(max_bin)}"
            )
        encoded_data, category_map = encode_categorical(
            binned_data, category_map=category_map
        )
        return encoded_data, category_map

    def create_dataset(self, raw_data_object=None, save=True):
        data = self.process_raw_data(save=save)
        return data

    def process_raw_data(self, save=True, smooth=True):
        self.raw_position = self.setup.process_data(save=save)
        smoothed_data = may_butter_lowpass_filter(
            self.raw_position,
            smooth=True,
            fps=self.metadata["imaging_fps"],
            order=2,
        )
        self.unfiltered_data = smoothed_data
        return self.unfiltered_data

    def get_lap_starts(self, fps=None, sec_thr=5):
        """
        Get the indices of the lap starts in the position data only tested for 1D data.
        """
        if self.lap_starts and sec_thr == 5:
            return self.lap_starts
        fps = fps or self.metadata["imaging_fps"]
        # get index of beeing at lap start
        encoded_data, category_map = self.bin_data()
        at_start_indices = np.where(encoded_data == 0)[0]
        # take first indices in case multiple frames are at start
        num_frames_threshold = fps * sec_thr
        start_indices = [at_start_indices[0]]
        old_idx = at_start_indices[0]
        for index in at_start_indices[1:]:
            if index - old_idx > num_frames_threshold:
                start_indices.append(index)
                old_idx = index
        return np.array(start_indices)

    def categorical_to_bin_position(
        self,
        values: Union[Dict[tuple, Any], np.ndarray],
        additional_title="",
        figsize=(7, 5),
        plot=True,
        save_dir=None,
        as_pdf=False,
    ):
        """
        parameters:
            values: dict or np.ndarray
                if dict: values are in dictionary with category as key
                if np.ndarray: values are already in 2D
        """
        # check if category map is 2D
        if self.category_map is None:
            raise ValueError("No category map found. Please bin data first.")

        values_array = np.full(self.max_bin, np.nan)

        if isinstance(values, dict):  # if values are in dictionary with category as key
            key_example = list(values.keys())[0]
            if (
                isinstance(key_example, tuple) and len(key_example) == 2
            ):  # if key is already 2D positional coordinates
                for bin_pos, value in values.items():
                    values_array[bin_pos] = value
            else:
                category_map_np = np.array(list(self.category_map.keys()))
                category_map_dimensions = category_map_np.shape[1]
                if category_map_dimensions != 2:
                    err_msg = (
                        "Only 2D category maps are supported. For plotting values in 2D"
                    )
                    global_logger.critical(err_msg)
                    raise NotImplementedError(err_msg)

                uncode_values = uncode_categories(values, self.category_map)
                for bin_pos, value in uncode_values.items():
                    values_array[bin_pos] = value

        elif isinstance(values, np.ndarray):  # if values are already in 2D
            if values.ndim != 2:
                raise ValueError("Values should be 2D array. For plotting values in 2D")
            if values.shape != values_array.shape:
                raise ValueError(
                    f"Values shape {values.shape} should be equal to category map shape {values_array.shape}"
                )
            values_array = values

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            xticks_pos = list(range(self.max_bin[0]))
            yticks_pos = list(range(self.max_bin[1]))
            xticks = [i if i % 2 == 0 else "" for i in xticks_pos]
            yticks = [i if i % 2 == 0 else "" for i in yticks]
            title = f"Accuracy for every class {additional_title} {self.task_id}"
            Vizualizer.plot_heatmap(
                values_array,
                ax=ax,
                title=title,
                title_size=figsize[0] * 1.5,
                xticks=xticks,
                yticks=yticks,
                xticks_pos=xticks_pos,
                yticks_pos=yticks_pos,
                xlabel="X Bin",
                ylabel="Y Bin",
                colorbar=True,
            )
            # add colorbar
            plt.tight_layout()
            Vizualizer.save_plot(save_dir, title, "pdf" if as_pdf else "png")
            plt.show()

        return values_array


class Data_Stimulus(BehaviorDataset):
    optional_keys = [
        "stimulus_dimensions",
        "stimulus_sequence",
        "stimulus_type",
        "stimulus_by",
        "fps",
    ]

    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="stimulus",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )

    def relative_data(self):
        global_logger.warning(
            f"Relative data is not implemented for {self.__class__}. Returning self.data"
        )
        return self.data

    def define_optional_attributes(self):
        self.metadata = add_missing_keys(
            self.metadata, Data_Stimulus.optional_keys, fill_value=None
        )
        self.stimulus_sequence = self.metadata["stimulus_sequence"]
        self.stimulus_dimensions = self.metadata["stimulus_dimensions"]
        self.stimulus_type = self.metadata["stimulus_type"]
        self.plot_attributes["figsize"] = (20, 5)
        self.stimulus_by = self.metadata["stimulus_by"] or "location"
        self.fps = self.metadata["fps"] if "fps" in self.metadata.keys() else None

    def define_plot_attributes(self):
        self.define_optional_attributes()
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 1:
                self.plot_attributes["ylable"] = "position cm"
            elif len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)
        if self.stimulus_by == "location":
            shapes, stimulus_type_at_frame = Environment.get_env_shapes_from_pos()
            self.plot_attributes["yticks"] = [range(len(shapes)), shapes]

    def process_raw_data(self, save=True, stimulus_by="location"):
        """ "
        Returns:
            - data: Numpy array composed of stimulus type at frames.
        """
        stimulus_raw_data = (
            self.raw_data_object.unfiltered_data
        )  # e.g. Position on a track/time
        stimulus_by = self.stimulus_by or stimulus_by
        if stimulus_by == "location":
            if len(self.metadata["environment_dimensions"]) == 1:
                stimulus_type_at_frame = Environment.get_stimulus_at_position(
                    positions=stimulus_raw_data,
                    stimulus_dimensions=self.stimulus_dimensions,
                    stimulus_sequence=self.stimulus_sequence,
                    max_position=self.stimulus_dimensions,
                )
            elif len(self.metadata["environment_dimensions"]) == 2:
                shapes, stimulus_type_at_frame = Environment.get_env_shapes_from_pos(
                    stimulus_raw_data
                )
                self.plot_attributes["yticks"] = [range(len(shapes)), shapes]
        elif stimulus_by == "frames":
            stimulus_type_at_frame = self.stimulus_by_time(stimulus_raw_data)
        elif stimulus_by == "seconds":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=self.self.metadata["imaging_fps"],
            )
        elif stimulus_by == "minutes":
            stimulus_type_at_frame = self.stimulus_by_time(
                stimulus_raw_data,
                time_to_frame_multiplier=60 * self.self.metadata["imaging_fps"],
            )
        self.data = stimulus_type_at_frame
        return self.data

    def stimulus_by_time(self, stimulus_raw_data, time_to_frame_multiplier=1):
        """
        time in frames in an experiment
        """
        stimulus_type_at_frame = []
        for stimulus, duration in zip(self.stimulus_sequence, self.stimulus_dimensions):
            stimulus_type_at_frame += [stimulus] * int(
                duration * time_to_frame_multiplier
            )
        raise NotImplementedError("Stimulus by time may not implemented. Check results")
        return np.array(stimulus_type_at_frame)


class Data_Distance(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="distance",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.metadata["environment_dimensions"] = make_list_ifnot(
            self.metadata["environment_dimensions"]
        )
        self.binning_size = (
            0.01
            if not "binning_size" in self.metadata.keys()
            else self.metadata["binning_size"]
        )

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "distance in m"
        self.plot_attributes["figsize"] = (20, 5)

    def plot_data(self):
        absolute_distance = np.linalg.norm(self.data, axis=1)
        Vizualizer.data_plot_1D(
            data=absolute_distance,
            plot_attributes=self.plot_attributes,
        )

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data, category_map = encode_categorical(binned_data)
        return encoded_data, category_map

    def process_raw_data(self, smooth=True, save=True):
        track_positions = self.raw_data_object.unfiltered_data
        self.unfiltered_data = Environment.get_cumdist_from_position(
            track_positions, imaging_fps=self.metadata["imaging_fps"]
        )
        return self.unfiltered_data


class Data_Velocity(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="velocity",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_velocity = None
        self.binning_size = 0.005  # 0.005m/s

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "velocity m/s"
        self.plot_attributes["figsize"] = (20, 5)
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data, category_map = encode_categorical(binned_data)
        return encoded_data, category_map

    def process_raw_data(self, save=True, smooth=True):
        """
        calculating velocity based on velocity data in raw_data_object
        """
        raw_data_type = self.raw_data_object.key
        data = self.raw_data_object.unfiltered_data
        if raw_data_type == "distance":
            walked_distances = (
                self.raw_data_object.process_raw_data(save=save)
                if data is None
                else data
            )
        elif raw_data_type == "position":
            walked_distances = Environment.get_cumdist_from_position(data)
        else:
            raise ValueError(f"Raw data type {raw_data_type} not supported.")

        self.raw_velocity = Environment.get_velocity_from_cumdist(
            walked_distances, imaging_fps=self.metadata["imaging_fps"], smooth=False
        )
        velocity_smoothed = may_butter_lowpass_filter(
            self.raw_velocity,
            smooth=True,
            cutoff=None,
            fps=self.metadata["imaging_fps"],
            order=2,
        )

        if velocity_smoothed.shape[1] == 1:
            self.unfiltered_data = velocity_smoothed
        else:
            self.unfiltered_data = abs(velocity_smoothed)
        return self.unfiltered_data

    @property
    def euclidean(self, data=None):
        """
        Calculate the Euclidean velocity from the position data.
        If data is not provided, it uses self.data.
        """

        if data is None:
            data = self.data

        v_scalar = Environment.to_euclidean(data)
        return v_scalar


class Data_Acceleration(BehaviorDataset):
    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="acceleration",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.raw_acceleration = None
        self.binning_size = 0.0005  # 0.0005m/s^2

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "acceleration m/s^2"
        if self.metadata["environment_dimensions"] is not None:
            if len(self.metadata["environment_dimensions"]) == 2:
                self.plot_attributes["figsize"] = (12, 10)

    def bin_data(self, data, bin_size=None):
        bin_size = bin_size or self.binning_size
        binned_data = bin_array(data, bin_size=bin_size, min_bin=0)
        encoded_data, category_map = encode_categorical(binned_data)
        return encoded_data, category_map

    def process_raw_data(self, save=True):
        """
        calculating acceleration based on velocity data in raw_data_object
        """
        if self.raw_data_object.key != "velocity":
            raise ValueError(
                f"Raw data object should be of type 'velocity', but got {self.raw_data_object.key}."
            )
        velocity = self.raw_data_object.unfiltered_data
        self.raw_acceleration = Environment.get_acceleration_from_velocity(
            velocity, smooth=False
        )
        smoothed_acceleration = may_butter_lowpass_filter(
            self.raw_acceleration,
            cutoff=None,
            fps=self.metadata["imaging_fps"],
            order=2,
        )
        if smoothed_acceleration.shape[1] == 1:
            self.unfiltered_data = smoothed_acceleration
        else:
            self.unfiltered_data = abs(smoothed_acceleration)
        return self.unfiltered_data


class Data_Moving(BehaviorDataset):
    # seconds
    brain_processing_delay = {
        "CA1": 2,
        "CA3": 2,
        "MEC": 2,
        "M1": None,
        "S1": None,
        "V1": None,
    }

    def __init__(
        self, raw_data_object=None, metadata=None, root_dir=None, task_id=None
    ):
        super().__init__(
            key="moving",
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )
        self.speed_thr = 0.02  # m/s

    @property
    def brain_reset_time(self):
        """
        Get the reset time for the brain area.
        """
        if self.metadata["area"] in Data_Moving.brain_processing_delay.keys():
            reset_time = Data_Moving.brain_processing_delay[self.metadata["area"]]
        else:
            global_logger.warning(
                f"Brain area {self.metadata['area']} not found. No movement reset time set."
            )
            reset_time = None
        return reset_time

    def plot_data(self):
        Vizualizer.data_plot_1D(
            data=self.data,
            plot_attributes=self.plot_attributes,
        )
        if self.raw_data_object is not None:
            # scale velocity to fit in the plot by scaling speed between 0 and 1
            velocity = Environment.to_euclidean(self.raw_data_object.data)
            v_max = np.max(velocity)
            scaled_velocity = velocity / v_max
            plt.plot(
                scaled_velocity,
                color="black",
                alpha=0.5,
                linewidth=1,
                label="velocity (m/s) scaled",
            )
            # add yaxis on left for velocity and xticks fitted to velocity
            plt.ylabel("Velocity m/s")
            ytick_num = 5
            yticks_pos = np.arange(0, 1.01, 1 / ytick_num)
            yticks = [round(v_max * y, 2) for y in yticks_pos]
            plt.yticks(yticks_pos, yticks)
            # add red dashed line at speed threshold
            thr = self.speed_thr / v_max
            plt.axhline(
                y=thr,
                color="red",
                linestyle="--",
                label=f"speed thr: {self.speed_thr} m/s",
            )

        plt.legend(loc="upper right")

    def relative_data(self):
        global_logger.warning(
            f"Relative data is not implemented for {self.__class__}. Returning self.data"
        )
        return self.data

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Movement State"
        self.plot_attributes["ylimits"] = (0, 1)
        self.plot_attributes["figsize"] = (20, 5)
        self.plot_attributes["yticks"] = []

    def process_raw_data(
        self,
        save: bool = True,
        speed_thr: float = None,
        fit_to_brainarea: bool = True,
    ):
        velocities = self.raw_data_object.unfiltered_data
        if velocities is None:
            velocities = self.raw_data_object.process_raw_data(save=False)
        reset_time = self.brain_reset_time if fit_to_brainarea else None
        if reset_time is None:
            global_logger.error(
                f"Moving data not fitted to brain area {self.metadata['area']} lag"
            )

        if speed_thr is not None and speed_thr != self.speed_thr:
            global_logger.warning(
                f"Using custom speed threshold {speed_thr} m/s instead of default {self.speed_thr} m/s"
            )
        else:
            speed_thr = self.speed_thr

        moving_frames = Environment.get_moving_from_velocity(
            velocities,
            reset_time=reset_time,
            thr=speed_thr,
            fps=self.metadata["imaging_fps"],
        )
        self.unfiltered_data = moving_frames
        return self.unfiltered_data

    def get_idx_to_keep(self, movement_state):
        if movement_state == "all":
            idx_to_keep = None
        elif movement_state == "moving":
            idx_to_keep = self.data
        elif movement_state == "stationary":
            idx_to_keep = self.data == False
        else:
            raise ValueError(f"Movement state {movement_state} not supported.")
        return idx_to_keep


class Data_Photon(NeuralDataset):
    """
    Dataset class for managing photon data.
    Attributes:
        - data (np.ndarray): The binariced traces.
        - method (str): The method used to record the data.
        - preprocessing_software (str): The software used to preprocess the data.
            - raw fluorescence traces can be found inside preprocessing object
        - setup (str): The setup used to record the data.
        - setup.data = raw_data_object
    """

    def __init__(
        self,
        path=None,
        raw_data_object=None,
        metadata=None,
        root_dir=None,
        task_id=None,
    ):
        super().__init__(
            key="photon",
            path=path,
            raw_data_object=raw_data_object,
            metadata=metadata,
            root_dir=root_dir,
            task_id=task_id,
        )

    def define_plot_attributes(self):
        self.plot_attributes["ylable"] = "Neuron ID"
        self.plot_attributes["figsize"] = (20, 10)
        if "fps" not in self.metadata.keys():
            self.metadata["fps"] = self.setup.get_fps()
            self.plot_attributes["fps"] = self.metadata["fps"]

    def process_raw_data(self, save=True):
        self.unfiltered_data = self.setup.process_data(save=save)
        return self.unfiltered_data

    def plot_data(self):
        Vizualizer.plot_neural_activity_raster(
            self.data,
            fps=self.fps,
            num_ticks=self.plot_attributes["num_ticks"],
        )

    def get_setup(self, setup_name):
        if setup_name == "femtonics":
            setup = Femtonics(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "thorlabs":
            setup = Thorlabs(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        elif setup_name == "inscopix":
            setup = Inscopix(
                key=self.key,
                root_dir=self.root_dir,
                metadata=self.metadata,
            )
        else:
            raise ValueError(f"Imaging setup {setup_name} not supported.")
        return setup


class Data_Probe(Dataset):  # Maybe NeuralDataset
    def __init__(self, path=None, raw_data_object=None, metadata=None):
        super().__init__(
            key="probe", path=path, raw_data_object=raw_data_object, metadata=metadata
        )
        # TODO: implement probe data loading
        raise NotImplementedError("Probe data loading not implemented.")


class Datasets:
    """
    Dataset class for managing multiple datasets. Getting correct data paths and loading data.
    """

    def __init__(self, root_dir, metadata={}, task_id=None):
        if "data_dir" in metadata.keys():
            self.data_dir = Path(root_dir).joinpath(metadata["data_dir"])
        else:
            self.data_dir = Path(root_dir)
        self.metadata = metadata
        self.task_id = task_id
        self.data_filter: DataFilter = None

    def get_object(self, data_source):
        raise NotImplementedError(
            f"ERROR: Function get_object is not defined for {self.__class__}."
        )

    def load(
        self,
        data_source,
        data_filter: DataFilter = None,
        regenerate=False,
        regenerate_plot=False,
        plot=None,
    ):
        """
        Load data from a specific data source.
            data_object is a object inhereting the Dataset attributes and functions.
        """
        check_obj_equivalence(self.data_filter, data_filter)
        # build path to data
        data_object = self.get_object(data_source)
        fpath = data_object.setup.get_data_path()
        data = data_object.load(
            fpath,
            regenerate=regenerate,
            regenerate_plot=regenerate_plot,
            plot=plot,
            data_filter=data_filter,
        )
        return data

    @property
    def class_sources(self):
        """
        Get the class defined Sources of the current object.
        This is used to get the sources of the datasets.
        For example, Datasets_Neural.Sources will return the sources of the neural datasets
        """
        return self.__class__.Sources

    @property
    def include_frames(self):
        """
        Get the include frames for all the datasets.
        If not defined, it will return None.
        """
        include_frames = self.define_include_frames()
        return include_frames

    def define_include_frames(self, include_frames: Union[np.ndarray, List] = None):
        """
        Define the include frames for all the datasets.


        All dataset.include_frames will be used to define the include frames.
        Parameters:
            - include_frames (Union[np.ndarray[bool], List[bool]]): The include frames to set.
                If None, all frames are used as the base to define the include frames.
        """
        global_include_frames = include_frames
        for data_source in self.class_sources:
            dataset_obj = self.get_object(data_source)
            if dataset_obj is None:
                continue
            obj_include_frames = dataset_obj.include_frames
            if obj_include_frames is None:
                continue
            global_logger.debug(
                f"Dataset {data_source} has frames to include {obj_include_frames.sum()}"
            )

            if global_include_frames is None:
                global_include_frames = obj_include_frames
            else:
                global_include_frames = logical_and_extend(
                    global_include_frames, obj_include_frames
                )
        return global_include_frames

    def get_multi_data(
        self,
        sources: List[str],
        shuffle: bool = False,
        idx_to_keep: Union[List, np.ndarray] = None,
        split_ratio: float = 1,
        transformation: Literal["binned", "relative"] = None,
        split_method: Literal["random", "sequential"] = "sequential",
    ):
        """
        Extract data from multiple sources and concatenate them.

        Parameters:
            - sources (List[str]): The sources to extract data from.
            - shuffle (bool): Shuffle the concatenated data.
            - idx_to_keep (Union[List, np.ndarray]): Indices to keep.
            - split_ratio (float): Ratio to split the data into train and test.
            - transformation (str): Transformation to apply to the data.
                transformation types are: "binned", "relative", None.

        Returns:
            - concatenated_data_tain (np.ndarray): The concatenated training data.
            - concatenated_data_test (np.ndarray): The concatenated test data.
        """
        sources = make_list_ifnot(sources)
        concatenated_data = None
        for source in sources:
            global_logger.info(f"Loading data from {source}.")
            dataset_object = getattr(self, source)
            if dataset_object.data is None:
                global_logger.warning(
                    f"Data from {source} is None. Loading for {dataset_object.key}."
                )
                return None, None
            data = dataset_object.get_transdata(transformation=transformation)
            data = np.array([data]).transpose() if len(data.shape) == 1 else data
            if type(concatenated_data) != np.ndarray:
                concatenated_data = data
            else:
                concatenated_data, data = force_equal_dimensions(
                    concatenated_data, data
                )
                concatenated_data = np.concatenate([concatenated_data, data], axis=1)

        conc_data_train, conc_data_test = Dataset.filter_shuffle_split(
            concatenated_data,
            idx_to_keep=idx_to_keep,
            shuffle=shuffle,
            split_ratio=split_ratio,
            split_method=split_method,
        )
        return conc_data_train, conc_data_test


class Datasets_Neural(Datasets):
    Photon_imaging_methods = ["femtonics", "thorlabs", "inscopix"]
    Probe_imaging_methods = ["neuropixels", "tetrode"]
    Sources = Photon_imaging_methods + Probe_imaging_methods

    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        # TODO: split into different datasets if needed
        self.imaging_type = self.define_imaging_type(self.metadata["setup"])
        if self.imaging_type == "photon":
            self.photon = Data_Photon(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )
        else:
            # TODO: implement probe data loading
            self.probe = Data_Probe(
                root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
            )

        if "fps" not in self.metadata.keys():
            # TODO: this is not implemented for probe data
            data_object = self.get_object(self.metadata["setup"])  # photon or probe
            self.metadata["fps"] = data_object.metadata["fps"]

    def define_imaging_type(self, data_source):
        if data_source in Datasets_Neural.Photon_imaging_methods:
            imaging_type = "photon"
        elif data_source in Datasets_Neural.Probe_imaging_methods:
            imaging_type = "probe"
        else:
            raise ValueError(f"Imaging type {data_source} not supported.")
        return imaging_type

    def get_object(
        self, data_source: Union[None, str] = None
    ) -> Union[Data_Photon, Data_Probe]:
        """
        Get the data object based on the data source.

        WARNING: The implementation only handles on type of neural imaging method, so multiple data sources are not supported yet.

        Args:
            data_source (str, optional): One data source with a name from the list self.photon_imaging_methods or self.probe_imaging_methods
                to get the data object from. If None, the already defined imaging type is used.

        Returns:
            data_object (Data_Photon or Data_Probe): The data object based on the data source
        """
        imaging_type = (
            self.imaging_type
            if data_source is None
            else self.define_imaging_type(data_source)
        )
        data_object = getattr(self, imaging_type, None)
        return data_object

    def get_process_data(
        self, data_source: Union[None, str] = None, type: str = "processed"
    ) -> np.ndarray:
        """
        Get the data from process object based on the data source.

        WARNING: The implementation only handles on type of neural imaging method, so multiple data sources are not supported yet.

        Args:
            data_source (str, optional): One data source with a name from the list self.photon_imaging_methods or self.probe_imaging_methods
                to get the data object from. If None, the already defined imaging type is used.

        Returns:
            data (np.ndarray): The processed data
        """
        data_object = self.get_object(data_source)
        data = data_object.get_process_data(type=type)
        return data


class Datasets_Behavior(Datasets):
    Sources = ["position", "distance", "stimulus", "velocity", "acceleration", "moving"]

    def __init__(self, root_dir, metadata={}, task_id=None):
        super().__init__(root_dir=root_dir, metadata=metadata, task_id=task_id)
        self.data_dir = (
            self.data_dir
            if not self.data_dir == root_dir
            else self.data_dir.joinpath(f"TRD-{self.metadata['method']}")
        )
        self.position = Data_Position(
            root_dir=root_dir, metadata=self.metadata, task_id=self.task_id
        )
        self.distance = Data_Distance(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.stimulus = Data_Stimulus(
            raw_data_object=self.position,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.velocity = Data_Velocity(
            raw_data_object=self.distance,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.acceleration = Data_Acceleration(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )
        self.moving = Data_Moving(
            raw_data_object=self.velocity,
            root_dir=root_dir,
            metadata=self.metadata,
            task_id=self.task_id,
        )

    def split_by_laps(self, data=None, lap_starts=None, fps=None, sec_thr=5):
        """
        Splits data into individual laps based on lap start indices or calculates lap start indices using the provided parameters.

        Parameters:
            - data (optional): The data to be split into laps. This can be either 1D or 2D data.
            - lap_starts (optional): Indices indicating the start of each lap in the data.
            - fps (optional): Frames per second. Required if lap_starts is not provided and data is in video format.
            - sec_thr (optional): Threshold in seconds to consider as the minimum lap duration when calculating lap starts.

        Returns:
            - laps: A list containing each individual lap of the data.

        If lap_starts is not provided, lap start indices are calculated using the given fps and sec_thr parameters.
        If data is not provided, it will be expected to be accessed from the instance attribute 'position'.

        Note:
        - 1D data: If the data is 1-dimensional (e.g., time series data), lap_starts must be provided.
        - 2D data: If the data is 2-dimensional (e.g., time series data for multiple neurons).
        ```
        """
        lap_starts = lap_starts or self.position.get_lap_starts(
            fps=fps, sec_thr=sec_thr
        )
        laps = []
        for start, end in zip(lap_starts[:-1], lap_starts[1:]):
            laps.append(data[start:end])
        return laps

    def get_object(self, data_source):
        data_object = getattr(self, data_source)
        return data_object

    def occupancy_by_binned_feature(
        self,
        data=None,
        group_values="count_norm",
        idx_to_keep: Union[str, np.ndarray] = None,
        plot=False,
        additional_title="",
        figsize=(6, 5),
        xticks=None,
        xticks_pos=None,
        yticks=None,
        ylabel="",
        save_dir=None,
    ):

        data = data if data is not None else self.position.binned_data
        if isinstance(idx_to_keep, str):
            if idx_to_keep == "moving":
                idx_to_keep = self.moving.data
            elif idx_to_keep == "stationary":
                idx_to_keep = self.moving.data == False

        filtered_data = self.position.filter_by_idx(data, idx_to_keep)
        if filtered_data is None or len(filtered_data) == 0:
            global_logger.error(f"Filtered data is None or empty. Returning None.")
            return None
        if "max_bin" not in self.position.__dict__.keys():
            self.position.max_bin = None
        occupancy, _ = group_by_binned_data(
            data=filtered_data,
            binned_data=filtered_data,
            category_map=self.position.category_map,
            group_values=group_values,
            max_bin=self.position.max_bin,
            as_array=True,
        )

        occupancy[occupancy == 0] = np.nan
        if plot:
            title = (
                f"Occupancy Map {self.position.metadata['task_id']} {additional_title}"
            )
            save_dir = save_dir or self.position.path.parent.parent.joinpath("figures")

            if self.position.key == "stimulus":
                xticks = xticks or self.position.plot_attributes["yticks"][1]
                xticks_pos = xticks_pos or self.position.plot_attributes["yticks"][0]
            else:
                unique_bins = np.array(list(self.position.category_map.keys()))
                xticks = np.unique(unique_bins[:, 0])
                yticks = np.unique(unique_bins[:, 1])
                xticks_pos = xticks
                yticks_pos = yticks

            # Plot occupancy map
            # y flip for correct orientation in heatmap
            Vizualizer.plot_heatmap(
                np.flip(occupancy, axis=0),
                figsize=figsize,
                title=title,
                xticks=xticks,
                xticks_pos=xticks_pos,
                yticks=yticks,
                yticks_pos=yticks_pos,
                xlabel=f"{self.position.key.capitalize()} bins",
                ylabel=ylabel,
                save_dir=save_dir,
            )

        return occupancy
