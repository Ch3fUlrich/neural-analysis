# type hints
from __future__ import annotations
import concurrent.futures
from typing import List, Union, Dict, Any, Tuple, Optional, Literal
from tqdm import tqdm, trange
import inspect

# paths
from pathlib import Path
from restructure import forbidden_names

# setups and preprocessing software
from Setups import *
from Helper import *
from Visualizer import *
from ModelCls import Models, PlaceCellDetectors
from Datasets import Datasets_Neural, Datasets_Behavior, Dataset
from restructure import naming_structure, forbidden_names
from Manimeasure import calc_shape_similarity, simple_embedd, load_df_sim
from calculations import get_auc, statistical_comparison

# calculations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch

# pip install binarize2pcalcium
# from binarize2pcalcium import binarize2pcalcium as binca

# set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def load_all_animals(
    root_dir,
    wanted_animal_ids=None,
    wanted_dates=None,
    wanted_tasks=None,
    model_settings=None,
    behavior_data_types=None,
    regenerate=False,
    regenerate_plots=False,
    plot=True,
    **kwargs,
):
    raise DeprecationWarning(
        f"load_all_animals is deprecated. Use Mother.get_children() instead."
    )


class Multi:
    def __init__(
        self,
        animals_dict,
        name=None,
        wanted_properties=None,
        model_settings=None,
        **kwargs,
    ):
        self.wanted_properties: Optional[Dict[str, Any]] = wanted_properties
        self.animals: Dict[str, Animal] = animals_dict
        self.filtered_tasks: Dict[str, Task] = (
            self.animals if not wanted_properties else self.filter()
        )
        self.model_settings: Dict[str, Any] = model_settings or kwargs
        self.name: str = self.define_name(name)
        self.id: str = self.define_id(self.name)
        self.model_dir: Path = self.animals[
            list(self.animals.keys())[0]
        ].dir.parent.joinpath("models")
        self.models: Models = self.init_models(model_settings=model_settings)

    def define_name(self, name):
        name = name if name else "UNDEFINED_NAME"
        name = f"multi_set_{name}"
        return name

    def define_id(self, name):
        id = f"{name}_{self.model_settings}_{self.wanted_properties}"
        remove_ilegal_chars = [" ", "[", "]", "{", "}", ":", ",", "'", "\\", "/"]
        for char in remove_ilegal_chars:
            id = id.replace(char, "")
        return id

    def filter(self, wanted_properties: Dict[str, Dict[str, Union[str, List]]] = None):
        """
        Filters the tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {  # "animal": {
                    "animal_id": ["DON-007021"],
                    "sex": "male",
                    },
                    "session": {
                        # "date": ["20211022", "20211030", "20211031"],
                    },
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            if self.wanted_properties:
                wanted_properties = self.wanted_properties
            else:
                print("No wanted properties given. Returning all tasks")
                wanted_properties = {}
        else:
            if self.wanted_properties == wanted_properties:
                return self.filtered_tasks
            else:
                self.wanted_properties = wanted_properties

        filtered_tasks = {}
        for animal_id, animal in self.animals.items():
            wanted = True
            if "animal" in wanted_properties:
                wanted = wanted_object(animal, wanted_properties["animal"])

            if wanted:
                filtered_animal_tasks = animal.filter_sessions(wanted_properties)
                filtered_tasks.update(filtered_animal_tasks)
        self.filtered_tasks = filtered_tasks
        return self.filtered_tasks

    def init_models(self, model_settings=None, **kwargs):
        if not model_settings:
            model_settings = kwargs
        models = Models(
            self.model_dir, model_id=self.name, model_settings=model_settings
        )
        return models

    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        tasks=None,
        wanted_properties=None,
        regenerate: bool = False,
        shuffle: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_name: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        binned: bool = True,
        relative: bool = False,
        neural_data_types: List[str] = None,  #
        behavior_data_types: List[str] = None,  # ["position"],
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
        create_embeddings: bool = True,
    ):
        if not tasks:
            tasks = self.filter(wanted_properties)

        datas = []
        labels = []
        for task_id, task in tasks.items():
            # get neural data
            idx_to_keep = task.behavior.moving.get_idx_to_keep(movement_state)
            neural_data, _ = task.neural.get_multi_data(
                sources=task.neural.imaging_type,
                idx_to_keep=idx_to_keep,
                binned=binned,
            )

            # get behavior data
            if behavior_data_types:
                behavior_data, _ = task.behavior.get_multi_data(
                    sources=behavior_data_types,  # e.g. ["position", "stimulus"]
                    idx_to_keep=idx_to_keep,
                    binned=binned,
                )

            datas.append(neural_data)
            labels.append(behavior_data)

        print(self.id)
        multi_model = self.models.train_model(
            neural_data=datas,
            behavior_data=labels,
            model_type=model_type,
            name_comment=model_name,
            movement_state=movement_state,
            shuffle=shuffle,
            binned=binned,
            relative=relative,
            split_ratio=split_ratio,
            model_settings=model_settings,
            pipeline=manifolds_pipeline,
            create_embeddings=create_embeddings,
            regenerate=regenerate,
        )

        return multi_model

    def plot_embeddings(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        to_2d: bool = False,
        show_hulls: bool = False,
        to_transform_data: Optional[np.ndarray] = None,
        colorbar_ticks: Optional[List] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        # FIXME: This function is outdated
        # FIXME: merge this function with tasks plot_embeddings

        # models = self.get_pipeline_models(manifolds_pipeline, model_naming_filter_include, model_naming_filter_exclude)
        if embeddings is not None and to_transform_data is not None:
            global_logger.critical(
                "Either provide embeddings or to_transform_data, not both."
            )
            raise ValueError(
                "Either provide embeddings or to_transform_data, not both."
            )
        if to_transform_data is not None:
            embeddings = self.models.create_embeddings(
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline="cebra",
            )

        if not embeddings:
            model_class, models = self.models.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
            embeddings = {}
            labels_dict = {}
            for model_name, model in models.items():
                embeddings[model_name] = model.data["train"]["embedding"]
                labels_dict[model_name] = model.data["train"]["behavior"]

        # get embedding lables
        if not isinstance(labels, np.ndarray) and not isinstance(labels, dict):
            behavior_label = []
            labels_dict = {}
            all_embeddings = {}
            behavior_data_types = make_list_ifnot(behavior_data_types)

            # create labels for all behavior data types
            for behavior_data_type in behavior_data_types:
                models_embeddings = print("Dont know how this should look like...")
                for embedding_model_name, embeddings in models_embeddings.items():
                    global_logger.warning(
                        f"Using behavior_data_types: {behavior_data_types}"
                    )
                    print(f"Using behavior_data_types: {behavior_data_types}")

                    # extract behavior labels from corresponding task
                    for (task_id, task), (embedding_title, embedding) in zip(
                        self.filtered_tasks.items(), embeddings.items()
                    ):
                        task_behavior_labels_dict = task.get_behavior_labels(
                            [behavior_data_type], idx_to_keep=None
                        )
                        if not equal_number_entries(
                            embedding, task_behavior_labels_dict
                        ):
                            task_behavior_labels_dict = task.get_behavior_labels(
                                [behavior_data_type], movement_state="moving"
                            )
                            if not equal_number_entries(
                                embedding, task_behavior_labels_dict
                            ):
                                task_behavior_labels_dict = task.get_behavior_labels(
                                    [behavior_data_type], movement_state="stationary"
                                )
                                if not equal_number_entries(
                                    embedding, task_behavior_labels_dict
                                ):
                                    raise ValueError(
                                        f"Number of labels is not equal to all, moving or stationary number of frames."
                                    )
                        behavior_label.append(
                            task_behavior_labels_dict[behavior_data_type]
                        )
                    all_embeddings.update(embeddings)
                labels_dict[behavior_data_type] = behavior_label
        else:
            if isinstance(labels, np.ndarray):
                labels_dict = {"Provided_labels": labels}
            else:
                labels_dict = labels

        # get ticks
        if len(behavior_data_types) == 1 and colorbar_ticks is None:
            dataset_object = getattr(task.behavior, behavior_data_types[0])
            # TODO: ticks are not always equal for all tasks, so this is not a good solution
            colorbar_ticks = dataset_object.plot_attributes["yticks"]

        viz = Vizualizer(self.model_dir.parent)
        self.id = self.define_id(self.name)
        for embedding_title, labels in labels_dict.items():
            if title:
                title = title
            else:
                title = f"{manifolds_pipeline.upper()} embeddings {self.id}"
                descriptive_metadata_keys = [
                    "stimulus_type",
                    "method",
                    "processing_software",
                ]
                title += (
                    get_str_from_dict(
                        dictionary=task.behavior_metadata,
                        keys=descriptive_metadata_keys,
                    )
                    + f"{' '+str(title_comment) if title_comment else ''}"
                )
            projection = "2d" if to_2d else "3d"
            labels_dict = {"name": embedding_title, "labels": labels}
            viz.plot_multiple_embeddings(
                all_embeddings,
                labels=labels_dict,
                ticks=colorbar_ticks,
                title=title,
                projection=projection,
                show_hulls=show_hulls,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        return embeddings


class MetaClass:
    """Metaclass for any type of Animal related object."""

    plot_levels = ["Mother", "Animal", "Session", "Task", "Model"]

    def __init__(self, dir, data_filter={}):
        """
        Parameters
        ----------
        dir : str
            The root directory path where the animal folders are stored.

        Initializes the directories for figures and output.
        Creates the directories if they do not exist.
        Also initializes the data filter for different types of data.
        Attributes
        ----------
        dir : Path
            The root directory path where the animal folders are stored.
        figures_dir : Path
            The directory where the figures are stored.
        output_dir : Path
            The directory where some the output files are stored.
        data_filter : dict
            A dictionary to filter the data based on different types of data.
            The keys are the data types and the values are the filter criteria.
            currently supports:
            - "behavior": for filtering behavior data
                - "range": Tuple for filtering based on a range of values

        """
        self.dir = Path(dir)
        self.figures_dir = self.dir.joinpath("figures")
        self.figures_dir.mkdir(exist_ok=True)
        self.output_dir = self.dir.joinpath("output")
        self.output_dir.mkdir(exist_ok=True)
        self.data_filter = (
            data_filter
            if isinstance(data_filter, DataFilter)
            else DataFilter(data_filter)
        )
        # TODO: add more properties to the meta class and reduce the amount of properties in the child classes

    @property
    def random_child(self):
        """
        Returns a random child object of the mother object.
        """
        if self.child_obj:
            if isinstance(self.child_obj, dict):
                child_obj = random.choice(list(self.child_obj.values()))
            else:
                do_critical(ValueError, f"Child object {self.child_obj} is not a dict.")
        else:
            do_critical(ValueError, f"No child objects found in {self.__class__}.")

        global_logger.info(
            f"Got random child object {child_obj.id} from {self.__class__}."
        )
        return child_obj

    def may_passon_data_filter(self, data_filter: Union[Dict, DataFilter]) -> Dict:
        """
        Loads the data filter from a dictionary or DataFilter object.
        Parameters:
        - data_filter (dict or DataFilter): The data filter to load.

        Returns:
        - data_filter (dict): Defined datafilters in a dictionary format
        """
        if not isinstance(data_filter, DataFilter):
            data_filter = data_filter
            if isinstance(data_filter, dict) and len(data_filter) == 0:
                data_filter = self.data_filter.filters
        return data_filter

    def get_model_information(
        self,
        behavior_data_type: str = "",
        wanted_information: List[
            Literal["binarized" "embedding" "loss" "label" "fluorescence", "all"]
        ] = ["embedding", "loss"],
        manifolds_pipeline: str = "cebra",
        wanted_tasks: List[str] = None,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        train_or_test: str = "train",
    ) -> Dict[str, Dict[str, Union[str, np.ndarray, List[np.ndarray]]]]:
        """
        Get the unique model information all animals, sessions and tasks.

        Only possible if a unique model is found for each task and behavior_data_type is in the model name.

        Parameters
        ----------
        behavior_data_type : str
            The name of the labels describing the behavior used for labeling the embeddings.
        wanted_information : list
            A list containing the wanted information to extract from the models (default is ["embedding", "loss"]).
            Options are
                - "model": the unique models
                - "embedding": the embeddings of the unique models
                - "loss": the training losses of the unique models
                - "raw": the raw data used for training the unique models, which is binarized neural data for models based on cebra
                - "labels": the labels of the data points used for training the unique models
                - "fluorescence": the fluorescence data which is the base for the raw binarized data
                - "all": all of the above
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included, which will result in an error if more than one model is found.
        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.

        Returns
        -------
        embeddings : dict
            A dictionary containing the embeddings of the unique models with the task identifier as key and the embeddings as value.
        losses : dict
            A dictionary containing the losses of the unique models with the task identifier as key and the losses as value.
        labels : dict
            A dictionary containing the labels variable name and a list of labels.

        """

        if self.__class__.__name__ == "Task":
            self.models: Models
            if wanted_tasks is not None and self.name not in wanted_tasks:
                return {}
            informations = self.models.get_model_information(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                wanted_information=wanted_information,
                train_or_test=train_or_test,
                behavior_data_type=behavior_data_type,
            )

        else:
            informations = {}
            for child_name, child in self.child_obj.items():
                information = child.get_model_information(
                    manifolds_pipeline=manifolds_pipeline,
                    model_naming_filter_include=model_naming_filter_include,
                    model_naming_filter_exclude=model_naming_filter_exclude,
                    wanted_information=wanted_information,
                    train_or_test=train_or_test,
                    behavior_data_type=behavior_data_type,
                )
                informations[child_name] = information

        return informations

    def get_unique_models_information(
        self,
        wanted_information: Literal[
            "binarized" "embedding" "loss" "label" "fluorescence", "model", "all"
        ] = "model",
        behavior_data_type: str = "position",
        manifolds_pipeline: str = "cebra",
        wanted_tasks: List[str] = None,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        train_or_test: str = "train",
    ) -> Dict[str, Dict[str, Union[str, np.ndarray, List[np.ndarray]]]]:
        """ """
        if not isinstance(wanted_information, str):
            do_critical(
                ValueError,
                f"wanted_information must be a Literal from this list ['binarized', 'embedding', 'loss', 'label', 'fluorescence', 'model', 'all']. Got {type(wanted_information)} instead.",
            )
        info = self.get_model_information(
            wanted_information=wanted_information,
            manifolds_pipeline=manifolds_pipeline,
            wanted_tasks=wanted_tasks,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            train_or_test=train_or_test,
            behavior_data_type=behavior_data_type,
        )

        # create unique models dict
        unique_model_name_infos = {}
        for key_list, wi in traverse_dicts(info):
            model_name_level = -2
            model_name = key_list[model_name_level]
            if model_name not in unique_model_name_infos:
                unique_model_name_infos[model_name] = {}

            current_dict_level = unique_model_name_infos[model_name]
            keys_until_level = key_list[:model_name_level]
            if len(keys_until_level) == 0:
                current_dict_level = wi
                unique_model_name_infos[model_name] = current_dict_level
            else:
                for key in keys_until_level:
                    if key not in current_dict_level:
                        if key == key_list[model_name_level - 1]:
                            current_dict_level[key] = wi
                        else:
                            current_dict_level[key] = {}
                    current_dict_level = current_dict_level[key]

        return unique_model_name_infos

    def child_obj_traversal(
        self,
        plot: Union[
            bool,
            Literal["Mother", "Animal", "Session", "Task", "Model"],
            List[Literal["Mother", "Animal", "Session", "Task", "Model"]],
        ] = True,
        additional_title: str = "",
        **kwargs: Dict[str, Any],
    ):
        """
        Traverses the child objects of the current object and applies the given function to each object.

        Parameters
        ----------
        kwargs : dict
            The parameters to pass to the function.
            Needed parameters are:
            - "plot"
            - "additional_title"
        """
        if plot is not False:
            if plot is not True:
                plot = make_list_ifnot(plot)
                # get function name of the calling function
                for plot_level in plot:
                    if plot_level not in MetaClass.plot_levels:
                        do_critical(
                            ValueError,
                            f"plot must be True or one of the following: {MetaClass.plot_levels}. Got {plot} instead.",
                        )
        params_dict = create_params_dict(
            exclude=["self", "kwargs", "plot_level"], **locals()
        )
        params_dict.update(kwargs)
        params_dict["additional_title"] += "-" + self.name
        params_dict["plot"] = plot

        obj_analysis = {}
        function_name = inspect.currentframe().f_back.f_code.co_name
        if not hasattr(self, "child_obj"):
            error_msg = f"{self.__class__} has no child objects. Cannot calculate {function_name}."
            global_logger.critical(error_msg)
            raise ValueError(error_msg)

        from Helper import safe_isinstance

        # run analysis in lower levels
        if isinstance(self.child_obj, dict):
            for key, obj in self.child_obj.items():
                function_to_call = getattr(obj, function_name)
                global_logger.info(f"Calculating {function_name} for {obj.id}")
                analysis_output = function_to_call(**params_dict)
                if analysis_output is not None and len(analysis_output) > 0:
                    obj_analysis[key] = analysis_output
        elif safe_isinstance(self.child_obj, Models):
            function_to_call = getattr(self.child_obj, function_name)
            obj = self.child_obj
            global_logger.info(f"Calculating {function_name} for models")
            obj_analysis = function_to_call(**params_dict)

        return obj_analysis

    def get_tasks_dict(self):
        """
        Yields the child objects of the current object.

        Yields
        ------
        object : object
            The child object.
        """
        tasks_dict = {}
        if hasattr(self, "child_obj"):
            if isinstance(self.child_obj, dict):
                for key, child_object in self.child_obj.items():
                    tasks_dict[key] = child_object.get_tasks_dict()
            elif safe_isinstance(self.child_obj, Models):
                # class Task has as child_object a Models object
                return {self.id: self}
            else:
                do_critical(
                    ValueError,
                    f"Child object {self.child_obj} is not a dict or Models object in {self.__class__}.",
                )
        else:
            do_critical(ValueError, f"{self.__class__} has no child objects.")

        return tasks_dict

    def extract_plot_values_dict_of_dicts(
        self, dict_of_dicts: Dict[str, Dict], value_key: str
    ):
        """
        Extracts the values from a dictionary of dictionaries based on the given value key.

        The function traverses the dictionary and extracts the values for each task and model.
        If multiple values are found, they are stacked vertically.
        """
        tasks_dict = self.get_tasks_dict()

        extracted_values = {}
        for keys_list, task in traverse_dicts(tasks_dict):
            # traverse the child objects to get the condition
            condition = task.behavior_metadata["stimulus_type"]
            animal_id, date, task_name = task.id.split("_")

            # extract the values from the model statistics
            model_dict_values = copy.deepcopy(dict_of_dicts)
            for key in keys_list:
                if key in model_dict_values:
                    model_dict_values = model_dict_values[key]

            for model_name, analysis_values in model_dict_values.items():
                values = check_extract_from_parameter_sweep(analysis_values, value_key)

                line_label = (
                    f"{task_name}"
                    if len(model_dict_values) == 1
                    else f"{task_name}_{model_name}"
                )
                if condition not in extracted_values:
                    extracted_values[condition] = {}

                # vertical stack if values already exist
                if line_label in extracted_values[condition]:
                    current_values = extracted_values[condition][line_label]
                    extracted_values[condition][line_label] = np.vstack(
                        [current_values, values]
                    )
                else:
                    extracted_values[condition][line_label] = values

        return extracted_values

    def cross_decode(
        self,
        behavior_data_type: str = "position",
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        add_random: Optional[bool] = False,
        n_neighbors: Optional[int] = None,
        additional_title: Optional[str] = "",
        plot: Union[bool, str] = True,
        regenerate: Optional[bool] = False,
    ):
        """
        Calculates the decoding performance between models from all tasks based on the wanted model names.

        Parameters
        ----------
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included.
        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.
        add_random : bool, optional
            If True, a random model will be added to the decoding (default is False).
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is None).
            if None, the number of neighbors will be determined by k-fold cross-validation in the decoding function.
        additional_title : str, optional
            A string to add to the title of the plot (default is None).
        plot : Union[bool, str], optional
            Whether to plot the decoding results (default is True).
            - If True, the results will be plotted.
            - If "Model", multiple plots will be created for each reference model.

        Returns
        -------
        task_decoding_statistics : dict
            A dictionary containing the decoding statistics between models based on the wanted model names.
        """

        unique_model_name_infos = self.get_unique_models_information(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            wanted_information="model",
            behavior_data_type=behavior_data_type,
        )

        task_decoding_statistics_dict = {}
        for unique_model_name, unique_models_dict in unique_model_name_infos.items():
            unique_models = {}
            for key_list, model in traverse_dicts(unique_models_dict):
                unique_models["_".join(key_list)] = model

            if add_random:
                first_unique_task_model_name = list(unique_models.keys())[0]
                first_unique_model = unique_models[first_unique_task_model_name]
                unique_models["random"] = first_unique_model.make_random()

            additional_title = f" - {self.id} {additional_title}"
            figures_dir = self.dir.joinpath("figures")
            figures_dir.mkdir(parents=True, exist_ok=True)

            global_logger.info(
                f"""Start calculating decoding statistics for all sessions, tasks and models found from {manifolds_pipeline} pipeline using naming filter including {model_naming_filter_include} and excluding {model_naming_filter_exclude}"""
            )
            task_decoding_statistics = Models.cross_decode(
                ref_models=unique_models,
                # models=models,
                n_neighbors=n_neighbors,
                additional_title=additional_title,
                plot=plot,
                regenerate=regenerate,
                save_dir=figures_dir,
            )
            task_decoding_statistics_dict[unique_model_name] = task_decoding_statistics
        return task_decoding_statistics_dict

    def structure_index(
        self,
        params: Dict[str, Union[int, bool, List[int]]],
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: Union[
            bool,
            Literal["Mother", "Animal", "Session", "Task", "Model"],
            List[Literal["Mother", "Animal", "Session", "Task", "Model"]],
        ] = True,
        additional_title: str = "",
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """
        The structural index is a measure of the structure of the neural data.

        Parameters
        ----------
        params : Dict[str, Union[int, bool, List[int]]]
            Parameters for the structural index calculation. If n_neighbors is a list, a parameter sweep is performed.
            Keywords:
                "n_bins": 10, - number of bins to split the data into
                "n_neighbors": 15, - number of neighbors for the KNN algorithm
                "discrete_label": False, - whether the labels are discrete or continuous
                "num_shuffles": 0, - number of shuffles to perform
                "verbose": True, -  whether to print verbose output
        plot : bool, str, optional
            Whether to plot the structural indices (default is True).
            Options:
                - True: plot the structural indices plots all models into one plot
                - "Animal": plot the structural indices for each animal
                - "Session": plot the structural indices for each session
                - "Task": plot the structural indices for each task
                - "Model": plot the structural indices for each model in a single plot

        Returns
        -----------
        structural_indices : dict
            A dictionary containing the structural indices for each model.
            The structure index is a measure of the structure of the neural data.
            The structure index is a dictionary with the following keys:
                - "mean": mean structure index
                - "variance": variance structure index
                - "distribution": distribution of the structure index
                - "n_neighbors": number of neighbors used for the KNN algorithm
                - "n_bins": number of bins
        """
        params_dict = create_params_dict(exclude=["self"], **locals())
        obj_analysis = self.child_obj_traversal(**params_dict)

        if isinstance(plot, List):
            plot = True if type(self).__name__ in plot else False
        if plot:
            global_logger.error(f"Plotting SI not implemented yet.")
            # param_range = copy.deepcopy(params["n_neighbors"])
            # parameter_sweep = is_array_like(param_range)
            # plot_save_dir = plot_save_dir or self.figures_dir
            # if parameter_sweep:
            #     task_line_plot_dict = self.extract_plot_values_dict_of_dicts(
            #         obj_analysis, "SI"
            #     )

            #     if space == "embedding":
            #         space = f"{manifolds_pipeline.upper()}"
            #     elif space == "fluorescence":
            #         space = "Activity"
            #     elif space == "binarized":
            #         space = "Binarized"
            #     else:
            #         raise ValueError(
            #             f"Unknown space {space}. Must be one of ['fluorescence', 'binarized', 'embedding']"
            #         )
            #     space += " Space"

            #     title = f"Structural Indices {space} for different tasks - {self.id}"
            #     Vizualizer.lineplot_from_dict_of_dicts(
            #         task_line_plot_dict,
            #         title=title,
            #         ylabel="Structural Index",
            #         xlabel="Number Neighbors",
            #         additional_title=additional_title,
            #         xticks=list(param_range),
            #         xtick_pos=np.arange(len(param_range)),
            #         save_dir=plot_save_dir or self.figures_dir,
            #     )
        return obj_analysis

    def feature_similarity(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        similarity: Literal["pairwise", "inside", "outside"] = "inside",
        out_det_method: Literal["density", "contamination"] = "density",
        remove_outliers: bool = True,
        parallel: bool = True,
        space: Literal["fluorescence", "binarized", "embedding"] = "embedding",
        metric: str = "cosine",
        plot: Union[
            bool, Literal["Mother", "Animal", "Session", "Task", "Model"]
        ] = True,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        regenerate: bool = False,
        additional_title: str = "",
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ):
        """
        Calculates the feature similarity between models from all tasks based on the wanted model names.

        Parameters
        ----------
        plot : bool, str, optional
            Whether to plot the feature similarity.
            Options:
                - True: plots the output based on the class
                - "Animal": plots the output based on the class
                - "Session": plots the output based on the class
                - "Task": plots the output based on the class
                - "Model": plots all outputs from lowest class (Model)


        """
        params_dict = create_params_dict(exclude=["self"], **locals())
        obj_analysis = self.child_obj_traversal(**params_dict)

        if plot is True or type(self).__name__ in plot:
            plot_save_dir = plot_save_dir or self.figures_dir

            raise NotImplementedError(f"Not sure how to plot this yet")
            similarities = self.average_dict_of_dicts(obj_fsim)

            space = (
                f"{manifolds_pipeline.upper()} Space"
                if not use_raw
                else "Activity Space"
            )
            title = f"Feature similarities {space} Averaged Tasks - {self.id}"
            xlabel = ("Position Bin X",)
            ylabel = ("Position Bin Y",)
            if similarity in ["inside", "pairwise"]:
                Vizualizer.plot_heatmap(
                    to_show_similarities,
                    additional_title=additional_title,
                    figsize=figsize,
                    title=title,
                    xticks=xticks,
                    yticks=yticks,
                    xticks_pos=xticks_pos,
                    yticks_pos=yticks_pos,
                    colorbar_label=metric,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    save_dir=plot_save_dir,
                    as_pdf=as_pdf,
                )
            elif similarity == "between":
                Vizualizer.plot_group_distr_similarities(
                    {metric: similarities},
                    additional_title=additional_title,
                    bins=plot_bins,
                    colorbar=True,
                    xticks=xticks,
                    yticks=yticks,
                    tick_steps=tick_steps,
                    colorbar_label=metric,
                    save_dir=plot_save_dir,
                    as_pdf=as_pdf,
                )
        return obj_analysis

    def shape_similarity(
        self,
        manifolds_pipeline: str = "cebra",
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        behavior_data_type: str = "position",
        train_or_test: Literal["train", "test"] = "train",
        method: Literal["procrustes", "one-to-one", "soft-matching"] = "soft-matching",
        plot_show: Literal[
            "center", "center_std", "samples", "flow", "annotate_dots"
        ] = "samples",
        task_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
        wanted_tasks: Optional[List[str]] = None,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        additional_title: str = "",
        n_components: int = 2,
        regenerate: bool = False,
        save_path: Optional[Path] = None,
        plot: Union[
            bool,
            List[
                Literal[
                    "heatmap",
                    "tsne",
                    "mds",
                    "isomap",
                    "lle",
                    "spectral_embedding",
                    "umap",
                    "pca",
                    "mds_pca",
                ]
            ],
        ] = ["umap", "heatmap"],
        group_by: Literal[
            "condition", "animal", "date", "task", "model", "stimulus"
        ] = "condition",
        compare_by: Literal[
            "animal", "date", "task", "model", "stimulus", "condition"
        ] = "task",
        plot_save_dir: Optional[Path] = None,
        cell_type: Literal["place", "non-place", "all"] = "place",
    ):
        """Compute and visualize similarity between neural or behavioral data manifolds.

        This method analyzes task-related data across conditions, animals, and tasks, grouping
        tasks by prefixes and conditions, assigning color palettes for visualization, and computing
        shape similarity using a specified method (e.g., Procrustes analysis). It supports multiple
        visualization options and saves results to specified paths.

        Args:
            manifolds_pipeline (str, optional): Pipeline for generating embeddings. Defaults to "cebra".
            space (Literal["fluorescence", "binarized", "embedding"], optional): Data space to analyze.
                Defaults to "binarized".
            behavior_data_type (str, optional): Type of behavioral data. Defaults to "position".
            train_or_test (Literal["train", "test"], optional): Use training or testing data.
                Defaults to "train".
            method (Literal["procrustes"], optional): Similarity computation method. Defaults to "procrustes".
            task_groups (dict, optional): Dictionary with keys as group names and values as lists of task names to plot. If None, plots all values.
                        example: {
                                    'condition1':
                                        task_group1 : ['task1', 'task2'],
                                        task_group2 : ['task3']

                                    'condition2':
                                        task_group1 : ['task1', 'task2'],
                                }.
            wanted_tasks (Optional[List[str]], optional): Specific tasks to include. If None, all tasks
                are used. Defaults to None.
            model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
                Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
                1. Include all models containing a specific string: "string"
                2. Include all models containing a specific combination of strings: ["string1", "string2"]
                3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
            model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
                Same as model_naming_filter_include but for excluding models.
            additional_title (str, optional): String to append to output filename. Defaults to None.
            regenerate (bool, optional): If True, regenerates results even if cached. Defaults to False.
            save_path (Optional[Path], optional): Path to save similarity results. Defaults to None
                (uses self.output_dir).
            plot (Union[bool, List[Literal[...]]], optional): Visualization types (e.g., "umap", "heatmap").
                If False, no plots are generated. Defaults to ["umap", "heatmap"].
            color_by (List[Literal["animal", "date", "task", "model", "stimulus"]], optional): Variables
                for plot labels and colors. Defaults to "animal".
            plot_save_dir (Optional[Path], optional): Directory to save plots. Defaults to None
                (uses self.figures_dir).
            only_placecells (bool, optional): If True, filters to place cells only. Defaults to True.

        Returns:
            np.ndarray: Matrix of shape similarity scores between tasks.

        Raises:
            ValueError: If a task name lacks a numeric component, logged via global_logger.

        Notes:
            - Assumes `self.animals` provides condition information and `model.tuning_map` returns
            compatible activity data.
            - Uses regex to extract task prefixes and numbers from task names.
            - Logs warnings for empty task groups or invalid task names via global_logger.
            - Supports visualization with libraries like Matplotlib and Seaborn.
        """
        # Define output filename and paths
        ofname = f"{method}_shape_similarity"
        if space.upper() == "EMBEDDING":
            space_type = f"{manifolds_pipeline}"
        else:
            space_type = space

        additional_title = (
            f"{space_type.upper()} activity space|tuned to {behavior_data_type}"
        )
        additional_title += f"|{cell_type.upper()} cells"
        additional_title += f"|{self.data_filter.filter_description(short=True)}"

        if additional_title:
            ofname += f"_{clean_filename(additional_title)}"

        save_path = save_path or self.output_dir.joinpath(ofname)

        # Get unique model information
        unique_model_name_infos = self.get_unique_models_information(
            manifolds_pipeline=manifolds_pipeline,
            wanted_tasks=wanted_tasks,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            wanted_information="model",
            behavior_data_type=behavior_data_type,
        )

        # Initialize DataFrame for tracking task information
        task_data_df = pd.DataFrame(
            columns=[
                "animal",
                "date",
                "task",
                "task_name",
                "task_number",
                "model",
                "condition",
                "id",
                "activity",
            ]
        )

        labels = []
        for key_list, model in traverse_dicts(unique_model_name_infos):
            task = model.id("task")
            delete_task = False
            if wanted_tasks and task not in wanted_tasks:
                delete_task = True
            else:
                task_name, task_number = search_split(r"\d+", task)
                if not task_number:
                    global_logger.error(
                        f"Task name {task} does not contain a number. Skipping."
                    )
                    delete_task = True
            if delete_task:
                del unique_model_name_infos[key_list[:-1]]
            else:
                labels.append(model.id())

        from Manimeasure import load_df_sim

        if save_path.with_suffix(".h5").exists() and not regenerate:
            missing_pairs, df_sim = load_df_sim(labels, save_path)

        # Collect task information
        for key_list, model in traverse_dicts(unique_model_name_infos):
            # condition = self.animals[key_list[1]].condition
            # filter activity for placecells
            # TODO: move cell filtering to photon class and filter when loading data
            wanted_indices = None
            if cell_type in ["place", "non-place"]:
                task_obj = (
                    self.animals[key_list[1]].sessions[key_list[2]].tasks[key_list[3]]
                )
                task_datafiler_str = task_obj.data_filter.filter_description(short=True)
                if self.data_filter.filter_description() != task_datafiler_str:
                    global_logger.warning(
                        f"Data filter of Mother object {self.data_filter.filter_description(short=True)} does not match data filter of Task object {task_datafiler_str}."
                    )
                pc_df = task_obj.get_cells(cell_type=cell_type)
                wanted_indices = pc_df.index.tolist()

            if (
                len(missing_pairs) > 0
                and model.id() in np.array(missing_pairs).flatten()
            ):
                return_activity_type = "tuning"
            else:
                return_activity_type = "none"

            model_df = model.to_df(
                train_or_test=train_or_test,
                space=space,
                behavior_data_type=behavior_data_type,
                return_activity_type=return_activity_type,
                cell_indices=wanted_indices,
            )

            task_data_df = pd.concat(
                [task_data_df, model_df],
                ignore_index=True,
            )

        task_data_df = Mother.improve_animals_df(
            df=task_data_df,
            group_by=group_by,
            compare_by=compare_by,
            task_groups=task_groups,
        )

        # Create plot_df as a copy of task_data_df with selected columns
        plot_df = task_data_df[
            [
                "animal",
                "plot_label",
                "group_by",
                "compare_by",
                group_by,
                compare_by,
                "group_key",
                "color",
                "task_number",
                "previous_task_idx",
                "next_task_idx",
                "group_task_number",
            ]
        ].copy()
        plot_df = plot_df.rename(columns={"task_number": "number"})

        additional_title += " " if additional_title else ""

        # Compute shape similarity
        shape_sim, labels = calc_shape_similarity(
            data=dict(zip(task_data_df["id"], task_data_df["activity"])),
            labels=task_data_df["id"].tolist(),
            plot=plot,
            plot_df=plot_df,
            method=method,
            plot_show=plot_show,
            additional_title=additional_title,
            save_path=save_path,
            plot_save_dir=plot_save_dir or self.figures_dir,
            regenerate=regenerate,
            n_components=n_components,
        )

        return shape_sim, task_data_df

    def plot_velocity_summary(
        self,
        manifolds_pipeline: str = "cebra",
        train_or_test: Literal["train", "test"] = None,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        merge_by: Literal["animal", "date", "task", "stimulus"] = None,
        filter_by: str = None,
        additional_title: str = "",
        xlim: Tuple[float, float] = (0.2, None),
        ylim: Tuple[float, float] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Plots the velocity of the animals based on the given parameters.

        Parameters
        ----------
        manifolds_pipeline : str, optional
            The name of the manifolds pipeline to use for decoding (default is "cebra").
        train_or_test : Literal["train", "test"], optional
            Whether to use the training or test data (default is None).
        model_naming_filter_include : Union[str, List[str], List[List[str]]], optional
            A list of strings to include in the model naming filter (default is None).
        model_naming_filter_exclude : Union[str, List[str], List[List[str]]], optional
            A list of strings to exclude in the model naming filter (default is None).
        additional_title : str, optional
            An additional title to add to the plot (default is "").

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the velocity summary of all tasks.
            The DataFrame has the shape (max_speed_count, n_tasks) filled with NaN if speed is missing
            Each column represents a task and each row represents a time frame.
        """

        if (
            train_or_test is not None
            or model_naming_filter_exclude is not None
            or model_naming_filter_include is not None
        ):
            do_critical(
                NotImplementedError,
                "Filtering of velocity data for models is not implemented yet.",
            )
        else:
            tasks_dict = self.get_tasks_dict()
            if not isinstance(tasks_dict, dict):
                tasks_dict = {tasks_dict.name: tasks_dict}

            max_frames = 0
            condition_task_dict = {}
            for key_list, task in traverse_dicts(tasks_dict):
                condition = task.session.animal.condition
                if condition not in condition_task_dict:
                    condition_task_dict[condition] = {}
                v = task.behavior.velocity.euclidean
                simulus_type = task.behavior_metadata.get("stimulus_type", "?stimulus")
                key = "_".join(key_list) + f"_{simulus_type}"
                condition_task_dict[condition][key] = v
                max_frames = max(max_frames, v.shape[0])

            for condition, tasks_v in condition_task_dict.items():
                # create a pandas dataframe with shape (max_speed_count, n_tasks) filled with NaN
                velocity_summary = np.full((max_frames, len(tasks_v)), np.nan)
                for i, (task_id, velocity) in enumerate(tasks_v.items()):
                    if velocity is not None:
                        velocity = (
                            np.linalg.norm(velocity, axis=1)
                            if velocity.ndim > 1
                            else velocity
                        )
                        velocity_summary[: len(velocity), i] = velocity
                velocity_df = pd.DataFrame(velocity_summary, columns=tasks_v.keys())

                merged_df = filter_merge_df(velocity_df, filter_by, merge_by)
                condition_task_dict[condition] = merged_df

            plot_2d_kde_dict_of_dicts(
                condition_task_dict,
                title=f"Velocity Histogram | {self.id} {additional_title} |filter: {filter_by} |merged: {merge_by}",
                xlabel="Velocity Bin",
                first_direction="y",
                xlim=xlim,
                ylim=ylim,
            )

        return velocity_df


class Mother(MetaClass):
    """
    Mother class for all animal objects
    """

    analysis_types = {
        "structure_index": {"key": "SI"},
        "feature_similarity": {"key": None},
        "cross_decode": {"key": "rmse"},
        "shape_similarity": {"key": None},
    }

    def __init__(self, dir: str, data_filter: Union[Dict[str, str], DataFilter] = None):
        """

        Parameters
        ----------
        - dir (string): The root directory path where the animal folders are stored.
        """
        super().__init__(dir, data_filter=data_filter)
        self.animals: Dict[str, Animal] = {}
        self.metadatas: Dict[str, Dict[str, Any]] = {}
        self.animal_ids: List[str] = None
        self.dates: List[str] = None
        self.task_names: List[str] = None
        self.model_settings: Dict[str, Dict[str, Union[str, float]]] = None
        self.behavior_data_types: List[str] = None
        self.extracted_analysis_data = {}

    @property
    def id(self):
        """
        Returns the id of the mother object composed of initialization parameters.
        """
        id = f"{self.__class__.__name__}"
        return id

    @property
    def child_obj(self):
        """
        Returns the child objects of the mother object.
        """
        return self.animals

    def owns(self, value, type: Literal["Animal", "Date", "Task"]) -> bool:
        owns = False
        if type == "Animal":
            if self.Animals is not None:
                if value in self.animal_ids:
                    owns = True
            else:
                # check structure of the value
                include_regex = naming_structure["animal"]
                if isinstance(value, str) and re.match(include_regex, value):
                    owns = True
        elif type == "Date":
            if self.dates is not None:
                if value in self.dates:
                    owns = True
            else:
                # check structure of the value
                include_regex = naming_structure["date"]
                if isinstance(value, str) and re.match(include_regex, value):
                    owns = True
        elif type == "Task":
            if self.task_names is not None:
                if value in self.task_names:
                    owns = True
            else:
                # check structure of the value
                exclude_regex = forbidden_names
                owns = regex_search(value, exclude_regex=exclude_regex)
        return owns

    @staticmethod
    def improve_animals_df(
        df: pd.DataFrame,
        task_groups: Optional[Dict[str, Dict[str, List[str]]]] = None,
        group_by: Literal[
            "condition", "animal", "date", "task", "model", "stimulus"
        ] = "condition",
        compare_by: Literal[
            "animal", "date", "task", "model", "stimulus", "condition"
        ] = "task",
        cmaps: Optional[List[str]] = None,
        sort_by: List[str] = ["group_key", "condition", "task_name", "task_number"],
    ) -> pd.DataFrame:
        """
        Enhances the input DataFrame by adding grouping, coloring, labeling, and sequencing columns
        for visualization purposes. Utilizes custom or auto-grouping for tasks.

        Parameters:
        - df (pd.DataFrame): Input DataFrame with required columns: 'animal', 'task', 'task_name',
          'task_number', 'date', and the columns specified by `group_by` and `compare_by`.
          A copy is made internally to avoid modifying the original.
        - task_groups (Optional[Dict[str, Dict[str, List[str]]]]): Dictionary for custom task grouping.
            If None, autogrouping is applied based on `group_by` and `compare_by`.
            Outer keys match unique values in `group_by`. Inner dictionaries have subgroup names as keys
            and lists of values (from `compare_by`) as values. Example:
                {
                    'condition1': {
                        'task_group1': ['task1', 'task2'],
                        'task_group2': ['task3']
                    },
                    'condition2': {
                        'task_group1': ['task1', 'task2']
                    }
                }
        - group_by (Literal[...]): Column for primary grouping (default: "condition").
        - compare_by (Literal[...]): Column for subgroup comparison (default: "task").
        - cmaps (Optional[List[str]]): List of matplotlib colormaps for color assignment. If None,
          uses a default list of varied colormaps.
        - sort_by (List[str]): Columns to sort the final DataFrame (default: ["condition", "task_name", "task_number"]).

        Returns:
        - pd.DataFrame: Enhanced DataFrame with added columns:
          - 'group_key': Group identifier (from grouping).
          - 'group_task_number': Position within the subgroup.
          - 'group_by': Values from the `group_by` column.
          - 'compare_by': Values from the `compare_by` column.
          - 'color': RGBA color tuple for each row, based on group and position (alpha=0.8).
          - 'group_key': Alias for 'group_name'.
          - 'plot_label': Per-row label as "{group_by_value} - {compare_by_value}".
          - 'previous_task_idx': Index of previous task in sequence per animal (None if first).
          - 'next_task_idx': Index of next task in sequence per animal (None if last).

        Raises:
        - ValueError: If required columns are missing, `group_by` equals `compare_by`, or grouping fails.
        """
        if cmaps is None:
            cmaps = [
                "Reds_r",
                "Blues_r",
                "Greens_r",
                "Wistia",
                "spring",
                "Purples_r",
                "plasma",
                "cool",
                "Oranges_r",
                "YlGn_r",
                "RdPu_r",
                "GnBu_r",
            ]

        # Validate parameters
        if group_by == compare_by:
            raise ValueError("`group_by` and `compare_by` must be different columns.")

        required_columns = {
            "animal",
            "task",
            "task_name",
            "task_number",
            "date",
            group_by,
            compare_by,
        }
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Work on a copy to avoid modifying the original DataFrame
        df = df.copy()

        # Apply grouping (adds 'group_name' and 'group_task_number')
        grouped_df = group_df_by_custom_groups(df, task_groups, group_by, compare_by)

        # Initialize new columns
        df["color"] = None
        df["group_key"] = None
        df["plot_label"] = None
        df["group_name"] = None
        df["group_task_number"] = None
        df["group_by"] = df[group_by]
        df["compare_by"] = df[compare_by]
        df["previous_task_idx"] = None
        df["next_task_idx"] = None

        # Assign colors and labels per group
        for i, (group_key, group_df) in enumerate(grouped_df):
            # Get position-based numbers for coloring
            numbers = group_df["group_task_number"]
            num_unique = len(numbers.unique())
            number_ranking = numbers.rank(method="dense").astype(int) - 1
            num_colors = num_unique * 2
            cmap = cmaps[i % len(cmaps)]
            offset = 0  # num_colors // 4
            colors = Vizualizer._get_base_color(
                number_ranking.to_numpy() + offset, num_colors, cmap
            )
            colors[:, -1] = 0.8  # Set alpha to 0.8

            # Assign to group_df indices
            color_tuples = [tuple(color) for color in colors]
            df.loc[group_df.index, "color"] = pd.Series(
                color_tuples, index=group_df.index
            )
            df.loc[group_df.index, "group_key"] = group_key
            df.loc[group_df.index, "plot_label"] = (
                df.loc[group_df.index, group_by].astype(str)
                + " - "
                + df.loc[group_df.index, compare_by].astype(str)
            )
            df.loc[group_df.index, "group_task_number"] = group_df["group_task_number"]

            # add previous and next task index based on group by extracting indexs of group_task_number with -1 and +1
            for animal_id, animal_group_df in group_df.groupby("animal"):
                previouse_task_indices = animal_group_df["group_task_number"].apply(
                    lambda x: animal_group_df.index[
                        animal_group_df["group_task_number"] == x - 1
                    ].tolist()
                )
                previouse_task_indice_list = []
                for previouse_task_idx in previouse_task_indices:
                    if len(previouse_task_idx) == 0:
                        previouse_task_indice_list.append(None)
                    elif len(previouse_task_idx) == 1:
                        previouse_task_indice_list.append(previouse_task_idx[0])
                    else:
                        raise ValueError(
                            f"Multiple previous tasks found for animal {animal_id} in group {group_key}. Expected only one."
                        )
                next_task_indices = animal_group_df["group_task_number"].apply(
                    lambda x: animal_group_df.index[
                        animal_group_df["group_task_number"] == x + 1
                    ].tolist()
                )
                next_task_indices_list = []
                for next_task_idx in next_task_indices:
                    if len(next_task_idx) == 0:
                        next_task_indices_list.append(None)
                    elif len(next_task_idx) == 1:
                        next_task_indices_list.append(next_task_idx[0])
                    else:
                        raise ValueError(
                            f"Multiple next tasks found for animal {animal_id} in group {group_key}. Expected only one."
                        )

                df.loc[animal_group_df.index, "previous_task_idx"] = (
                    previouse_task_indice_list
                )
                df.loc[animal_group_df.index, "next_task_idx"] = next_task_indices_list

        # Check if all rows have colors assigned
        if df["color"].isna().any():
            raise ValueError(
                "Some rows were not assigned colors. Ensure all rows are grouped correctly."
            )

        # Final sort
        df.sort_values(by=sort_by, inplace=True)
        return df

    def load_children_info(
        self,
        animal_ids: Union[str, List[str]],
        dates: Union[str, List[str]],
        task_names: Union[str, List[str]],
        model_settings: Dict[str, Dict[str, Union[str, float]]],
        behavior_data_types: Literal[
            "position", "distance", "moving", "velocity", "acceleration", "stimulus"
        ] = "position",
    ):
        self.animal_ids = animal_ids
        self.dates = dates
        self.task_names = task_names
        self.model_settings = model_settings
        self.behavior_data_types = behavior_data_types

    def extract_metadata(
        self, animal: Animal = None, animal_id: str = None
    ) -> Dict[str, Any]:
        """
        Extracts metadata from the animal object and stores it in the metadatas dictionary.

        Parameters
        ----------
        animal : Animal
            The animal object to extract metadata from.
        """
        if animal is None and animal_id is None:
            do_critical(ValueError, "Either animal or animal_id must be provided.")
        if animal is None and animal_id is not None and animal.id != animal_id:
            do_critical(
                ValueError,
                f"Animal ID {animal_id} does not match the provided animal object.",
            )
        if animal is None:
            if animal_id not in self.child_obj:
                do_critical(
                    ValueError,
                    f"Animal ID {animal_id} is not in the wanted animal IDs list (animal does not belong to mother).",
                )

        if animal_id is None:
            animal_id = animal.id
            if animal.id not in self.metadatas:
                # extract metadata from animal object
                animal_metadata = copy.deepcopy(animal.metadata)
                for date, session in animal.child_obj.items():
                    date_metadata = copy.deepcopy(session.metadata)
                    animal_metadata[date] = date_metadata
                self.metadatas[animal.id] = animal_metadata

        return self.metadatas[animal_id]

    def get_children(
        self,
        animal_ids: Union[str, List[str]] = None,
        dates: Union[str, List[str]] = None,
        task_names: Union[str, List[str]] = None,
        model_settings: Dict[str, Dict[str, Union[str, float]]] = None,
        behavior_data_types: Literal[
            "position", "distance", "moving", "velocity", "acceleration", "stimulus"
        ] = ["position", "moving"],
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
        ] = {},
        regenerate: bool = False,
        regenerate_plots: bool = False,
        plot: bool = True,
        forget: bool = False,
    ):
        """
        Loads animal data from the specified root directory for the given animal IDs.

        Parameters:
        - animal_ids (str or list): The animal IDs to load. If None, all animals will be loaded.
        - dates (str or list): The dates to load. If None, all dates will be loaded.
        - task_names (str or list): The task_names to load. If None, all task_names will be loaded.
        - model_settings (dict): The model settings to use for loading the data.
            Example:
                model_settings = {
                    "place_cell": {"method": "skaggs"},
                    "cebra": {"max_iterations": 5000, "output_dimension": 3},
                }
            For CEBRA: quick CPU run-time demo, you can drop `max_iterations` to 100-500; otherwise set to 5000
        - behavior_data_types (str): The type of behavior data to load. Default is "position".
            Options:
            - "position": Position data.
            - "velocity": Velocity data.
            - "acceleration": Acceleration data.
            - "stimulus": Stimulus data.
            - "distance": Distance data.
        - regenerate (bool): Whether to regenerate the data behavior data.
        - regenerate_plots (bool): Whether to regenerate the plots. Default is False.
        - plot (bool): Whether to plot the data. Default is True.
        - forget (bool): Whether to forget the loaded data. Default is False.

        Returns:
        - animals (dict): A dictionary containing animal IDs as keys and corresponding Animal objects as values.
        """

        if self.dir is None or not self.dir.exists():
            do_critical(FileExistsError, f"Root directory {self.dir} does not exist.")

        animal_ids = make_list_ifnot(animal_ids)
        dates = make_list_ifnot(dates)
        task_names = make_list_ifnot(task_names)

        if not forget:
            self.load_children_info(
                animal_ids=animal_ids,
                dates=dates,
                task_names=task_names,
                model_settings=model_settings,
            )
        else:
            err_msg = None
            if self.animal_ids is not None:
                for animal_id in animal_ids:
                    if animal_id not in self.animal_ids:
                        err_msg = f"Animal ID {animal_ids} is not in the Mother object defined wanted list (does not belong to mother)"
            if self.dates is not None:
                for date in dates:
                    if date not in self.dates:
                        err_msg = f"Date {dates} is not in the Mother object defined wanted list (does not belong to mother)"
            if self.task_names is not None:
                for task in task_names:
                    if task not in self.task_names:
                        err_msg = f"Task {task_names} is not in the Mother object defined wanted list (does not belong to mother)"
            if err_msg:
                do_critical(ValueError, err_msg)

        present_animal_folders = search_filedir(
            path=self.dir,
            include_regex=naming_structure["animal"],
            type="dir",
        )

        data_filter = self.may_passon_data_filter(data_filter)
        animals = {}

        def process_animal_folder(animal_folder):
            animal_id = animal_folder.name
            if animal_ids is None or animal_id in animal_ids:
                if animal_id in self.animals:
                    animal = self.animals[animal_id]
                else:
                    animal = Animal(
                        animal_id=animal_id,
                        root_dir=self.dir,
                        model_settings=model_settings,
                        data_filter=data_filter,
                    )
                if animal.usable is False:
                    global_logger.info(
                        f"Animal {animal_id} marked as unusable. Skipping."
                    )
                    return None
                date_folders = search_filedir(
                    path=animal.dir,
                    include_regex=naming_structure["date"],
                    type="dir",
                )
                for date_folder in date_folders:
                    date = date_folder.name
                    if dates is None or date in dates:
                        session: Session
                        if date in animal.sessions:
                            session = animal.sessions[date]
                        else:
                            animal.add_session(
                                dates=date,
                                model_settings=model_settings,
                                data_filter=data_filter,
                            )
                            session = animal.sessions[date]

                        task_folders = search_filedir(
                            path=date_folder,
                            exclude_regex=forbidden_names,
                            type="dir",
                        )
                        for task_folder in task_folders:
                            task_name = task_folder.name
                            if task_names is None or task_name in task_names:
                                if task_name not in session.tasks:
                                    session.add_task(
                                        task_names=task_name,
                                        data_filter=data_filter,
                                    )
                                    task = session.tasks[task_name]
                                    if task.usable is False:
                                        global_logger.info(
                                            f"Task {task_name} of session {date} of animal {animal_id} marked as unusable. Skipping."
                                        )
                                        del session.tasks[task_name]
                                        continue
                                    task.load_all_data(
                                        behavior_data_types=behavior_data_types,
                                        regenerate=regenerate,
                                        regenerate_plots=regenerate_plots,
                                        plot=plot,
                                    )
                                    if plot:
                                        plt.show()
                        if len(session.tasks) == 0:
                            global_logger.info(
                                f"No tasks loaded from folder {animal_id}\{date_folder}. Removing Date from animal."
                            )
                            del animal.sessions[date]

                if len(animal.sessions) == 0:
                    global_logger.info(
                        f"No sessions loaded from folder {animal_id}. Removing animal."
                    )
                    if animal_id in self.animals:
                        del self.animals[animal_id]
                else:
                    self.extract_metadata(animal=animal)
                    return animal_id, animal
            return None

        for animal_folder in present_animal_folders:
            result = process_animal_folder(animal_folder)
            if result:
                animal_id, animal = result
                animals[animal_id] = animal

        if not forget:
            self.animals = sort_dict(animals)

        return animals

    def train_children(
        self,
        animal: Union[Animal, List[Animal]] = None,
        load_params: Dict[str, Any] = None,
        train_params: Dict[str, Any] = None,
    ):
        """ """
        load_params = copy.deepcopy(load_params)
        train_params = copy.deepcopy(train_params)
        if load_params is None:
            load_params = self.default_params("load")
        if train_params is None:
            train_params = self.default_params("train")

        load_params = self.preprocess_loading_params(load_params=load_params)

        if animal is not None and load_params["animal_ids"] is not None:
            global_logger.warning(
                f"animal and animal_ids are both provided, using animal."
            )
        if animal is not None:
            if isinstance(animal, Animal):
                animals = make_list_ifnot(animal)
            elif isinstance(animal, list):
                animals = {}
                for a in animal:
                    if not isinstance(a, Animal):
                        do_critical(
                            ValueError,
                            f"animal is not of type Animal: {a.__class__}",
                        )
                    else:
                        animals[a.id] = a
            else:
                do_critical(
                    ValueError,
                    f"animal is not of type Animal: {animal.__class__}",
                )
        else:
            animals = self.get_children(**load_params)

        if len(animals) == 0:
            do_critical(
                ValueError,
                "No animal to train, either provide an animal or ensure that animals are loaded via Mother.get_children.",
            )
        for animal_id, animal in animals.items():
            if not isinstance(animal, Animal):
                do_critical(ValueError, f"animal is not of type Animal: {animal}")

            animal.train_model(**train_params)

        return animals

    def default_params(self, type: str):
        """
        Returns the default parameters for the given type.

        Parameters:
            type (str): The type of parameters to return.

        Returns:
            dict: The default parameters for the given type.
        """
        params = {}
        if type == "train":
            model_settings = {
                "place_cell": {"method": "skaggs"},
                "cebra": {"max_iterations": 12800, "output_dimension": 3},
            }
            params = {
                "model_settings": model_settings,
                "manifolds_pipeline": "cebra",
                "model_type": "behavior",
                "behavior_data_types": ["position"],
                "shuffle": False,
                "movement_state": "moving",
                "split_ratio": 1,
                "name_comment": None,
                "transformation": None,
                "create_embeddings": True,
                "verbose": False,
                "regenerate": False,
                "plot": False,
            }
        elif type == "load":
            params = {
                "animal_ids": None,
                "dates": None,
                "task_names": None,
                "behavior_data_types": ["position", "moving"],
                "regenerate_plots": False,
                "regenerate": False,
                "plot": False,
            }
        elif type == "analysis":
            params = {
                "analysis_type": "feature_similarity",
                "manifolds_pipeline": "cebra",
                "metric": "cosine",
                "similarity": "inside",
                "plot": "all_Model",
                "use_raw": False,
                "regenerate": False,
            }
        else:
            raise ValueError(f"Unknown type: {type}")
        return params

    def preprocess_loading_params(
        self,
        load_params: Dict[str, Any] = None,
    ):
        animal_ids = load_params.pop("animal_ids")
        dates = load_params.pop("dates")
        task_names = load_params.pop("task_names")
        behavior_data_types = load_params.pop("behavior_data_types")
        model_settings = load_params.pop("model_settings")
        behavior_data_types = make_list_ifnot(behavior_data_types)

        animal_ids = make_list_ifnot(animal_ids)
        dates = make_list_ifnot(dates)
        task_names = make_list_ifnot(task_names)
        model_settings = model_settings or self.model_settings
        behavior_data_types = behavior_data_types or self.behavior_data_types
        behavior_data_types = make_list_ifnot(behavior_data_types)

        self.load_children_info(
            animal_ids=animal_ids,
            dates=dates,
            task_names=task_names,
            model_settings=model_settings,
        )

        updated_load_params = {
            "animal_ids": animal_ids,
            "dates": dates,
            "task_names": task_names,
            "behavior_data_types": behavior_data_types,
            "model_settings": model_settings,
            **load_params,
        }
        return updated_load_params

    def analyse_children(
        self,
        load_params: Dict[str, Any] = None,
        train_params: Dict[str, Any] = None,
        analysis_params: Dict[str, Any] = None,
        forget: bool = True,
        skip_error: bool = False,
    ):
        """
        Loads animal data from the specified root directory based on the given parameters extract data and forget animals.

        Parameters:
            - load_params (dict): Parameters for loading animal data.
            - train_params (dict): Parameters for training models.
            - analysis_params (dict): Parameters for analysis. Different analysis types require different parameters.
                - analysis_type (str): The type of analysis to perform. Options are:
                    - "structure_index": Calculate the structure index of the neural activity based on behavior.
                    - "feature_similarity": Calculate the feature similarity between neural activity and behavior.
                    - "cross_decode": Perform cross-decoding analysis.
                    - "shape_similarity": Calculate the shape similarity of neural manifolds.
                - task_groups (dict, optional): Custom task grouping for plotting.
                    - example:
                        task_groups = {

            - forget (bool): Whether to forget the loaded animal. Default is True.


        Returns:
            Extracted animal data.
        """
        load_params = copy.deepcopy(load_params)
        train_params = copy.deepcopy(train_params)
        analysis_params = copy.deepcopy(analysis_params)
        analysis_type = analysis_params.pop("analysis_type")
        task_groups = analysis_params.pop("task_groups", None)

        if self.extracted_analysis_data.get(analysis_type) is not None:
            for num, analysis_values in self.extracted_analysis_data[
                analysis_type
            ].items():
                if (
                    analysis_values["load_params"] == load_params
                    and analysis_values["train_params"] == train_params
                    and analysis_values["analysis_params"] == analysis_params
                ):
                    print(
                        f"Already extracted analysis data for {analysis_type} with the same parameters."
                    )
                    return analysis_values["data"]
        else:
            self.extracted_analysis_data[analysis_type] = {}

        if analysis_type not in Mother.analysis_types.keys():
            do_critical(
                ValueError,
                f"Analysis type {analysis_type} not in accepted analysis types from Mother: {Mother.analysis_types.keys()}",
            )

        if load_params is None:
            load_params = self.default_params("load")
        if train_params is None:
            train_params = self.default_params("train")
        if analysis_params is None:
            analysis_params = self.default_params("analysis")

        load_params = self.preprocess_loading_params(load_params=load_params)

        extracted_data = {}
        animal_ids = load_params.pop("animal_ids")
        # for animal_id in tqdm(
        #     animal_ids,
        #     position=tqdm._get_free_pos(),
        #     leave=False,
        #     desc="Loading and analysing animals",
        # ):
        for animal_id in animal_ids:
            global_logger.info(
                f"Loading animal {animal_id} for analysis {analysis_type}"
            )
            # Load animal behavior data
            animals = self.get_children(
                animal_ids=animal_id, forget=True, **load_params
            )
            animal = animals.get(animal_id, None)
            if animal is None:
                global_logger.warning(
                    f"Animal {animal_id} not found in loaded Mother.get_children. Skipping."
                )
                continue

            # Train or load models
            animal: Animal
            animal.train_model(**train_params)

            # Extract data based on analysis type
            analysis_function = getattr(animal, analysis_type, None)
            if skip_error:
                try:
                    if analysis_function is not None:
                        extracted_data[animal_id] = analysis_function(**analysis_params)
                    else:
                        do_critical(
                            ValueError,
                            f"Analysis type {analysis_type} not found in Animal class.",
                        )
                except Exception as e:
                    print(f"Error: {e}")
            else:
                global_logger.info(
                    f"Running analysis {analysis_type} for animal {animal_id}"
                )
                if analysis_function is not None:
                    extracted_data[animal_id] = analysis_function(**analysis_params)
                else:
                    do_critical(
                        ValueError,
                        f"Analysis type {analysis_type} not found in Animal class.",
                    )
            if not forget:
                self.animals[animal_id] = animal

        num_run_analysis = len(self.extracted_analysis_data[analysis_type])
        self.extracted_analysis_data[analysis_type] = {
            num_run_analysis: {
                "data": extracted_data,
                "load_params": load_params,
                "train_params": train_params,
                "analysis_params": analysis_params,
            }
        }

        if (
            isinstance(analysis_params["plot"], List)
            and "Mother" in analysis_params["plot"]
        ):
            # Plot the analysis results
            plot_func_name = f"plot_{analysis_type}"
            plotting_function = getattr(self, plot_func_name, None)
            if plotting_function is None:
                do_critical(
                    ValueError,
                    f"Plotting function {plot_func_name} not found in Mother class.",
                )
            else:
                analysis_params["analysis_type"] = analysis_type
                plotting_function(
                    data=extracted_data,
                    analysis_params=analysis_params,
                    task_groups=task_groups,
                )
        return extracted_data

    def get_extracted_analysis(
        self,
        analysis_type: Literal[
            "structure_index",
            "feature_similarity",
            "cross_decode",
            "shape_similarity",
        ],
        type: Literal[
            "data", "train_params", "load_params", "analysis_params"
        ] = "data",
    ):
        if len(self.extracted_analysis_data) == 0:
            do_critical(
                ValueError,
                "No analysis data extracted yet. Please run analyse_children first.",
            )
        if analysis_type not in self.extracted_analysis_data:
            do_critical(
                ValueError,
                f"Analysis type {analysis_type} not found in extracted analysis data.",
            )
        animals_analysis_dict = self.extracted_analysis_data[analysis_type]
        if len(animals_analysis_dict) > 1:
            do_critical(
                ValueError,
                "More than one analysis type found. Please provide animals_analysis_dict.",
            )
        else:
            animals_analysis_dict = next(iter(animals_analysis_dict.values()))[type]
        return animals_analysis_dict

    def is_modeldatetask_cross_compare_output(self, dict_of_dicts) -> bool:
        """
        Detect type of dict_of_dicts structure is cross compare output.
        """
        first_value = list(dict_of_dicts.values())[0]
        key = list(first_value.keys())[0]
        split_key = key.split("_")
        if len(split_key) > 1:
            date = split_key[0]
            task = split_key[1]
            if self.owns(date, type="Date") and self.owns(task, type="Task"):
                return True
        return False

    def yield_datetask_info_from_dict_of_dicts(
        self,
        dict_of_dicts,
        wanted_tasks: List[str] = None,
    ):
        """
        Detect type of dict_of_dicts structure and output data, task, analysis data format.
        """
        if self.is_modeldatetask_cross_compare_output(dict_of_dicts):
            if len(dict_of_dicts) > 2:
                raise NotImplementedError("More than 2 tasks are not implemented yet.")
            for model_name, date_task_dict in dict_of_dicts.items():
                for ref_date_task, analysis_values in date_task_dict.items():
                    if ref_date_task == "random":
                        date, task_name = "random", "random"
                    else:
                        date, task_name = ref_date_task.split("_")
                    if wanted_tasks is not None and task_name not in wanted_tasks:
                        continue
                    yield date, task_name, model_name, analysis_values
        else:
            for date, task_dict in dict_of_dicts.items():
                for task_name, model_dict in task_dict.items():
                    if wanted_tasks is not None and task_name not in wanted_tasks:
                        continue
                    for model_name, analysis_values in model_dict.items():
                        yield date, task_name, model_name, analysis_values

    def reorder_animal_analysis(
        self,
        analysis_key: str,
        animals_analysis_dict: Dict[str, Any] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition"] = "condition",
        color_by: Literal["task", "stimulus_type"] = "task",
    ) -> Tuple[
        Dict[str, Dict[Union[str, int], Union[np.ndarray, Dict[str, np.ndarray]]]],
        pd.DataFrame,
    ]:
        """
        Reorder the data in a dictionary of dictionaries by condition and task.

        Parameters:
        ------------
        - animals_analysis_dict: dict of dicts to be reordered
            Structure:
                {animal_id:
                    {date:
                        {task_name:
                            {model_name: analysis_values
                            }
                        }
                    }
                }
        - reorder_by: str

        Returns
        -----------
        - rdata: dict of dicts
        - data_df: pandas DataFrame
            Structure:
                Columns: ['animal_id', 'date', 'model_name', 'stimulus_type', 'condition', 'task_name', 'cross_name', 'mean', 'variance']
                Each row represents a single analysis value for a specific animal, date, task, and model.
        """
        metadict = self.metadatas

        data_df = pd.DataFrame()
        rdata = {}
        for animal_id, dict_of_dicts in animals_analysis_dict.items():
            condition = metadict[animal_id].get("condition", None)
            for (
                date,
                task_name,
                model_name,
                analysis_values,
            ) in self.yield_datetask_info_from_dict_of_dicts(
                dict_of_dicts, wanted_tasks=wanted_tasks
            ):
                if task_name == "random":
                    stimulus_type = "random"
                else:
                    stimulus_type = metadict[animal_id][date]["tasks_metadata"][
                        task_name
                    ]["behavior_metadata"]["stimulus_type"]

                # create reordered dictionary
                if order_by == "condition":
                    first_key = condition
                elif order_by == "task":
                    first_key = task_name

                if color_by == "task":
                    second_key = task_name
                elif color_by == "condition":
                    second_key = condition
                elif color_by == "stimulus_type":
                    second_key = stimulus_type

                if first_key not in rdata.keys():
                    rdata[first_key] = {}

                if second_key not in rdata[first_key].keys():
                    rdata[first_key][second_key] = {}

                # format values for plotting
                if self.is_modeldatetask_cross_compare_output(dict_of_dicts):
                    # handle cross compare outputs

                    for cross_name, cross_values in analysis_values.items():
                        ca_values = cross_values[analysis_key]
                        cp_desription = cross_name.split("_")[-1].replace("to ", "")
                        if (
                            wanted_tasks is not None
                            and cp_desription not in wanted_tasks
                        ):
                            continue
                        if "to" in cross_name[:2]:
                            cp_desription = "to " + cp_desription

                        if cp_desription not in rdata[first_key][second_key].keys():
                            # save values in 2d array
                            # ca_values convert 1d array to 2d
                            ca_means = np.atleast_2d(ca_values["mean"])
                            ca_variance = np.atleast_2d(ca_values["variance"])
                            rdata[first_key][second_key][cp_desription] = {
                                "mean": ca_means,
                                "variance": ca_variance,
                            }

                            # create long format pandas dataframe
                        else:
                            previous_values = rdata[first_key][second_key][
                                cp_desription
                            ]
                            ca_means = extend_vstack(
                                previous_values["mean"], ca_values["mean"]
                            )
                            ca_variance = extend_vstack(
                                previous_values["variance"], ca_values["variance"]
                            )
                            new_values = {
                                "mean": ca_means,
                                "variance": ca_variance,
                            }
                            rdata[first_key][second_key][cp_desription] = new_values

                        task_name_part, task_number = search_split(r"\d+", task_name)
                        if task_number:
                            task_number = int(task_number)
                        else:
                            task_number = None
                        data_df = pd.concat(
                            [
                                data_df,
                                pd.DataFrame(
                                    {
                                        "animal": animal_id,
                                        "date": date,
                                        "model_name": model_name,
                                        "stimulus_type": stimulus_type,
                                        "condition": condition,
                                        "task": task_name,
                                        "task_name": task_name_part,
                                        "task_number": task_number,
                                        "stimulus_type": stimulus_type,
                                        "cross_name": cp_desription,
                                        "values": cross_values,
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )
                        raise NotImplementedError(
                            "Cross compare outputs for pandas dataframe is not tested and probably should be changed, check below dataframes entry"
                        )

                else:
                    # handle single model outputs
                    sweep_values, values = check_extract_from_parameter_sweep(
                        analysis_values, analysis_key
                    )
                    line_label = (
                        task_name if not second_key == task_name else stimulus_type
                    )
                    if line_label not in rdata[first_key][second_key].keys():
                        values_2d = np.atleast_2d(values)
                        rdata[first_key][second_key][line_label] = values_2d
                    else:
                        previous_values = rdata[first_key][second_key][line_label]
                        new_values = extend_vstack(previous_values, values)
                        rdata[first_key][second_key][line_label] = new_values

                    task_name_part, task_number = search_split(r"\d+", task_name)
                    if task_number:
                        task_number = int(task_number)
                    else:
                        task_number = None

                    # add analysis_values: Dict[str, Dict[str, Any]] to data_df for every entry in analysis_values creating a single entry with mutliple columns
                    data_frame_entry_dict = {
                        "animal": animal_id,
                        "date": date,
                        "model_name": model_name,
                        "stimulus_type": stimulus_type,
                        "condition": condition,
                        "task": task_name,
                        "task_name": task_name_part,
                        "task_number": task_number,
                        "sweep_value": sweep_values,
                    }
                    if sweep_values is not None:
                        for sweep_value, values in analysis_values.items():
                            # add every entry in values dict to data_df
                            data_frame_entry_dict["sweep_value"] = sweep_value
                            for key, value in values.items():
                                if is_array_like(value) or isinstance(value, tuple):
                                    value = [value]
                                data_frame_entry_dict[key] = value
                            if isinstance(values, dict):
                                data_df = pd.concat(
                                    [
                                        data_df,
                                        pd.DataFrame(data_frame_entry_dict),
                                    ],
                                    ignore_index=True,
                                )
        return rdata, data_df

    def prepare_plotting_data(
        self,
        data: Dict[str, Any] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition", "task"] = "condition",
        color_by: Literal["task", "stimulus_type"] = "task",
        analysis_params: Dict[str, Any] = None,
        analysis_type: Literal[
            "structure_index",
            "feature_similarity",
            "cross_decode",
            "shape_similarity",
        ] = None,
    ):
        if analysis_params is None:
            if analysis_type is None:
                do_critical(
                    ValueError,
                    f"Either data or analysis_params must be provided for preparing plotting data in {self.__class__}.",
                )

            analysis_params = self.get_extracted_analysis(
                analysis_type=analysis_type, type="analysis_params"
            )
        else:
            if analysis_type is not None:
                if analysis_type != analysis_params["analysis_type"]:
                    do_critical(
                        ValueError,
                        f"Analysis type {analysis_type} does not match the analysis type in analysis_params: {analysis_params['analysis_type']}",
                    )

            if "analysis_type" not in analysis_params:
                do_critical(
                    ValueError,
                    f"Analysis type not found in analysis_params: {analysis_params}",
                )

            analysis_type = analysis_params["analysis_type"]
            if analysis_type not in Mother.analysis_types.keys():
                do_critical(
                    ValueError,
                    f"Analysis type {analysis_type} not in accepted analysis types from Mother: {Mother.analysis_types.keys()}",
                )

        if data is None:
            data = self.get_extracted_analysis(analysis_type=analysis_type)

        analysis_key = Mother.analysis_types[analysis_type]["key"]
        if analysis_key is None:
            do_critical(
                ValueError,
                f"Analysis type {analysis_type} not found in accepted analysis types from Mother: {Mother.analysis_types.keys()}",
            )

        reordered_data, data_df = self.reorder_animal_analysis(
            animals_analysis_dict=data,
            analysis_key=analysis_key,
            wanted_tasks=wanted_tasks,
            order_by=order_by,
            color_by=color_by,
        )
        return reordered_data, data_df, analysis_params

    def pre_plot(
        self,
        data: Dict[str, Any] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition", "task"] = "condition",
        color_by: Literal["task", "stimulus_type"] = "task",
        analysis_params: Dict[str, Any] = None,
        analysis_type: Literal[
            "structure_index",
            "feature_similarity",
            "cross_decode",
            "shape_similarity",
        ] = None,
    ):
        """
        Prepares the data for plotting.
        """
        wanted_tasks = make_list_ifnot(wanted_tasks)
        rdata, data_df, analysis_params = self.prepare_plotting_data(
            data=data,
            wanted_tasks=wanted_tasks,
            order_by=order_by,
            color_by=color_by,
            analysis_params=analysis_params,
            analysis_type=analysis_type,
        )
        if analysis_params is None:
            do_critical(f"No analysis parameters provided for plotting.")

        conditions = list(rdata.keys())

        wanted_tasks = (
            list(rdata[conditions[0]].keys()) if wanted_tasks is None else wanted_tasks
        )

        if "space" in analysis_params:
            if analysis_params["space"] not in ["fluorescence", "binarized"]:
                space = analysis_params["manifolds_pipeline"].upper()
            else:
                if analysis_params["space"] == "fluorescence":
                    space = "Activity"
                else:
                    space = "Binarized Activity"
        else:
            space = analysis_params["manifolds_pipeline"].upper()

        movement_state = self.get_extracted_analysis(
            analysis_type=analysis_type, type="train_params"
        )["movement_state"]
        shuffle = self.get_extracted_analysis(
            analysis_type=analysis_type, type="train_params"
        )["shuffle"]

        load_params = self.get_extracted_analysis(
            analysis_type=analysis_type, type="load_params"
        )
        if "data_filter" in load_params:
            data_filter = load_params.get("data_filter", None)
            if len(data_filter) == 0:
                data_filter = None
            data_filter_short_txt = (
                DataFilter.filter_dict_to_str(data_filter) if data_filter else None
            )

        additional_description = f" {movement_state}{' shuffled' if shuffle else ''}{data_filter_short_txt if data_filter_short_txt else ''}"

        return (
            rdata,
            data_df,
            analysis_params,
            conditions,
            space,
            wanted_tasks,
            additional_description,
        )

    def plot_structure_index(
        self,
        data: Dict[str, Any] = None,
        task_groups: Optional[Dict[str, List[str]]] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition", "task"] = "condition",
        color_by: Literal["task", "stimulus_type"] = "task",
        save_dir: Optional[Union[str, Path]] = None,
        analysis_params: Dict[str, Any] = None,
        additional_title: str = "",
    ):
        """
        Plots the structure index for the given data sorted by condition.

        Parameters:
            - task_groups (Optional[Dict[str, Dict[str, List[str]]]]): Dictionary for custom task grouping.
            If None, autogrouping is applied based on `group_by` and `compare_by`.
            Outer keys match unique values in `group_by`. Inner dictionaries have subgroup names as keys
            and lists of values (from `compare_by`) as values. Example:
                {
                    'condition1': {
                        'task_group1': ['task1', 'task2'],
                        'task_group2': ['task3']
                    },
                    'condition2': {
                        'task_group1': ['task1', 'task2']
                    }
                }
        """
        task_title = "" if wanted_tasks is None else f" {wanted_tasks}"
        analysis_type = "structure_index"
        analysis_key = Mother.analysis_types[analysis_type]["key"]
        (
            rdata,
            data_df,
            analysis_params,
            conditions,
            space,
            wanted_tasks,
            additional_description,
        ) = self.pre_plot(
            data=data,
            wanted_tasks=wanted_tasks,
            order_by=order_by,
            color_by=color_by,
            analysis_params=analysis_params,
            analysis_type=analysis_type,
        )
        param_range = analysis_params["params"]["n_neighbors"]
        title = f"Structural Indices in {space} Space for different tasks for {conditions} animals {task_title}{additional_description}"

        title = title.replace("'", "")
        if len(wanted_tasks) == 1:
            to_show = ["mean", "samples"]
        else:
            to_show = ["mean", "std"]

        # # # create a summary plot of all values from the parameter sweep
        ax = Vizualizer.plot_1d_line_dict_of_dicts(
            rdata,
            title=title,
            to_show=to_show,
            ylabel="Structural Index",
            xlabel="Number Neighbors",
            additional_title=additional_title,
            xticks=list(param_range),
            xtick_pos=np.arange(len(param_range)),
            save_dir=save_dir or self.figures_dir,
        )

        # Plot structure indices summary for different ranges if wanted_tasks is bigger than 1
        if len(wanted_tasks) > 1:
            # extract indices for given range
            local_indices = np.where((param_range >= 3) & (param_range < 25))[0]
            mid_indices = np.where((param_range >= 43) & (param_range < 103))[0]
            global_indices = np.where((param_range >= 203) & (param_range < 300))[0]
            param_ranges = {
                "local": local_indices,
                "mid": mid_indices,
                "global": global_indices,
            }

            # create dictionary for plotting with reordered data
            global_labels = []
            ranges_dict = {}
            for i, (structure_type, neigh_range) in enumerate(param_ranges.items()):
                values_by_condition = {}
                for condition in conditions:
                    values_by_condition[condition] = {}
                    all_task_stats = []
                    labels = []
                    sorted_dict = sort_dict(rdata[condition])
                    for task, _ in sorted_dict.items():
                        # for task in single_tasks:

                        box_type = list(rdata[condition][task].keys())[0]
                        task_si = rdata[condition][task][box_type]
                        # extend task_si to have the same number as param_range
                        if len(task_si) < len(param_range):
                            len_diff = len(param_range) - task_si.shape[1]
                            nan_array = np.full(
                                (task_si.shape[0], len_diff),
                                np.nan,
                            )
                            task_si = np.hstack((task_si, nan_array))
                        # extract values range
                        # create array with maximum possible value first
                        filtered_numbers = task_si[:, neigh_range]
                        # all_task_stats["mean"].append(mean)
                        # all_task_stats["std"].append(std)
                        all_task_stats.append(filtered_numbers)
                        labels += [task]

                    if len(global_labels) != 0 and labels != global_labels:
                        do_critical(
                            ValueError, "Labels do not match across conditions."
                        )

                    # sort all_task_stats by name ensure that number FS10 is after FS9
                    values_by_condition[condition][condition] = all_task_stats
                ranges_key = f"{structure_type} neighbors {param_range[neigh_range[0]]}-{param_range[neigh_range[-1]]}"
                ranges_dict[ranges_key] = values_by_condition

            ####################################################################################
            #     calculate  AUC to describe the behavior of a function (better than average)  #
            ####################################################################################

            power_func_param_names = {
                "k": "slope",
                "n": "exponent",
                "a": "intercept",
            }
            # Initialize DataFrame to store results
            animal_param_sweep_stats_df = pd.DataFrame(
                columns=list(data_df.columns[:-5])
                + ["values", "auc"]
                + list(power_func_param_names.values())
            )

            # Traverse animal_id and task_name
            for animal_id, animal_data in data_df.groupby("animal"):
                for task_name, task_data in animal_data.groupby("task"):
                    # Extract analysis_key values
                    analysis_key_values = task_data[analysis_key].values.flatten()

                    # Skip if insufficient data
                    if len(analysis_key_values) < 2:
                        continue

                    # Initialize row for results
                    row = {col: task_data[col].iloc[0] for col in data_df.columns[:-5]}

                    # Fit a power function to the data
                    auc, best_fit_dict = get_auc(
                        x=param_range,
                        y=analysis_key_values,
                        functions="power",
                        plot=False,
                        return_fit=True,
                    )
                    row["auc"] = auc

                    power_func_params = best_fit_dict.get("param", np.nan)
                    for param_name, param_description in power_func_param_names.items():
                        row[param_description] = power_func_params[param_name]

                    row["values"] = analysis_key_values.tolist()
                    # Append row to DataFrame
                    animal_param_sweep_stats_df = pd.concat(
                        [animal_param_sweep_stats_df, pd.DataFrame([row])],
                        ignore_index=True,
                    )

            # # # # # # Compare area under the curve (AUC) change over time using mixed ANOVA

            grouped_df = linebar_df_group_plot(
                df=animal_param_sweep_stats_df,
                title=f"Lineplot AUC Structural Indices for {space} Space <br> {conditions} animals{additional_description}",
                value_col="auc",
                group_by="condition",
                compare_by="task",
                fit_line="linear",
                # compare_by_filter=wanted_tasks,
                show_std=False,
                groups=task_groups,
                save_dir=save_dir or self.figures_dir,
            )
            # convert grouped_df to normal dataframe
            tmp_df = grouped_df.obj

            from calculations import mixedlm_learning_curve_analysis

            # Perform mixed linear model analysis for every group pair
            unique_task_group_names = tmp_df["group_name"].unique()
            # create pairs of all unique_task_group_names
            from itertools import combinations

            unique_task_group_name_pairs = list(
                combinations(unique_task_group_names, 2)
            )

            # # # # # Perform mixed linear model analysis for every group pair
            for unique_task_group_name_pair in unique_task_group_name_pairs:
                filtered_df = tmp_df[
                    (tmp_df["group_name"] == unique_task_group_name_pair[0])
                    | (tmp_df["group_name"] == unique_task_group_name_pair[1])
                ]
                # filter for group_name
                print(f"\nAnalyzing group pair: {unique_task_group_name_pair}")
                result = mixedlm_learning_curve_analysis(
                    df=filtered_df,
                    additional_title=unique_task_group_name_pair,
                    subject_col="animal",
                    outcome_col="auc",
                    # group_col="condition",
                    group_col="group_name",
                    time_col="group_task_number",
                    plot_predicted=True,
                    show_diagnostics=True,
                )

            # plot plots data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # improve dataframe by adding useful columns
            animal_param_sweep_stats_df = Mother.improve_animals_df(
                df=animal_param_sweep_stats_df,
                group_by="condition",
                compare_by="task",
                task_groups=task_groups,
            )

            # Combine animal_param_sweep_stats_df to task_condition_df grouping by task and condition
            for value_col in ["auc"]:  # + list(power_func_param_names.values()):

                # None to Check if before before manipulation some information representation is already significantly different
                # # # # # for group_by_case in [None, "condition"]:
                for group_by_case in ["condition"]:
                    group_by_txt = (
                        str(conditions) if group_by_case is not None else "all"
                    )

                    # Run combined paired and unpaired tests with no correction
                    result = statistical_comparison(
                        df=animal_param_sweep_stats_df,
                        pair_name_col="animal",
                        groups=conditions,
                        labels=labels,
                        test_type=["paired", "unpaired"],
                        method="auto",
                        additional_title=f"NO CORRECTION !!! {value_col.upper()} for {space} Space - {group_by_txt} animals{additional_description}",
                        value_col=value_col,
                        compare_by="task",
                        group_by=group_by_case,
                        correction_method="none",
                        plot_heatmaps=True,
                        plot_violins=True,
                        save_dir=save_dir or self.figures_dir,
                    )

                    # Run combined paired and unpaired tests with Holm correction
                    result = statistical_comparison(
                        df=animal_param_sweep_stats_df,
                        pair_name_col="animal",
                        groups=conditions,
                        labels=labels,
                        test_type=["paired", "unpaired"],
                        method="auto",
                        additional_title=f"{value_col.upper()} for {space} Space - {group_by_txt} animals{additional_description}",
                        value_col=value_col,
                        compare_by="task",
                        group_by=group_by_case,
                        correction_method="holm",
                        plot_heatmaps=True,
                        plot_violins=True,
                        save_dir=save_dir or self.figures_dir,
                    )

            # # # ####################################################################################
            # # # #     do wilcoxon paired test for significance inside experimental conditions with merged ranges     #
            # # # ####################################################################################
            # # # from scipy.stats import wilcoxon

            # # # # data_df columns:
            # # # # "animal": [animal_id],
            # # # # "date": [date],
            # # # # "model_name": [model_name],
            # # # # "stimulus_type": [stimulus_type],
            # # # # "condition": [condition],
            # # # # "task": [task],
            # # # # "sweep_values": [sweep_values],
            # # # # analysis_key: [values],

            # # # # tasks are labels
            # # # tasks = labels
            # # # # unique animal ids
            # # # animals = data_df["animal"].unique()

            # # # # create 2d dataframe for wilcoxon test
            # # # wil_heatmaps = {}
            # # # for range_str, param_range_part in param_ranges.items():
            # # #     range_label = f"{range_str} neighbors {param_range[param_range_part[0]]}-{param_range[param_range_part[-1]]}"
            # # #     wil_heatmaps[range_label] = {}
            # # #     for condition in conditions:
            # # #         # filter data_df for condition
            # # #         condition_data_df = data_df[data_df["condition"] == condition]

            # # #         # create Task based groups with values for wilcoxon test
            # # #         ## filter condition_data_df for tasks
            # # #         task_animal_groups = {}
            # # #         for task in tasks:
            # # #             task_condition_data_df = condition_data_df[
            # # #                 condition_data_df["task"] == task
            # # #             ]
            # # #             task_animal_groups[task] = task_condition_data_df

            # # #         # iterate over task pairs and perform wilcoxon test
            # # #         wil_heatmap_df = pd.DataFrame(
            # # #             index=tasks, columns=tasks, dtype=float
            # # #         )
            # # #         wil_heatmap_df = wil_heatmap_df.fillna(np.nan)
            # # #         for task1, task_data1 in task_animal_groups.items():
            # # #             for task2, task_data2 in task_animal_groups.items():
            # # #                 if task1 == task2:
            # # #                     continue
            # # #                 # only compute upper triangle of the matrix
            # # #                 if not np.isnan(wil_heatmap_df.loc[task2, task1]):
            # # #                     continue

            # # #                 # remove animals that are not in both tasks
            # # #                 common_animals = set(task_data1["animal"]).intersection(
            # # #                     set(task_data2["animal"])
            # # #                 )
            # # #                 if len(common_animals) < 2:
            # # #                     continue
            # # #                 task_data1_common = task_data1[
            # # #                     task_data1["animal"].isin(common_animals)
            # # #                 ]
            # # #                 task_data2_common = task_data2[
            # # #                     task_data2["animal"].isin(common_animals)
            # # #                 ]
            # # #                 # ensure sorting of tasks is the same
            # # #                 task_data1_common = task_data1_common.sort_values(
            # # #                     by=["animal"]
            # # #                 )
            # # #                 task_data2_common = task_data2_common.sort_values(
            # # #                     by=["animal"]
            # # #                 )
            # # #                 # perform wilcoxon test
            # # #                 p_values = []
            # # #                 for param_idx in param_range_part:
            # # #                     sweep_range_value = param_range[param_idx]
            # # #                     # filter for sweep_value
            # # #                     task_data1_common_param_values = task_data1_common[
            # # #                         analysis_key
            # # #                     ][task_data1_common["sweep_value"] == sweep_range_value]
            # # #                     task_data2_common_param_values = task_data2_common[
            # # #                         analysis_key
            # # #                     ][task_data2_common["sweep_value"] == sweep_range_value]

            # # #                     stat, p_value = wilcoxon(
            # # #                         task_data1_common_param_values,
            # # #                         task_data2_common_param_values,
            # # #                     )
            # # #                     p_values.append(p_value)

            # # #                 averaged_p_value = np.mean(p_values)
            # # #                 wil_heatmap_df.loc[task1, task2] = (
            # # #                     averaged_p_value
            # # #                 )
            # # #                 wil_heatmap_df.loc[task2, task1] = (
            # # #                     averaged_p_value
            # # #                 )

            # # #         wil_heatmaps[range_label][condition] = wil_heatmap_df

            # # # # make figsize depending on number of tasks
            # # # mult_by = len(labels) / 10
            # # # figsize = (
            # # #     10 * mult_by,
            # # #     6 * mult_by,
            # # # )

            # # # Vizualizer.plot_heatmap_dict_of_dicts(
            # # #     data=wil_heatmaps,
            # # #     title=f"Wilcoxon Test p-values RANGES for {space} Space - {conditions} animals",
            # # #     additional_title=additional_description,
            # # #     labels=labels,
            # # #     colorbar_ticks=[0, 0.025, 0.05],
            # # #     colorbar_ticks_labels=["0", "0.025", "0.05"],
            # # #     first_direction="y",
            # # #     vmin=0,
            # # #     vmax=0.05,
            # # #     colorbar_label="p-value",
            # # #     xlabel="Tasks",
            # # #     ylabel="Tasks",
            # # #     sharex=False,
            # # #     sharey=False,
            # # #     cmap="viridis_r",  # Use a perceptually uniform colormap
            # # #     figsize=figsize,
            # # #     save_dir=save_dir or self.figures_dir,
            # # #     as_pdf=False,  # Optional: save as PDF
            # # # )
        return rdata, data_df

    def plot_shape_similarity():

        ## create a 2D heatmap of shape similarity between different tasks using the following function
        # from temporary import analyze_and_plot_distances
        # TODO: plot_shape_similarity
        # TODO: from temporary import analyze_and_plot_distances is not fit to the current shape similarity structure and needs to be adapted
        pass

    def plot_cross_decode(
        self,
        data: Dict[str, Any] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition", "task"] = "condition",
        color_by: Literal["condition", "task", "stimulus_type"] = "task",
        save_dir: Optional[Union[str, Path]] = None,
        analysis_params: Dict[str, Any] = None,
        plot: Literal["single", "all"] = "all",
        additional_title: str = "",
    ):
        """
        Plots the structure index for the given data sorted by condition.
        """
        (
            rdata,
            data_df,
            analysis_params,
            conditions,
            space,
            wanted_tasks,
            additional_description,
        ) = self.pre_plot(
            data=data,
            wanted_tasks=wanted_tasks,
            order_by=order_by,
            color_by=color_by,
            analysis_params=analysis_params,
            analysis_type="cross_decode",
        )
        for condition, condition_data in rdata.items():
            # summarize condition data for each task
            plot_analysis_data = {}
            for task_name, task_data in condition_data.items():
                task_cross_values = {}
                for to_task_name, to_task_values in task_data.items():
                    mean_analysis_array = {
                        "mean": np.nanmean(to_task_values["mean"]),
                        "variance": np.nanmean(to_task_values["variance"]),
                    }
                    task_cross_values[to_task_name] = mean_analysis_array
                plot_analysis_data[task_name] = task_cross_values

            if order_by == "condition":
                plot_condition = condition
                task_title = "" if wanted_tasks is None else f" {wanted_tasks}"
            elif order_by == "task":
                plot_condition = list(condition_data.keys())[0]
                task_title = condition

            title = f"Cross Decoding in {space} Space for different tasks for {plot_condition} animals{additional_description}"
            title = title.replace("'", "")
            if plot == "single":
                # format decoding ouput for plotting
                for i, (ref_model_name, ref_to_decodings) in enumerate(
                    plot_analysis_data.items()
                ):
                    Vizualizer.barplot_from_dict(
                        ref_to_decodings,
                        title=title,
                        ylabel="RMSE (cm)",
                        xlabel="Models",
                        # additional_title=additional_title
                        # + f" reference - Task {xticks_single[0]}",
                        save_dir=save_dir or self.figures_dir,
                    )
            else:
                Vizualizer.barplot_from_dict_of_dicts(
                    plot_analysis_data,
                    title=title,
                    additional_title=additional_title,
                    save_dir=save_dir or self.figures_dir,
                )

        return rdata

    def plot_feature_similarity(
        self,
        data: Dict[str, Any] = None,
        wanted_tasks: List[str] = None,
        order_by: Literal["condition", "task"] = "condition",
        color_by: Literal["task", "stimulus_type"] = "task",
        save_dir: Optional[Union[str, Path]] = None,
        analysis_params: Dict[str, Any] = None,
        additional_title: str = "",
    ):
        """
        Plots the structure index for the given data sorted by condition.
        """
        analysis_type = "feature_similarity"
        (
            rdata,
            data_df,
            analysis_params,
            conditions,
            space,
            wanted_tasks,
            additional_description,
        ) = self.pre_plot(
            data=data,
            wanted_tasks=wanted_tasks,
            order_by=order_by,
            color_by=color_by,
            analysis_params=analysis_params,
            analysis_type=analysis_type,
        )

        raise NotImplementedError(
            "Feature similarity plotting is not implemented yet. Please check the code."
        )

        for condition, condition_data in rdata.items():
            # summarize condition data for each task
            plot_analysis_data = {}
            for task_name, task_data in condition_data.items():
                task_cross_values = {}
                for to_task_name, to_task_values in task_data.items():
                    mean_analysis_array = {
                        "mean": np.nanmean(to_task_values["mean"]),
                        "variance": np.nanmean(to_task_values["variance"]),
                    }
                    task_cross_values[to_task_name] = mean_analysis_array
                plot_analysis_data[task_name] = task_cross_values

            if order_by == "condition":
                plot_condition = condition
                task_title = "" if wanted_tasks is None else f" {wanted_tasks}"
            elif order_by == "task":
                plot_condition = list(condition_data.keys())[0]
                task_title = condition

            title = f"Cross Decoding in {space} Space for different tasks for {plot_condition} animals{additional_description}"
            title = title.replace("'", "")
            if plot == "single":
                # format decoding ouput for plotting
                for i, (ref_model_name, ref_to_decodings) in enumerate(
                    plot_analysis_data.items()
                ):
                    Vizualizer.barplot_from_dict(
                        ref_to_decodings,
                        title=title,
                        ylabel="RMSE (cm)",
                        xlabel="Models",
                        # additional_title=additional_title
                        # + f" reference - Task {xticks_single[0]}",
                        save_dir=save_dir or self.figures_dir,
                    )
            else:
                Vizualizer.barplot_from_dict_of_dicts(
                    plot_analysis_data,
                    title=title,
                    additional_title=additional_title,
                    save_dir=save_dir or self.figures_dir,
                )

        return rdata


class Animal(MetaClass):
    """Represents an animal in the dataset."""

    descriptive_metadata_keys = []
    needed_attributes: List[str] = ["animal_id", "dob"]

    def __init__(
        self,
        animal_id: str,
        root_dir,
        animal_dir=None,
        model_settings=None,
        data_filter={},
    ):
        self.dir: Path = animal_dir or root_dir.joinpath(animal_id)
        super().__init__(self.dir, data_filter=data_filter)
        self.id: str = animal_id
        self.name: str = animal_id
        self.dob: str = None
        self.sex: str = None
        self.yaml_path: Path = self.dir.joinpath(f"{animal_id}.yaml")
        self.sessions: Dict[str, Session] = {}
        self.model_settings: Dict[str, Any] = model_settings
        self.metadata: Dict[str, Any] = {}
        self.load_metadata()

    @property
    def child_obj(self):
        return self.sessions

    def load_metadata(self, yaml_path=None, name_parts=None):
        self.metadata = load_yaml_data_into_class(
            cls=self,
            yaml_path=yaml_path,
            name_parts=name_parts,
            needed_attributes=Animal.needed_attributes,
        )

    def add_session(
        self,
        dates: Union[str, List[str]],
        model_settings: Dict[str, Dict[str, Union[str, float]]] = None,
        data_filter: Union[Dict[str, str], DataFilter] = {},
    ):
        """
        Adds one or multiple sessions to the animal object.
        """

        model_settings = self.model_settings if not model_settings else model_settings

        data_filter = self.may_passon_data_filter(data_filter)

        dates = make_list_ifnot(dates)

        for date in dates:
            session = Session(
                animal=self,
                date=date,
                model_settings=model_settings,
                data_filter=data_filter,
            )

            if session:
                session.pday = (num_to_date(session.date) - num_to_date(self.dob)).days
                self.sessions[date] = session
                self.sessions = sort_dict(self.sessions)
            else:
                global_logger.warning(f"Session {self.id} date: {date} not found.")
                print(f"Skipping {self.animal_id} {date}")

        self.sessions = sort_dict(self.sessions)
        return self.sessions

    def train_model(
        self,
        dates: Union[str, List[str]] = None,
        task_names: Union[str, List[str]] = None,
        shuffle: bool = False,
        movement_state: str = "moving",
        split_ratio: float = 1,
        name_comment: str = None,
        transformation: str = None,
        manifolds_pipeline: str = "cebra",
        create_embeddings: bool = True,
        verbose: bool = False,
        model_type: Literal["time", "behavior", "hybrid"] = "behavior",
        model_settings: Dict[str, Dict[str, Union[str, float]]] = None,
        behavior_data_types: Literal[
            "position", "distance", "moving", "velocity", "acceleration", "stimulus"
        ] = ["position", "moving"],
        regenerate: bool = False,
        plot: bool = False,
    ):
        """
        Train the model for all sessions or a specific session.
        """
        # take function parameters and put into dictionary for faster access

        if dates is None:
            dates = list(self.sessions.keys())
        dates = make_list_ifnot(dates)
        model_settings = model_settings or self.model_settings

        params = create_params_dict(exclude=["self", "dates"], **locals())
        for date, session in self.sessions.items():
            if date not in dates:
                continue
            session.train_model(**params)

    def load_all_data(
        self,
        behavior_data_types=["position"],
        regenerate=False,
        regenerate_plots=False,
        plot=None,
    ):
        data = {}
        for date, session in self.sessions.items():
            data[date] = session.load_all_data(
                behavior_data_types=behavior_data_types,
                regenerate=regenerate,
                regenerate_plots=regenerate_plots,
                plot=plot,
            )
        return data

    def get_pipeline_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        manifolds_pipeline: str = "cebra",
    ):
        session: Session
        models = {}
        for session_date, session in self.sessions.items():
            session_models = session.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
            models[session_date] = session_models
        return models

    def filter_sessions(
        self, wanted_properties: Dict[str, Dict[str, Union[str, List]]] = None
    ):
        """

        Filter the available session tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {
                    "session": {
                        # "date": ["20211022", "20211030", "20211031"],
                    },
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            print("No wanted properties given. Returning tasks sessions")
            wanted_properties = {}
        filtered_tasks = {}

        for session_date, session in self.sessions.items():
            wanted = True
            if "session" in wanted_properties:
                wanted = wanted_object(session, wanted_properties["session"])

            if wanted:
                filtered_session_tasks = session.filter_tasks(wanted_properties)
                filtered_tasks.update(filtered_session_tasks)
        return filtered_tasks

    def plot_task_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        train_or_test: str = "train",
        plot: Union[str, List[str]] = ["embedding", "loss"],
        to_2d: bool = False,
        behavior_type: str = "position",
        manifolds_pipeline: str = "cebra",
        embeddings_title: Optional[str] = None,
        losses_title: Optional[str] = None,
        losses_coloring: Optional[str] = "rainbow",
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        """
        Plot model embeddings and losses nearby each other for every task in every session.

        Only possible if a unique model is found for each task. Losses can be
        colored by rainbow, distinct, or mono colors.

        Parameters
        ----------
        model_naming_filter_include : list, optional
            A list of lists containing the model naming parts to include (default is None).
            If None, all models will be included, which will result in an error if more than one model is found.
            Options:
                - single string: only one property has to be included
                - list of strings: all properties have to be included
                - list of lists of strings: Either one of the properties in the inner list has to be included

        model_naming_filter_exclude : list, optional
            A list of lists containing the model naming parts to exclude (default is None).
            If None, no models will be excluded.
            Options:
                - single string: only one property has to be excluded
                - list of strings: all properties have to be excluded
                - list of lists of strings: Either one of the properties in the inner list has to be excluded

        plot : str or list oi str, optional
            A list containing the plots to show (default is ["embedding", "loss"]).

        train_or_test : str, optional
            The data type to plot (default is "train").

        to_2d : bool, optional
            If True, the embeddings will be plotted in 2D (default is False).

        behavior_type : str, optional
            The behavior type to use for labeling the embeddings (default is "position").
        """
        plot = make_list_ifnot(plot)

        info = self.get_model_information(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline=manifolds_pipeline,
            behavior_data_type=behavior_type,
            train_or_test=train_or_test,
            wanted_information=["model"],
        )
        embeddings, losses, labels, train_labels = (
            {},
            {},
            {"name": behavior_type, "labels": []},
            {"name": behavior_type, "labels": []},
        )

        for key_list, model in traverse_dicts(info):
            movement_state = model.train_info["movement_state"]
            transformation = model.train_info["transformation"]
            transformation = "" if transformation is None else transformation
            task_name = f"{key_list[1]}|{movement_state}|{transformation}"
            emb = model.get_data(train_or_test=train_or_test, type="embedding")
            loss = model.get_loss()
            label = model.get_data(train_or_test=train_or_test, type="behavior")
            train_label = model.get_data(train_or_test="train", type="behavior")

            embeddings[task_name] = emb
            losses[task_name] = loss
            labels["labels"].append(label)
            train_labels["labels"].append(train_label)

        min_val_labels, max_val_labels = find_min_max_values(train_labels["labels"])

        # plot embeddings
        embeddings_title = (
            f"{manifolds_pipeline.upper()} embeddings {self.id} - {train_or_test}"
            if not embeddings_title
            else embeddings_title
        )
        embeddings_title = add_descriptive_metadata(
            text=embeddings_title, comment=title_comment, metadata=None, keys=None
        )

        viz = Vizualizer(root_dir=self.dir.parent)
        if "embedding" in plot:
            # embeddings nearby each other
            viz.plot_multiple_embeddings(
                embeddings=embeddings,
                labels=labels,
                min_val=min_val_labels,
                max_val=max_val_labels,
                title=embeddings_title,
                projection="2d" if to_2d else "3d",
                show_hulls=False,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )

        if "loss" in plot:
            losses_title = (
                f"{manifolds_pipeline.upper()} losses {self.id}"
                if not losses_title
                else losses_title
            )
            losses_title = add_descriptive_metadata(
                text=losses_title, comment=title_comment, metadata=None, keys=None
            )
            # losses nearby each other
            viz.plot_losses(
                losses=losses,
                title=losses_title,
                alpha=0.8,
                figsize=figsize,
                coloring_type=losses_coloring,
                plot_iterations=False,
                as_pdf=as_pdf,
            )


class Session(MetaClass):
    """Represents a session in the dataset."""

    tasks: Dict[str, Task]
    needed_attributes: List[str] = ["tasks_metadata"]

    def __init__(
        self,
        animal: Animal,
        date: str,
        data_dir=None,
        model_dir=None,
        model_settings={},
        data_filter: Union[Dict[str, Any], DataFilter] = {},
    ):
        self.dir: Path = Path(animal.dir.joinpath(date))
        super().__init__(self.dir, data_filter=data_filter)
        self.data_dir: Path = data_dir or self.dir
        self.model_dir: Path = Path(model_dir or self.dir.joinpath("models"))
        self.animal = animal
        self.name = date
        self.date: str = date
        self.id: str = f"{self.animal.id}_{self.date}"
        self.model_settings: Dict[str, Dict] = model_settings
        self.yaml_path: Path = self.dir.joinpath(f"{self.date}.yaml")
        self.tasks_metadata: Dict[str, Dict] = None
        self.tasks: Dict[str, Task] = {}
        self.metadata: Dict[str, Any] = {}
        self.load_metadata()

    @property
    def child_obj(self):
        return self.tasks

    def load_metadata(self, yaml_path=None, name_parts=None):
        self.metadata = load_yaml_data_into_class(
            cls=self,
            yaml_path=yaml_path,
            name_parts=name_parts,
            needed_attributes=Session.needed_attributes,
        )

    def add_task(
        self,
        task_names: Union[str, List[str]],
        metadata: Dict[str, Dict] = None,
        model_settings: Dict[str, Dict] = None,
        data_filter: Union[Dict[str, Any], DataFilter] = {},
    ):

        task_names = make_list_ifnot(task_names)
        data_filter = self.may_passon_data_filter(data_filter)

        model_settings = model_settings or self.model_settings
        for task_name in task_names:
            success = check_correct_metadata(
                string_or_list=self.tasks_metadata.keys(), name_parts=task_name
            )
            if success:
                metadata = self.tasks_metadata[task_name]
                task_object = Task(
                    session=self,
                    task_name=task_name,
                    metadata=metadata,
                    model_settings=model_settings,
                    data_filter=data_filter,
                )
                self.tasks[task_name] = task_object
            else:
                global_logger.warning(
                    f"Task {task_name} properties not found in metadata."
                )
                print(f"Task {task_name} properties not found in metadata. Skipping.")
        self.tasks = sort_dict(self.tasks)
        return self.tasks

    def train_model(
        self,
        task_names: Union[str, List[str]] = None,
        shuffle: bool = False,
        movement_state: str = "moving",
        split_ratio: float = 1,
        name_comment: str = None,
        transformation: str = None,
        manifolds_pipeline: str = "cebra",
        create_embeddings: bool = True,
        verbose: bool = False,
        model_type: Literal["time", "behavior", "hybrid"] = "behavior",
        model_settings: Dict[str, Dict[str, Union[str, float]]] = None,
        behavior_data_types: Literal[
            "position", "distance", "moving", "velocity", "acceleration", "stimulus"
        ] = ["position", "moving"],
        regenerate: bool = False,
        plot: bool = False,
    ):
        """ """
        if task_names is None:
            task_names = list(self.tasks.keys())
        task_names = make_list_ifnot(task_names)
        model_settings = model_settings or self.model_settings

        params = create_params_dict(exclude=["self", "task_names"], **locals())
        for task_name, task in self.tasks.items():
            if task_name not in task_names:
                continue
            task.train_model(**params)

    def load_all_data(
        self,
        behavior_data_types=["position"],
        regenerate=False,
        regenerate_plots=False,
        plot=None,
    ):
        data = {}
        for task_name, task in self.tasks.items():
            data[task_name] = task.load_data(
                behavior_data_types=behavior_data_types,
                regenerate=regenerate,
                regenerate_plots=regenerate_plots,
                plot=plot,
            )
        return data

    def get_pipeline_models(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        manifolds_pipeline: str = "cebra",
    ):
        task: Task
        models = {}
        for task_name, task in self.tasks.items():
            models_class, task_models = task.models.get_pipeline_models(
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline=manifolds_pipeline,
            )
            models[task_name] = task_models
        return models

    def plot_multiple_consistency_scores(
        self,
        animals,
        wanted_stimulus_types=["time", "behavior"],
        wanted_embeddings=["A", "B", "A'"],
        exclude_properties=None,
        figsize=(7, 7),
    ):
        # TODO: implement consistency session plot
        pass

    def filter_tasks(self, wanted_properties=None):
        """
        Filters the tasks based on the wanted properties.

        Parameters
        ----------
        wanted_properties : dict, optional
            A dictionary containing the properties to filter by. The dictionary should have the following structure:
            Example:
                wanted_properties = {
                    "task": {
                        "name": ["FS1"],
                    },
                    "neural_metadata": {
                        "area": "CA3",
                        "method": "1P",
                    },
                    "behavior_metadata": {
                        "setup": "openfield",
                    },
                }

        Returns
        -------
        filtered_tasks : dict
            dict containing the filtered tasks based on the wanted properties with the task id as key and the task object as value.
        """
        if not wanted_properties:
            print("No wanted properties given. Returning tasks sessions")
            wanted_properties = {}

        filtered_tasks = {}
        for task_name, task in self.tasks.items():
            # check if task has all wanted properties
            wanted = True
            if "task" in wanted_properties:
                wanted = wanted_object(task, wanted_properties["task"])
                for metadata_type in ["behavior_metadata", "neural_metadata"]:
                    if not wanted:
                        break
                    if metadata_type in wanted_properties:
                        wanted = wanted and wanted_object(
                            getattr(task, metadata_type),
                            wanted_properties[metadata_type],
                        )
            if wanted:
                filtered_tasks[task.id] = task
        return filtered_tasks

    # Place Cells for all sessions
    def plot_cell_activity_pos_by_time(
        self,
        cell_ids: int = None,  # overrides the top_n parameter and plots only the cell with the given ids in the given order
        task_ids: str = None,
        movement_state: str = "moving",
        sort_by: str = "zscore",  # peak, spatial_information, spatial_content, zscore or indices
        reference_task: str = None,  # task id to use for sorting
        top_n: str = 10,  # if set to "significant" will use zscore_thr to get top n cells
        n_tests: int = 1000,
        provided_zscore: np.ndarray = None,
        zscore_thr: float = 2.5,
        smooth: bool = True,
        norm: bool = True,
        window_size: int = 5,
        lines_per_y: int = 3,
        figsize_x=20,
        use_discrete_colors=False,
        cmap: str = "inferno",
        show: bool = False,
        save_pdf: bool = True,
    ):
        """
        Plots the cell activity by position and time.

        Parameters
        ----------
        cell_ids : int or list, optional
            The IDs of the cells to plot. Overrides the top_n parameter (default is None).
        task_ids : str or list, optional
            The IDs of the tasks to include in the plot (default is None).
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        sort_by : str, optional
            The criterion to sort by: "zscore", "peak", "spatial_information", "spatial_content", or "indices" (default is "zscore").
        reference_task : str, optional
            The task ID to use for sorting (default is None).
        top_n : int or str, optional
            The number of top cells to plot, or "significant" to plot all significant cells (default is 10).
        n_tests : int, optional
            The number of tests to perform for z-score calculation (default is 1000).
        provided_zscore : np.ndarray, optional
            An array of precomputed z-scores (default is None).
        zscore_thr : float, optional
            The z-score threshold for significance (default is 2.5).
        smooth : bool, optional
            Whether to smooth the data (default is True).
        norm : bool, optional
            Whether to normalize the data (default is True).
        window_size : int, optional
            The size of the smoothing window (default is 5).
        lines_per_y : int, optional
            The number of lines per y-axis unit (default is 3).
        figsize_x : int, optional
            The width of the figure in inches (default is 20).
        use_discrete_colors : bool, optional
            Whether to use discrete colors for the traces (default is False).
        cmap : str, optional
            The colormap to use for the traces (default is "inferno").
            Colormap to use for the plot. Default is 'inferno' (black to yellow) for better visibility.
            colormaps for dark backgrounds: 'gray', 'inferno', 'magma', 'plasma', 'viridis'
            colormaps for light backgrounds: 'binary', 'cividis', 'spring', 'summer', 'autumn', 'winter'
        show : bool, optional
            Whether to show the plot (default is False).
        save_pdf : bool, optional
            Whether to save the plot as a PDF (default is True).

        Returns
        -------
        list
            A list of figure objects containing the plots.

        Raises
        ------
        ValueError
            If the reference task is not found in the session.
        """

        sort_by = (
            "custom"
            if isinstance(sort_by, list) or isinstance(sort_by, np.ndarray)
            else sort_by
        )

        # get rate map and time map and other info for sorting
        if not reference_task and cell_ids is None:
            print("No reference task or cell ids given. Printing all cells.")
        elif reference_task and cell_ids is None:
            if reference_task not in self.tasks.keys():
                raise ValueError(
                    f"Reference task {reference_task} not found. in session {self.id}"
                )
            task = (
                self.tasks[reference_task]
                if reference_task
                else self.tasks[list(self.tasks.keys())[0]]
            )
            if not provided_zscore:
                print(f"Extracting rate map info for reference task {task.id}")
                (
                    reference_rate_map,
                    reference_time_map,
                    reference_zscore,
                    reference_si_rate,
                    reference_si_content,
                    reference_sorting_indices,
                    reference_labels,
                ) = task.extract_rate_map_info_for_sorting(
                    movement_state=movement_state,
                    smooth=smooth,
                    window_size=window_size,
                    n_tests=n_tests,
                    top_n=top_n,
                    provided_zscore=provided_zscore,
                    zscore_thr=zscore_thr,
                )
            cell_ids = reference_sorting_indices

        cell_ids = make_list_ifnot(cell_ids) if cell_ids is not None else None
        task_cell_dict = {}
        for task_id, task in self.tasks.items():
            if task_ids and task_id not in task_ids:
                continue

            if task_id == reference_task:
                zscore = reference_zscore
                si_rate = reference_si_rate
                labels = reference_labels

            else:
                cell_ids = cell_ids or np.arange(task.neural.photon.data.shape[1])

                # extract zscore and spatial information for labels
                print(f"Extracting rate map info for task {task_id}")
                _, _, zscore, si_rate, _, _, _ = task.extract_rate_map_info_for_sorting(
                    movement_state=movement_state,
                    smooth=smooth,
                    window_size=window_size,
                    n_tests=n_tests,
                )

                labels = [
                    f"zscore: {zscore[cell]:.2f}, SI: {si_rate[cell]:.2f}"
                    for cell in cell_ids
                ]

            cell_dict = PlaceCellDetectors.get_spike_maps_per_laps(
                cell_ids=cell_ids,
                neural_data=task.neural.photon.data,
                behavior=task.behavior,
            )

            for cell_id, label in zip(cell_ids, labels):
                additional_title = (
                    f"{task_id} Stimulus: {task.behavior_metadata['stimulus_type']}"
                )
                cell_dict[cell_id]["additional_title"] = additional_title
                cell_dict[cell_id]["label"] = label

            task_cell_dict[task_id] = cell_dict

        # plotting
        only_moving = True if movement_state == "moving" else False

        cell_task_activity_dict = {}
        for cell_id in cell_ids:
            cell_task_activity_dict[cell_id] = {}
            for task_id, cell_dict in task_cell_dict.items():
                cell_task_activity_dict[cell_id][task_id] = cell_dict[cell_id]

        figures = [None] * len(cell_task_activity_dict)
        for cell_num, (cell_id, cells_task_activity_dict) in enumerate(
            cell_task_activity_dict.items()
        ):
            additional_title = f"{self.id} sorted by {sort_by if top_n != 'significant' else 'zscore'} Cell {cell_id}"
            if only_moving:
                additional_title += " (Moving only)"

            fig = Vizualizer.plot_multi_task_cell_activity_pos_by_time(
                cells_task_activity_dict,
                figsize_x=figsize_x,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                additional_title=additional_title,
                savepath=None,
                lines_per_y=lines_per_y,
                use_discrete_colors=use_discrete_colors,
                cmap=cmap,
                show=show,
            )

            figures[cell_num] = fig

        if save_pdf:
            with PdfPages(f"{self.id}_cell_task_activity.pdf") as pdf:
                for fig in figures:
                    pdf.savefig(fig)


class Task(MetaClass):
    """Represents a task in the dataset."""

    needed_attributes: List[str] = ["neural_metadata", "behavior_metadata"]
    descriptive_metadata_keys = [
        "stimulus_type",
        "method",
        "processing_software",
    ]

    def __init__(
        self,
        session,
        task_name,
        data_dir=None,
        model_dir=None,
        metadata: dict = {},
        model_settings: dict = {},
        data_filter: Union[Dict[str, Any], DataFilter] = {},
    ):
        self.name: str = task_name
        self.session = session
        self.id: str = f"{session.id}_{task_name}"
        self.dir = Path(session.dir).joinpath(task_name)
        super().__init__(self.dir, data_filter=data_filter)
        self.data_dir: Path = data_dir or self.define_data_dir(session.dir)

        self.neural_metadata: Dict[str, Dict]
        self.behavior_metadata: Dict[str, Dict]
        self.neural_metadata, self.behavior_metadata = self.load_metadata(
            copy.deepcopy(metadata)
        )
        self.neural: Datasets_Neural = Datasets_Neural(
            root_dir=self.data_dir, metadata=self.neural_metadata, task_id=self.id
        )
        self.animal_metadata = session.animal.metadata
        self.behavior_metadata = self.fit_behavior_metadata(self.neural)
        self.behavior: Datasets_Behavior = Datasets_Behavior(
            root_dir=self.data_dir, metadata=self.behavior_metadata, task_id=self.id
        )
        self.model_dir: Path = model_dir or self.data_dir.joinpath("models")

        self.models = Models(
            self.model_dir,
            model_id=self.name,
            model_settings=model_settings,
            data_filter=self.data_filter,
        )

    @property
    def child_obj(self):
        return self.models

    @property
    def random_child(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: List[List[str]] = "12800",  # or [str] or str
        model_naming_filter_exclude: List[List[str]] = "0.",  # or [str] or str
    ):
        model_class, models = self.models.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        model = random.choice(list(models.values()))
        global_logger.info(f"Got random model: {model.name}")
        return model

    def define_data_dir(self, session_dir):
        data_dir = session_dir
        folders = search_filedir(
            data_dir,
            type="dir",
        )
        for folder in folders:
            if self.name in folder.name:
                data_dir = data_dir.joinpath(self.name)
        # if self.name in get_directories(data_dir):
        #    data_dir = data_dir.joinpath(self.name)
        return data_dir

    def load_metadata(self, metadata: dict = {}):
        set_attributes_check_presents(
            propertie_name_list=metadata.keys(),
            set_object=self,
            propertie_values=metadata.values(),
            needed_attributes=Task.needed_attributes,
        )
        self.neural_metadata["task_id"] = self.id
        self.behavior_metadata["task_id"] = self.id
        return self.neural_metadata, self.behavior_metadata

    def fit_behavior_metadata(self, neural):
        self.behavior_metadata["method"] = neural.metadata["method"]
        self.behavior_metadata["imaging_fps"] = neural.metadata["fps"]
        if "area" in neural.metadata.keys():
            self.behavior_metadata["area"] = neural.metadata["area"]
        return self.behavior_metadata

    def load_all_data(
        self,
        behavior_data_types=["position"],
        regenerate=False,
        regenerate_plots=False,
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
        plot=None,
    ):
        """
        neural_Data = ["photon"]
        behavior_data_types = ["position", "velocity", "stimulus"]
        data_filter: For behavior data types, a dictionary with keys as behavior data types and values as the filter to apply.
            example:
                data_filter = {
                "velocity": {"range" : (0.0, 0.35)}, # m/s
                }
        """
        if data_filter is not None:
            if self.data_filter is not None:
                self.data_filter.update(**data_filter)
            else:
                self.data_filter = DataFilter(data_filter)

        if "velocity" in behavior_data_types and "moving" not in behavior_data_types:
            behavior_data_types.append("moving")

        global_include_frames = None
        for data_type in ["behavior", "neural"]:
            datasets_object = getattr(self, data_type)

            if data_type == "neural":
                data_to_load = self.neural_metadata["setup"]
            else:
                data_to_load = behavior_data_types

            data_to_load = make_list_ifnot(data_to_load)
            for data_source in data_to_load:
                try:
                    source_data = datasets_object.load(
                        data_source=data_source,
                        regenerate=regenerate,
                        regenerate_plot=regenerate_plots,
                        data_filter=self.data_filter,
                        plot=False,
                    )

                except Exception as e:
                    global_logger.error(
                        f"Error while adding task {self.name} to session {self.id}: {e}"
                    )
                    return None

        # plot data if requested
        data = {"neural": {}, "behavior": {}}
        for data_type in ["neural", "behavior"]:
            datasets_object = getattr(self, data_type)
            if data_type == "neural":
                sources = self.neural_metadata["setup"]
            else:
                sources = behavior_data_types

            sources = make_list_ifnot(sources)
            for data_source in sources:
                dataset_obj = datasets_object.get_object(data_source=data_source)
                data[data_type][data_source] = dataset_obj.data
                if plot or regenerate_plots:
                    dataset_obj.plot(regenerate_plot=regenerate_plots, show=plot)

        # create initial insides using combined information

        return data

    # Cebra
    def train_model(
        self,
        model_type: str,  # types: time, behavior, hybrid
        regenerate: bool = False,
        shuffle: bool = False,
        movement_state: str = "moving",
        split_ratio: float = 1,
        name_comment: str = None,
        neural_data: np.ndarray = None,
        behavior_data: np.ndarray = None,
        transformation: Literal["binned", "relative"] = None,
        neural_data_types: Literal["photon", "probe"] = "photon",
        behavior_data_types: Literal[
            "position", "moving", "distance", "velocity", "acceleration"
        ] = None,
        manifolds_pipeline: str = "cebra",
        model_settings: dict = None,
        create_embeddings: bool = True,
        verbose: bool = False,
        plot: bool = False,
    ):
        """
        Train a model for a task using neural, behavior data and the choosen pipeline.

        A created model will inherit the train and test data as well as the embedded data.

        Parameters
        ----------
        model_type : str
            The type of model to train. The available model_types are: time, behavior, hybrid
            time: models the neural data over time
            behavior: models the neural data over behavior data
            hybrid: models the neural data over time and behavior data
        regenerate : bool, optional
            Whether to regenerate and overwrite the model (default is False).
        shuffle : bool, optional
            Whether to shuffle the data before training (default is False).
        movement_state : str, optional
            The movement state to filter by (default is "all").
            The available movement_states are: "all", "moving", "stationary"
        split_ratio : float, optional
            The ratio to data used for training the model, the rest is used for the testing sets. (default is 1).
        name_comment : str, optional
            The name of the model (default is None). The name is modified by other parameters to create a more unique name.
            Reserved words that should not be used:
                Models:
                    - CEBRA:
                        - model types:
                            - time
                            - behavior
                            - hybrid
                        - iter-<some number>
                - movement states:
                    - moving
                    - stationary
                - transformations:
                    - binned
                    - relative
                    - shuffled
                - random
        neural_data : np.ndarray, optional
            The neural data to use for training the model (default is None). If None, the neural data is loaded from the task.
            The shape of the neural data should be (n_samples, n_features).
        behavior_data : np.ndarray, optional
            The behavior data to use for training the model (default is None). If None, the behavior data is loaded from the task.
            The shape of the behavior data should be (n_samples, n_features).
        transformation : str, optional
            Whether the behavior data should be transformed.
            Transformation types: "binned", "relative" (default is None).
        behavior_data_types : list, optional
            The types of behavior data to use for training the model (default is None).
            Available types are: "position", "velocity", "stimulus", "moving", "acceleration"
        manifolds_pipeline : str, optional
            The pipeline to use for creating the embeddings (default is "cebra").
        model_settings : dict, optional
            The settings for the model (default is None). If not defined the default settings of the pipeline are used
            The dictionary should contain a key for the model type and the corresponding settings.
        create_embeddings : bool, optional
            Whether to create embeddings for the model (default is True).
        verbose : bool, optional
            show progress bar (default is False).

        Returns
        -------
        Model
            The trained model.
        """
        behavior_data_types = make_list_ifnot(behavior_data_types)
        neural_data_types = make_list_ifnot(neural_data_types)
        # get neural data
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)

        # neural_data_types = neural_data_types or self.neural_metadata["preprocessing"]
        if neural_data is None:
            neural_data, _ = self.neural.get_multi_data(
                sources=self.neural.imaging_type,
            )
        elif not isinstance(neural_data, np.ndarray):
            raise ValueError("neural_data must be a numpy array.")

        # get behavior data
        if behavior_data is None:
            behavior_data, _ = self.behavior.get_multi_data(
                sources=behavior_data_types,
                transformation=transformation,
            )
        elif not isinstance(behavior_data, np.ndarray):
            raise ValueError("behavior_data must be a numpy array.")

        if behavior_data_types is not None and behavior_data is None:
            global_logger.error(
                f"Behavior data types {behavior_data_types} not found in task {self.id}. Not able to train model."
            )
            return None

        global_logger.info(
            f"Training model {model_type} for task {self.id} using pipeline {manifolds_pipeline}"
        )

        model = self.models.train_model(
            neural_data=neural_data,
            behavior_data=behavior_data,
            behavior_data_types=behavior_data_types,
            idx_to_keep=idx_to_keep,
            model_type=model_type,
            name_comment=name_comment,
            movement_state=movement_state,
            transformation=transformation,
            shuffle=shuffle,
            split_ratio=split_ratio,
            model_settings=model_settings,
            pipeline=manifolds_pipeline,
            create_embeddings=create_embeddings,
            regenerate=regenerate,
            metadata={
                "animal": self.session.animal.metadata,
                "session": self.session.metadata,
                "neural": self.neural_metadata,
                "behavior": self.behavior_metadata,
            },
            verbose=verbose,
            plot=plot,
        )

        # add pointer to behaviour data objects to model
        behavior_data_objects = {}
        for behavior_data_type in behavior_data_types:
            if behavior_data_type in self.behavior.__dict__.keys():
                behavior_data_objects[behavior_data_type] = getattr(
                    self.behavior, behavior_data_type
                )

        # add pointer to neural data objects to model
        neural_data_objects = {}
        for neural_data_type in neural_data_types:
            if neural_data_type in self.neural.__dict__.keys():
                neural_data_objects[neural_data_type] = getattr(
                    self.neural, neural_data_type
                )
        model.neural_data_objects = neural_data_objects
        model.set_train_objects(
            neural_objects=neural_data_objects, behavior_objects=behavior_data_objects
        )
        return model

    def get_behavior_labels(
        self,
        behavior_data_types,
        transformation: str = None,
        idx_to_keep: np.ndarray = None,
        movement_state="all",
    ):
        if idx_to_keep is None and movement_state != "all":
            idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)
        labels_dict = {}
        for behavior_data_type in behavior_data_types:
            behavior_data, _ = self.behavior.get_multi_data(
                behavior_data_type,
                transformation=transformation,
                idx_to_keep=idx_to_keep,
            )
            labels_dict[behavior_data_type] = behavior_data
        return labels_dict

    def plot_embeddings(
        self,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        train_or_test: str = "train",
        to_2d: bool = False,
        show_hulls: bool = False,
        to_transform_data: Optional[np.ndarray] = None,
        given_colorbar_ticks: Optional[List] = None,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        behavior_data_types: List[str] = ["position"],
        manifolds_pipeline: str = "cebra",
        title: Optional[str] = None,
        title_comment: Optional[str] = None,
        markersize: float = None,
        alpha: float = None,
        figsize: Tuple[int, int] = None,
        dpi: int = 300,
        as_pdf: bool = False,
    ):
        """
        behavior_data_types : list, optional
            The types of behavior data to use when extracting the labels (default is ["position"]).
            The available types are: "position", "velocity", "stimulus", "moving", "acceleration".
        """
        embeddings, labels_dict = self.extract_wanted_embedding_and_labels(
            cls=self,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            embeddings=embeddings,
            train_or_test=train_or_test,
            manifolds_pipeline=manifolds_pipeline,
            to_transform_data=to_transform_data,
            labels=labels,
            to_2d=to_2d,
        )

        viz = Vizualizer(self.data_dir.parent.parent)
        title = (
            f"{manifolds_pipeline.upper()} embeddings {self.id}" if not title else title
        )
        title = add_descriptive_metadata(
            text=title,
            comment=title_comment,
            metadata=self.behavior_metadata,
            keys=Task.descriptive_metadata_keys,
        )

        projection = "2d" if to_2d else "3d"
        # plot embeddings if behavior_data_type is in embedding_title
        for behavior_data_type in behavior_data_types:
            # get ticks
            if given_colorbar_ticks is None:
                dataset_object = getattr(self.behavior, behavior_data_type)
                colorbar_ticks = dataset_object.plot_attributes["yticks"]
            else:
                colorbar_ticks = given_colorbar_ticks
            labels_dict_plot = {"name": behavior_data_type, "labels": []}
            embeddings_to_plot = {}
            for embedding_title, labels in labels_dict.items():
                if behavior_data_type not in embedding_title:
                    continue
                labels_dict_plot["labels"].append(labels)
                min_val_labels = np.min(labels)
                max_val_labels = np.max(labels)
                embeddings_to_plot[embedding_title] = embeddings[embedding_title]

            viz.plot_multiple_embeddings(
                embeddings_to_plot,
                labels=labels_dict_plot,
                min_val=min_val_labels,
                max_val=max_val_labels,
                ticks=colorbar_ticks,
                title=title,
                projection=projection,
                show_hulls=show_hulls,
                markersize=markersize,
                figsize=figsize,
                alpha=alpha,
                dpi=dpi,
                as_pdf=as_pdf,
            )
        return embeddings

    def get_spike_map_per_lap(self, cell_id):
        """
        Returns the spike map for a given cell_id. Used for parallel processing.
        """
        # get data
        cell_activity = self.neural.photon.data[:, cell_id]
        binned_pos = self.behavior.position.binned_data

        # split data by laps
        cell_activity_by_lap = self.behavior.split_by_laps(cell_activity)
        binned_pos_by_lap = self.behavior.split_by_laps(binned_pos)

        # count spikes at position
        max_bin = self.behavior.position.max_bin
        cell_lap_activity = np.zeros((len(cell_activity_by_lap), max_bin))
        for i, (lap_act, lap_pos) in enumerate(
            zip(cell_activity_by_lap, binned_pos_by_lap)
        ):
            counts_at = PlaceCellDetectors.get_spike_map(lap_act, lap_pos, max_bin)
            cell_lap_activity[i] = counts_at

        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']} - Cell {cell_id}"

        return cell_lap_activity, additional_title

    def plot_model_losses(
        self,
        models=None,
        title=None,
        manifolds_pipeline="cebra",
        coloring_type="rainbow",
        plot_original=True,
        plot_shuffled=True,
        num_iterations=None,
        plot_iterations=False,
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        alpha=0.8,
        figsize=(10, 10),
        as_pdf=False,
    ):
        models_original, models_shuffled = (
            self.models.get_models_splitted_original_shuffled(
                models=models,
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )
        )

        stimulus_type = (
            self.behavior_metadata["stimulus_type"]
            if "stimulus_type" in self.behavior_metadata.keys()
            else ""
        )
        num_iterations = (
            models_original[0].max_iterations if not num_iterations else num_iterations
        )
        title = title or f"Losses {self.id} {stimulus_type}" if not title else title
        comment = f" - {num_iterations} Iterartions" if not plot_iterations else ""
        title = add_descriptive_metadata(
            text=title,
            comment=comment,
            metadata=self.behavior_metadata,
            keys=Task.descriptive_metadata_keys,
        )

        viz = Vizualizer(self.data_dir.parent.parent)
        losses_original = {}
        for model in models_original:
            if model.fitted:
                losses_original[model.name] = model.state_dict_["loss"]
            else:
                global_logger.warning(
                    f"{model.name} Not fitted. Skipping model {model.name}."
                )
                print(f"Skipping model {model.name}.")

        losses_shuffled = {}
        for model in models_shuffled:
            if model.fitted:
                losses_shuffled[model.name] = model.state_dict_["loss"]
            else:
                global_logger.warning(
                    f"{model.name} Not fitted. Skipping model {model.name}."
                )
                print(f"Skipping model {model.name}.")

        viz.plot_losses(
            losses=losses_original,
            losses_shuffled=losses_shuffled,
            title=title,
            coloring_type=coloring_type,
            plot_original=plot_original,
            plot_shuffled=plot_shuffled,
            alpha=alpha,
            figsize=figsize,
            plot_iterations=plot_iterations,
            as_pdf=as_pdf,
        )

    @staticmethod
    def extract_wanted_embedding_and_labels(
        cls: Union[Task, Multi],
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        manifolds_pipeline: str = "cebra",
        train_or_test: str = "train",
        to_transform_data: Optional[np.ndarray] = None,
        use_raw: bool = False,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_2d: bool = False,
    ):
        """
        Extracts the wanted data for plotting embeddings.

        Parameters
        ----------
        model_naming_filter_include: List[List[str]] = None,  # or [str] or str
            Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
            1. Include all models containing a specific string: "string"
            2. Include all models containing a specific combination of strings: ["string1", "string2"]
            3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
        model_naming_filter_exclude: List[List[str]] = None,  # or [str] or str
            Same as model_naming_filter_include but for excluding models.
        use_raw : bool, optional
            Whether to use raw data neural data instead of embeddings (default is False).
        embeddings : dict, optional
            A dictionary containing the embeddings to plot (default is None). If None, the embeddings are created from the models,
            which should inherit the default embedding data (data that was defined to train the model) based on neural data.
        manifolds_pipeline : str, optional
            The pipeline to search for models (default is "cebra").
        to_transform_data : np.ndarray, optional
            The data to transform to embeddings (default is None). If None, the embeddings are created from the models inheriting the default data (data that was defined to train the model)
        labels : np.ndarray or dict, optional
            The labels for the embeddings (default is None). If None, the labels are extracted from the default behavior data (data that was defined to train the model).
        """
        if embeddings is not None and to_transform_data is not None:
            global_logger.critical(
                "Either provide embeddings or to_transform_data, not both."
            )
            raise ValueError(
                "Either provide embeddings or to_transform_data, not both."
            )
        if to_transform_data is not None:
            embeddings = cls.models.create_embeddings(
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
                manifolds_pipeline="cebra",
            )

        if not isinstance(embeddings, dict) or labels is None:
            models_class, models = cls.models.get_pipeline_models(
                manifolds_pipeline=manifolds_pipeline,
                model_naming_filter_include=model_naming_filter_include,
                model_naming_filter_exclude=model_naming_filter_exclude,
            )

            if not isinstance(embeddings, dict):
                embeddings = {}
                for model_name, model in models.items():
                    embeddings[model_name] = (
                        model.get_data()
                        if not use_raw
                        else model.get_data(
                            train_or_test=train_or_test, type="binarized"
                        )
                    )

            if labels is None:
                labels_dict = {}
                for model_name, model in models.items():
                    labels_dict[model_name] = model.get_data(
                        train_or_test=train_or_test, type="behavior"
                    )

        # get embedding lables
        if labels is not None:
            if not isinstance(labels, np.ndarray) and not isinstance(labels, dict):
                raise ValueError(f"Provided labels is not a numpy array or dictionary.")
            else:
                if isinstance(labels, np.ndarray):
                    labels_dict = {"Provided_labels": labels}
                else:
                    labels_dict = labels
                    if not equal_number_entries(embeddings, labels_dict):
                        global_logger.warning(
                            f"Number of labels is not equal to all, moving or stationary number of frames. You could extract labels corresponding to only moving using the function self.get_behavior_labels(behavior_data_types, movement_state='moving'\)"
                        )

        return embeddings, labels_dict

    ############################################################## Place Cells
    def get_cells(
        self,
        # movement_state: Literal["moving", "stationary"]="moving",
        cell_type: Literal["all", "place", "non-place"] = "place",
        plot: bool = False,
    ):
        """
        Get the place cells for the task using self.models.place_cell.si_model.

        Opexebo_cell_analysis columns:
            'cell_id',
            'activity_rate_hz':
                is less than 0.01, the cell got skipped in the place cell analysis
            'skipped_due_to_low_rate',
            'overlaps',
            'spatial_info',
            'spatial_info_zscores':
                first entry gives the number of standard deviations the spatial information level of the cell is
                from it's own shuffle mean. If it's > 2.326 then it's a place cell (in the 99th percentile).
            'coherence',
            'fields_map_split',
            'occ_maps_split',
            'res_array',
            'rms_split',
            'rms_all',
            'fields_map_all'
                this indicates the location of a cells place fields in the binned environment
                (bins with 0 = background, bins with 1 = place field 1, bins with 2 = place field 2, etc
                Only cells with at least 1 field are actually considered to be a place cell.
        """
        available_cell_types = ["all", "place", "non-place"]
        cell_analysis_dir = self.dir.joinpath("Opexebo_cell_analysis")

        if cell_analysis_dir.exists():
            cell_list = list(Path(cell_analysis_dir).rglob("*.npy"))
            cell_df = pd.DataFrame(
                [np.load(f, allow_pickle=True).item() for f in cell_list]
            )
            cell_df.set_index("cell_id", inplace=True)

            if cell_type == "all":
                filtered_df = cell_df
            elif cell_type in ["place", "non-place"]:

                # Extract first element if list/ndarray, else NaN
                si_zscore_mask = (
                    cell_df["spatial_info_zscores"]
                    .apply(
                        lambda v: v[0] if isinstance(v, (list, np.ndarray)) else np.nan
                    )
                    .to_numpy()
                    >= 2.326
                )

                max_fields_mask = (
                    cell_df["fields_map_all"]
                    .apply(
                        lambda x: (
                            np.max(x)
                            if isinstance(x, (list, np.ndarray)) and x is not None
                            else np.nan
                        )
                    )
                    .to_numpy()
                    > 0
                )

                rate_si_zscore_mask = si_zscore_mask & max_fields_mask

                if cell_type == "non-place":
                    rate_si_zscore_mask = ~rate_si_zscore_mask

                # create filter mask
                rate_mask = cell_df["skipped_due_to_low_rate"] == False

                combined_mask = rate_si_zscore_mask & max_fields_mask
                filtered_df = cell_df[combined_mask]
            else:
                do_critical(
                    NotImplementedError,
                    f"cell_type {cell_type} not implemented. Choose from {available_cell_types}.",
                )

        else:
            filtered_df = self.models.place_cell.si_model.get_place_cells(
                # neural_data=self.neural.photon.data,
                # behavior=self.behavior,
                # movement_state=movement_state,
                # task_id=self.id,
                # save_dir=self.dir.joinpath(cell_analysis_folder_name),
            )

        if plot:
            plot_all_cells_modular(
                filtered_df, col_name="fields_map_all", max_cells_per_plot=10
            )
        return filtered_df

    def get_rate_time_map(
        self,
        movement_state="moving",
        smooth=True,
        window_size=2,
        norm_plot_rate=True,
        sorting_indices=None,
        plot: bool = False,
        plot_cells: Union[int, List[int]] = None,
    ):
        """
        Gets the rate and time maps for the place cells.

        Parameters
        ----------
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        smooth : bool, optional
            Whether to smooth the data (default is True).
        window_size : int, optional
            The size of the smoothing window in frames (default is 2).

        Returns
        -------
        tuple
            rate_map : np.ndarray
                The rate map of the place cells.
            time_map : np.ndarray
                The time map of the place cells.
            activity : np.ndarray
                The filtered neural activity data.
            binned_pos : np.ndarray
                The filtered binned position data.
        """
        idx_to_keep = self.behavior.moving.get_idx_to_keep(movement_state)
        f, idx_to_keep = force_equal_dimensions(self.neural.photon.data, idx_to_keep)
        activity = Dataset.filter_by_idx(f, idx_to_keep=idx_to_keep)

        p, idx_to_keep = force_equal_dimensions(
            self.behavior.position.binned_data, idx_to_keep
        )
        binned_pos = Dataset.filter_by_idx(p, idx_to_keep=idx_to_keep)

        max_bin = self.behavior.position.max_bin

        category_map = self.behavior.position.category_map

        rate_map, time_map = self.models.place_cell.get_rate_map(
            activity=activity,
            binned_pos=binned_pos,
            smooth=smooth,
            window_size=window_size,
            max_bin=max_bin,
            category_map=category_map,
            fps=self.behavior_metadata["imaging_fps"],
        )

        only_moving = True if movement_state == "moving" else False
        additional_title = f"{self.id} {self.behavior_metadata['stimulus_type']}"
        if only_moving:
            additional_title += " (moving)"

        # flatten tuning map
        tuning_map = rate_map.reshape(rate_map.shape[0], -1)
        cmap = create_2d_colormap(x_bins=max_bin[0], y_bins=max_bin[1])
        if plot:
            if plot_cells is not None:
                if isinstance(plot_cells, int):
                    plot_cells = list(plot_cells)
                elif not isinstance(plot_cells, list):
                    raise ValueError("plot_cells must be an int or a list of ints.")
                filtered_rates = rate_map[plot_cells]
                filtered_tuning = tuning_map[plot_cells]
                additional_title += f" Cells: {plot_cells}"
            else:
                filtered_rates = rate_map
                filtered_tuning = tuning_map

            Vizualizer.plot_cell_activites_heatmap(
                filtered_rates,
                additional_title="(Rate) " + additional_title,
                cbar_title="Spike Rate",
                norm_rate=norm_plot_rate,
                sorting_indices=sorting_indices,
            )

            # plot tuning curves of random cells
            colors = []
            plot_labels = []
            for x, y in category_map.keys():
                colors.append(cmap[x, y])
                plot_labels.append(f"{x}, {y}")

            # Generate xticks
            xticks = []
            for i in range(256):  # Adjusted to 256 bins, assuming 16x16 grid
                x = i // 16
                y = i % 16
                xticks.append(f"({x}, {y})")

            # Only show every 16th xtick
            plot_xticks = []
            xticks_pos = []
            for i in range(len(xticks)):
                if i % 16 == 0:
                    plot_xticks.append(xticks[i])
                    xticks_pos.append(i)

            # Create figure with subplots
            fig, ax = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
            ax = ax.ravel()

            # Select random cell IDs
            cell_ids = np.random.choice(len(filtered_tuning), size=4, replace=False)

            # Plot tuning curves
            for i, cell_id in enumerate(cell_ids):
                ax[i].plot(filtered_tuning[cell_id])
                ax[i].set_title(f"Tuning curve of cell {cell_id}")
                ax[i].set_xlabel("Position bin")
                ax[i].set_ylabel("Spike Rate")
                ax[i].set_xticks(xticks_pos)
                ax[i].set_xticklabels(
                    plot_xticks, rotation=45
                )  # Added rotation for better visibility

            simple_embedd(
                filtered_tuning.T,
                additional_title=f"Rate Map {additional_title}",
                labels=plot_labels,
                plot_show=["samples"],
                n_components=3,
                colors=colors,
                legend=False,
                method="pca",
                save_dir=self.figures_dir,
                add_cmap=cmap,
                # as_pdf=as_pdf,
            )

        return rate_map, time_map, activity, binned_pos, tuning_map

    def extract_rate_map_info_for_sorting(
        self,
        sort_by="zscore",
        movement_state="moving",
        smooth=True,
        window_size=2,
        n_tests=1000,
        top_n=10,
        provided_zscore=None,
        zscore_thr=2.5,
    ):
        """
        Extracts rate map information for sorting.

        Parameters
        ----------
        sort_by : str, optional
            The criterion to sort by: "zscore", "peak", "spatial_information", or "spatial_content" (default is "zscore").
        movement_state : str, optional
            The movement state to filter by (default is "moving").
        smooth : bool, optional
            Whether to smooth the data (default is True).
        window_size : int, optional
            The size of the smoothing window (default is 2).
        n_tests : int, optional
            The number of tests to perform for z-score calculation (default is 1000).
        top_n : int or str, optional
            The number of top cells to return, or "significant" to return all significant cells (default is 10).
        provided_zscore : np.ndarray, optional
            An array of precomputed z-scores (default is None).
        zscore_thr : float, optional
            The z-score threshold for significance (default is 2.5).

        Returns
        -------
        tuple
            rate_map : np.ndarray
                The rate map of the place cells.
            time_map : np.ndarray
                The time map of the place cells.
            zscore : np.ndarray
                The z-scores of the place cells.
            si_rate : np.ndarray
                The spatial information rate of the place cells.
            si_content : np.ndarray
                The spatial information content of the place cells.
            real_sorting_indices : np.ndarray
                The indices of the sorted cells.
            labels : np.ndarray
                The labels for the sorted cells.

        Raises
        ------
        ValueError
            If the sorting criterion is not supported.
        """
        # get rate map and time map
        rate_map, time_map, activity, binned_pos = self.get_rate_time_map(
            movement_state=movement_state,
            smooth=smooth,
            window_size=window_size,
        )

        # get zscore, si_rate and si_content
        max_bin = self.behavior.position.max_bin
        fps = self.behavior_metadata["imaging_fps"]

        if sort_by == "peak":
            zscore = None
        elif sort_by in ["spatial_information", "spatial_content", "zscore"]:
            if sort_by in ["spatial_information", "spatial_content"]:
                n_test = (
                    n_tests if top_n == "significant" and provided_zscore is None else 1
                )
            elif sort_by == "zscore":
                n_test = n_tests if provided_zscore is None else 1

            zscore, si_rate, si_content = (
                self.models.place_cell.si_model.compute_si_zscores(
                    # rate_map, time_map, n_tests=n_test, fps=fps
                    activity=activity,
                    binned_pos=binned_pos,
                    n_tests=n_test,
                    fps=fps,
                    max_bin=max_bin,
                )
            )
            zscore = zscore if provided_zscore is None else provided_zscore
        else:
            raise ValueError(f"Sorting of the rate_map by {sort_by} not supported.")

        # define labels
        if sort_by == "peak":
            sorting_indices = np.argsort(np.argmax(rate_map, axis=1))
            labels = sorting_indices
        elif sort_by == "spatial_information":
            labels = si_rate
        elif sort_by == "spatial_content":
            labels = si_content
        elif sort_by == "zscore":
            labels = zscore

        sorting_indices = np.argsort(labels)[::-1]
        num_nan = np.sum(np.isnan(labels)) if top_n != "significant" else 0

        # get top n rate maps
        if top_n == "all":
            top_n = len(sorting_indices) - num_nan
        elif top_n == "significant" and sort_by != "peak":
            # sorted_zscores = zscore[np.argsort(zscore)[::-1]]
            significant_zscore_indices = np.where(zscore >= zscore_thr)[0]
            significant_zscores = zscore[significant_zscore_indices]
            zscore_sorting_indices = np.argsort(significant_zscores)[::-1]
            sorted_significant_zscore_indices = significant_zscore_indices[
                zscore_sorting_indices
            ]
            sorting_indices = sorted_significant_zscore_indices
            num_significant = sorting_indices.shape[0]
            labels = np.array(
                [
                    f"si: {spatial_info:.1f}  sc: {spatial_cont:.1f}  (zscore: {z_val:.1f})"
                    for spatial_info, spatial_cont, z_val in zip(
                        si_rate, si_content, zscore
                    )
                ]
            )
            top_n = num_significant
        elif isinstance(top_n, int):
            top_n = min(top_n, len(sorting_indices) - num_nan)
        else:
            raise ValueError("top_n must be an integer, 'all' or 'significant'.")

        real_sorting_indices = sorting_indices[num_nan : num_nan + top_n]

        return (
            rate_map,
            time_map,
            zscore,
            si_rate,
            si_content,
            real_sorting_indices,
            labels,
        )

    def plot_rate_map_per_cell(
        self,
        provided_zscore=None,
        movement_state="moving",
        norm=True,
        smooth=True,
        window_size=2,
        n_tests=1000,
        sort_by="zscore",  # peak, spatial_information, spatial_content, zscore or indices
        top_n=10,
        zscore_thr=2.5,
        use_discrete_colors=True,
        cmap="inferno",
    ):
        """
        parameter:
        - norm: bool
            If True, normalizes the rate map.
        - smooth: bool
            If True, smooths the rate map if no rate_map is given
        - sort_by: str
            "peak", "spatial_information" or "zscore"
        - top_n: int or str
            If int, plots the top n rate maps.
            If "all", plots all rate maps.
            if "significant", plots all significant rate maps.
        """
        only_moving = True if movement_state == "moving" else False
        sort_by = (
            "custom"
            if isinstance(sort_by, list) or isinstance(sort_by, np.ndarray)
            else sort_by
        )
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']} sorted by {sort_by if top_n != 'significant' else 'zscore'}"
        if only_moving:
            additional_title += " (Moving only)"

        rate_map, time_map, zscore, si_rate, si_content, sorting_indices, labels = (
            self.extract_rate_map_info_for_sorting(
                sort_by=sort_by,
                movement_state=movement_state,
                smooth=smooth,
                window_size=window_size,
                n_tests=n_tests,
                top_n=top_n,
                provided_zscore=provided_zscore,
                zscore_thr=zscore_thr,
            )
        )

        to_plot_rate_map = rate_map[sorting_indices]
        to_plot_labels = labels[sorting_indices]

        if sort_by != "zscore":
            to_plot_labels_list = []
            for cell_id, label in zip(sorting_indices, to_plot_labels):
                label = label if isinstance(label, str) else f"{label:.1f}"
                to_plot_labels_list.append(f"Cell {cell_id:>4}: {label}")

        Vizualizer.plot_traces_shifted(
            to_plot_rate_map,
            labels=to_plot_labels,
            additional_title=additional_title,
            use_discrete_colors=use_discrete_colors,
            norm=norm,
            cmap=cmap,
        )
        outputs = {
            "rate_map": rate_map,
            "time_map": time_map,
            "zscore": zscore,
            "sorting_indices": sorting_indices,
            "si_rate": si_rate,
        }
        return outputs

    def plot_cell_activity_pos_by_time(
        self,
        cell_ids,
        labels=None,
        norm=True,
        smooth=True,
        window_size=5,
        lines_per_y=3,
        cmap="inferno",
        show=False,
        save_pdf=False,
    ):
        """
        Plots the cell activity by position and time.
        cmap: str
            Colormap to use for the plot. Default is 'inferno' (black to yellow) for better visibility.
        """
        cell_ids = make_list_ifnot(cell_ids)
        labels = labels or [None] * len(cell_ids)
        if len(labels) != len(cell_ids):
            raise ValueError("Labels must be the same length as cell_ids.")

        if "all" in cell_ids:
            cell_ids = np.arange(self.neural.photon.data.shape[1])

        cell_dict = PlaceCellDetectors.get_spike_maps_per_laps(
            cell_ids=cell_ids,
            neural_data=self.neural.photon.data,
            behavior=self.behavior,
        )  # cell_dict[cell_id]["lap_activity"] = cell_lap_activity

        for cell_id in cell_ids:
            additional_title = f"{self.id} Stimulus: {self.behavior_metadata['stimulus_type']} - Cell {cell_id}"
            cell_dict[cell_id]["additional_title"] = additional_title

        figures = [None] * len(cell_dict)
        for i, (cell_id, cell_data) in enumerate(cell_dict.items()):
            label = labels[i]
            fig = Vizualizer.plot_single_cell_activity(
                cell_data["lap_activity"],
                additional_title=cell_data["additional_title"],
                labels=label,
                norm=norm,
                smooth=smooth,
                window_size=window_size,
                lines_per_y=lines_per_y,
                cmap=cmap,
                show=show,
            )
            figures[i] = fig

        if save_pdf:
            with PdfPages(f"{self.id}_cells_activity.pdf") as pdf:
                for fig in figures:
                    pdf.savefig(fig)

        return

    def plot_place_cell_si_scores(
        self,
        movement_state="moving",
        smooth=True,
        window_size=2,
        n_tests=1000,
        colors=["red", "tab:blue"],
        method="skaggs",
    ):
        """
        Plots the spatial information scores of the place cells.
        """
        rate_map, time_map, activity, binned_pos = self.get_rate_time_map(
            movement_state=movement_state,
            smooth=smooth,
            window_size=window_size,
        )

        fps = self.behavior_metadata["imaging_fps"]
        max_bin = self.behavior.position.max_bin

        zscore, si_rate, si_content = (
            self.models.place_cell.si_model.compute_si_zscores(
                activity=activity,
                binned_pos=binned_pos,
                # rate_map=rate_map,
                # time_map=time_map,
                n_tests=n_tests,
                spatial_information_method=method,
                fps=fps,
                max_bin=max_bin,
            )
        )
        additional_title = f"{self.id} Belt: {self.behavior_metadata['stimulus_type']}"
        additional_title += f" ({method})"

        Vizualizer.plot_zscore(
            zscore,
            additional_title=additional_title,
            color=colors,
        )
        Vizualizer.plot_si_rates(
            si_rate,
            zscores=zscore,
            additional_title=additional_title,
            color=colors,
        )
        return zscore, si_rate, si_content
