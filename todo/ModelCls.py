from __future__ import annotations

# type hints
from typing import List, Union, Dict, Tuple, Optional, Literal

# show progress bar
from tqdm import tqdm, trange

# calculations
import numpy as np
import scipy

# plotting
import matplotlib.pyplot as plt

# parralelize
from numba import jit, njit, prange
from numba import cuda  # @jit(target='cuda')

# manifolds
from cebra import CEBRA
import cebra.models
import cebra.solver
import cebra.integrations.sklearn.utils as sklearn_utils
import torch
from torch import nn
import itertools

# own
from Datasets import Datasets, Dataset, BehaviorDataset, NeuralDataset
from Helper import *
from Visualizer import Vizualizer, plot_simple_embedd
from Manimeasure import decode, structure_index, feature_similarity
from Setups import Environment


class Models:
    pipelines = ["cebra"]  # , "place_cell"]

    def __init__(
        self,
        model_dir: Union[str, Path],
        model_id: str,
        model_settings: Dict[str, Any] = None,
        data_filter: DataFilter = None,
    ):
        place_cell_settings = (
            model_settings["place_cell"] if "place_cell" in model_settings else None
        )
        cebra_cell_settings = (
            model_settings["cebra"] if "cebra" in model_settings else None
        )
        self.data_filter: DataFilter = data_filter
        self.place_cell = PlaceCellDetectors(model_dir, model_id, place_cell_settings)
        self.cebras = Cebras(model_dir, model_id, cebra_cell_settings, data_filter)

    def define_model_name(
        self,
        model_type: str,
        name_comment: str = None,
        behavior_data_types: List[str] = None,
        shuffled: bool = False,
        transformation: str = None,
        movement_state: str = "all",
        split_ratio: float = 1,
        model_settings: dict = None,
    ):
        """
        Define the name of the model based on the parameters provided.

        It is important to mention that the model name is also influenced by self.data_filter, which is not passed as a parameter here.

        """
        behavior_data_types = [] if behavior_data_types is None else behavior_data_types
        if name_comment:
            model_name = name_comment
            for behavior_data_type in behavior_data_types:
                if behavior_data_type not in model_name:
                    model_name = f"{model_name}_{behavior_data_type}"

            if model_type not in model_name:
                model_name = f"{model_type}_{model_name}"
            if shuffled and "shuffled" not in model_name:
                model_name = f"{model_name}_shuffled"
        else:
            model_name = model_type
            for behavior_data_type in behavior_data_types:
                model_name = f"{model_name}_{behavior_data_type}"
            model_name = f"{model_name}_shuffled" if shuffled else model_name

        if movement_state not in ["all", "moving", "stationary"]:
            global_logger.critical(
                f"Movement state {movement_state} not supported. Choose 'all', 'moving', or 'stationary'."
            )
            raise ValueError(
                f"Movement state {movement_state} not supported. Choose 'all', 'moving', or 'stationary'."
            )
        elif movement_state != "all":
            model_name = f"{model_name}_{movement_state}"

        if split_ratio != 1:
            model_name = f"{model_name}_{split_ratio}"

        if transformation:
            if transformation not in ["relative", "binned"]:
                global_logger.critical(
                    f"Transformation {transformation} not supported. Choose 'relative' or 'binned'."
                )
                raise ValueError(
                    f"Transformation {transformation} not supported. Choose 'relative' or 'binned'."
                )
            else:
                if transformation not in model_name:
                    model_name = f"{model_name}_{transformation}"

        if model_settings is not None:
            max_iterations = model_settings["max_iterations"]
            model_name = f"{model_name}_iter-{max_iterations}"

        dfilter_txt = self.data_filter.filter_description(short=True)
        model_name += f"_{dfilter_txt}" if len(dfilter_txt) > 0 else ""
        return model_name

    def define_properties(self, **kwargs):
        """
        Define properties for the model.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_model(
        self,
        name_comment: str = None,
        model_type: str = "time",
        behavior_data_types: List[str] = None,
        shuffle: bool = False,
        movement_state: str = "all",
        split_ratio: float = 1,
        transformation: str = None,
        pipeline="cebra",
        model_settings=None,
        metadata: Dict[str, Dict[str, Any]] = None,
    ):
        models_class = self.get_model_class(pipeline)

        model_name = self.define_model_name(
            model_type=model_type,
            name_comment=name_comment,
            behavior_data_types=behavior_data_types,
            shuffled=shuffle,
            movement_state=movement_state,
            split_ratio=split_ratio,
            transformation=transformation,
            model_settings=model_settings[pipeline],
        )

        model_settings = model_settings[pipeline]
        # check if model with given model_settings is available
        model_available = False
        if model_name in models_class.models.keys():
            model_available = True
            model = models_class.models[model_name]
            model_parameter = model.get_params()
            if model_settings:
                for (
                    model_setting_name,
                    model_setting_value,
                ) in model_settings.items():
                    if model_parameter[model_setting_name] != model_setting_value:
                        model_available = False
                        global_logger.warning(
                            f"Model {model_name} with different model settings found. Creating new model."
                        )
                        break

        if not model_available:
            model_creation_function = getattr(models_class, model_type)
            model = model_creation_function(
                name=model_name,
                behavior_data_types=behavior_data_types,
                model_settings=model_settings,
            )

        model.define_metadata(
            data_transformation=transformation,
            data_split_ratio=split_ratio,
            data_shuffled=shuffle,
            data_movement_state=movement_state,
            behavior_data_types=behavior_data_types,
            metadata=metadata,
        )

        return model

    def get_pipeline_models(
        self,
        manifolds_pipeline="cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ) -> Tuple[Models, Dict[str, Model]]:
        """Get models from a specific model class.

        Parameters:
        -----------
            manifolds_pipeline : str
                The pipeline to use.
            model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
                Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
                1. Include all models containing a specific string: "string"
                2. Include all models containing a specific combination of strings: ["string1", "string2"]
                3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
            model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
                Same as model_naming_filter_include but for excluding models.

        Returns:
        --------
            model_class : class
                The class of the model.
            models : dict
                A dictionary containing the models.
        """
        models_class = self.get_model_class(manifolds_pipeline)
        global_logger.info(
            f"Getting models from {models_class.__class__.__name__} class including {model_naming_filter_include} and excluding {model_naming_filter_exclude}."
        )
        models = models_class.get_models(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        return models_class, models

    def get_model_information(
        self,
        wanted_information: List[
            Literal["binarized" "embedding" "loss" "label" "fluorescence", "all"]
        ] = ["embedding", "loss"],
        train_or_test: str = "train",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        manifolds_pipeline="cebra",
        behavior_data_type: str = "",
    ):
        """Get information from models.

        Parameters:
        -----------
        - include_properties: Union[List[List[str]], List[str], str] = None,
            Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
            1. Include all models containing a specific string: "string"
            2. Include all models containing a specific combination of strings: ["string1", "string2"]
            3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
        - exclude_properties: Union[List[List[str]], List[str], str] = None,
            Same as model_naming_filter_include but for excluding models.
        """
        # extract embeddings and losses
        label = {"name": behavior_data_type, "labels": []}

        if wanted_information == "all":
            wanted_information = [
                "model",
                "binarized",
                "embedding",
                "loss",
                "label",
                "fluorescence",
            ]
        wanted_information = make_list_ifnot(wanted_information)
        for wanted_info in wanted_information:
            if wanted_info not in [
                "model",
                "binarized",
                "embedding",
                "loss",
                "label",
                "fluorescence",
                "all",
            ]:
                raise ValueError(
                    f"Unknown wanted information: {wanted_info}. Only 'binarized', 'embedding', 'loss', 'label', 'fluorescence' and 'all' are allowed."
                )

        models_class, models = self.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )

        informations = {}
        data_types = ["fluorescence", "binarized", "embedding", "labels"]

        for model_name, model in models.items():
            model_info = {}

            for data_type in data_types:
                if data_type in wanted_information:
                    data = model.get_data(train_or_test=train_or_test, type=data_type)
                    model_info[data_type] = data
                    if data_type == "labels":
                        label["labels"].append(data)

            if "model" in wanted_information:
                model_info["model"] = model

            if "loss" in wanted_information:
                model_info["loss"] = model.get_loss()

            informations[model_name] = model_info

        return informations

    def get_models_splitted_original_shuffled(
        self,
        models=None,
        manifolds_pipeline="cebra",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ):
        models_original = []
        models_shuffled = []
        if not models:
            models_class, models = self.get_pipeline_models(
                manifolds_pipeline,
                model_naming_filter_include,
                model_naming_filter_exclude,
            )

        # convert list of model objects to dictionary[model_name] = model
        if isinstance(models, list):
            models_dict = {}
            for model in models:
                models_dict[model.name] = model
            models = models_dict

        for model_name, model in models.items():
            if "shuffled" in model.name:
                models_shuffled.append(model)
            else:
                models_original.append(model)
        return models_original, models_shuffled

    def create_embeddings(
        self,
        models=None,
        to_transform_data=None,
        to_2d=False,
        model_naming_filter_include: List[List[str]] = None,
        model_naming_filter_exclude: List[List[str]] = None,
        manifolds_pipeline="cebra",
        save=False,
        return_labels=False,
    ):
        """
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        """
        if not type(to_transform_data) == np.ndarray:
            global_logger.warning(
                f"No data to transform given. Using model training data."
            )
            print(f"No data to transform given. Using model training data.")

        model_class, models = self.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )

        embeddings = model_class.create_embeddings(
            models=models,
            to_transform_data=to_transform_data,
            to_2d=to_2d,
            save=save,
        )
        if return_labels:
            labels = []
            for model_name, model in models.items():
                labels.append(model_name)
            return embeddings, labels

        return embeddings

    def define_decoding_statistics(
        self,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        manifolds_pipeline: str = "cebra",
    ):
        models_class, models = self.get_pipeline_models(
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            manifolds_pipeline=manifolds_pipeline,
        )
        to_delete_models_key_list = []
        for keys_list, model in traverse_dicts(models):
            decoding_statistics = model.define_decoding_statistics()
            if decoding_statistics is None:
                to_delete_models_key_list.append(keys_list)

            # model = dict_value_keylist(models, keys_list)

        for keys_list in to_delete_models_key_list:
            delete_nested_key(models, keys_list)
        return models

    def is_model_fitted(self, model, pipeline="cebra"):
        return self.get_model_class(pipeline).is_fitted(model)

    def train_model(
        self,
        neural_data,
        model=None,
        behavior_data=None,
        behavior_data_types=None,
        pipeline="cebra",
        model_type="time",
        name_comment=None,
        transformation=None,
        idx_to_keep=None,
        shuffle=False,
        movement_state="moving",
        split_ratio=1,
        model_settings=None,
        create_embeddings=True,
        regenerate=False,
        metadata: Dict[str, Dict[str, Any]] = None,
        verbose: bool = True,
        plot: bool = False,
    ):
        if not is_dict_of_dicts(model_settings):
            model_settings = {pipeline: model_settings}

        model = model or self.get_model(
            pipeline=pipeline,
            model_type=model_type,
            name_comment=name_comment,
            behavior_data_types=behavior_data_types,
            shuffle=shuffle,
            movement_state=movement_state,
            split_ratio=split_ratio,
            transformation=transformation,
            model_settings=model_settings,
            metadata=metadata,
        )

        neural_data_train, neural_data_test, idx_train, idx_test = (
            Dataset.manipulate_data(
                neural_data,
                idx_to_keep=idx_to_keep,
                shuffle=shuffle,
                split_ratio=split_ratio,
                return_ids=True,
            )
        )

        if behavior_data is not None:
            behavior_data_train, behavior_data_test = Dataset.manipulate_data(
                behavior_data,
                shuffle=shuffle,
                idx_train=idx_train,
                idx_test=idx_test,
                return_ids=False,
            )

        model = model.train(
            neural_data=neural_data_train,
            behavior_data=behavior_data_train,
            regenerate=regenerate,
            verbose=verbose,
        )

        model.set_train_info(
            idx_train=idx_train,
            idx_test=idx_test,
            split_ratio=split_ratio,
            movement_state=movement_state,
            space="binarized",
            transformation=transformation,
            shuffle=shuffle,
        )

        if create_embeddings:
            train_embedding = model.create_embedding(
                to_transform_data=neural_data_train,
                transform_data_labels=behavior_data_train,
                plot=plot,
            )
            test_embedding = model.create_embedding(
                to_transform_data=neural_data_test,
                transform_data_labels=behavior_data_test,
                plot=plot,
            )

            model.set_embedding(data=train_embedding, train_or_test="train")
            model.set_embedding(data=test_embedding, train_or_test="test")
        return model

    def get_model_class(self, pipeline="cebra"):
        if pipeline not in Models.pipelines:
            raise ValueError(f"Pipeline {pipeline} not supported. Choose 'cebra'.")
        else:
            if pipeline == "cebra":
                models_class = self.cebras
            elif pipeline == "place_cell":
                models_class = self.place_cell
        return models_class

    @staticmethod
    def xticks_from_model_dict(unique_models):
        xticks = []
        for task_name, ref_task_model in unique_models.items():
            # stimulus_type = self.tasks[task_name].behavior_metadata["stimulus_type"]
            task_name_type = ref_task_model.get_metadata("behavior", "task_id").split(
                "_"
            )[-1]
            stimulus_type = ref_task_model.get_metadata("behavior", "stimulus_type")
            task_plot_name1 = f"{task_name_type}"  # _{stimulus_type}"
            task_plot_name1 = "random" if task_name == "random" else task_plot_name1
            xticks.append(task_plot_name1)

            for task_model_name2, model2 in unique_models.items():
                task_name_type2 = model2.get_metadata("behavior", "task_id").split("_")[
                    -1
                ]
                stimulus_type2 = model2.get_metadata("behavior", "stimulus_type")
                task_plot_name2 = f"{task_name_type2}"  # _{stimulus_type2}"
                task_plot_name2 = (
                    "random" if task_model_name2 == "random" else task_plot_name2
                )
                stimulus_decoding_name = f"{task_plot_name1} to {task_plot_name2}"
                xticks.append(stimulus_decoding_name)

        return xticks

    @staticmethod
    def cross_decode(
        ref_models: Union[CebraOwn, List[CebraOwn], Dict[str, CebraOwn]],
        models: Union[CebraOwn, List[CebraOwn], Dict[str, CebraOwn]] = None,
        title="Decoding perfomance between Models",
        additional_title: str = "",
        xticks: Optional[List[str]] = None,
        adapt: bool = True,
        n_neighbors: int = None,
        plot: Union[bool, str] = True,
        save_dir: Optional[Union[str, Path]] = None,
        regenerate: bool = False,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Decodes the data using the one-to-one decoding method.

        It is important to mention, that the model name "random" is reserved for the random model.
        The behavior variables will be shuffled for decoding.

        Parameters
        ----------
        ref_models : Union[Model, List[Model]]
            The reference models to use for decoding.
        models : Union[Model, List[Model]], optional
            The models with data to adapt to the reference models (default is None).
            If not provided, the reference models will be used.
        xticks : Optional[List[str]], optional
            The labels for the x-axis (default is None). If not provided, the model names will be used.
            xticks must match the number of comparisons.
        adapt : bool, optional
            Whether to adapt the model (default is True). Currently only CEBRA models can be adapted.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is None).
            if None, the number of neighbors will be determined by k-fold cross-validation in the decoding function.
        plot : Union[bool, str], optional
            Whether to plot the decoding results (default is True).
            - If True, the results will be plotted.
            - If "single", multiple plots will be created for each reference model.
        add_random : bool, optional
            Whether to add a random model to the decoding (default is True).
        regenerate : bool, optional
            Whether to regenerate the adapted models (default is False).

        Returns
        -------
        model_decoding_statistics : Dict[str, Dict[str, Union[float, np.ndarray]]]
            A dictionary containing the decoding statistics.
        """

        # convert reference models to dict
        if not isinstance(ref_models, dict):
            ref_models = make_list_ifnot(ref_models)
            # ensure that model_name is unique
            unique_ref_models = {}
            for model in ref_models:
                model_name = (
                    create_unique_dict_key(unique_ref_models, model.name)
                    if model.name in unique_ref_models.keys()
                    else model.name
                )
                unique_ref_models[model_name] = model
            ref_models = unique_ref_models

        if models is None:
            models = ref_models

        # convert models to dict
        if not isinstance(models, dict):
            models = make_list_ifnot(models)
            # ensure that model_name is unique
            unique_models = {}
            for model in models:
                model_name = (
                    create_unique_dict_key(unique_models, model.name)
                    if model.name in unique_models.keys()
                    else model.name
                )
                unique_models[model_name] = model
            models = unique_models

        # dummy decoding statistics
        dummy_decoding_stats = {
            "k_neighbors": np.nan,
            "rmse": {
                "mean": np.nan,
                "variance": np.nan,
            },
            "r2": {
                "mean": np.nan,
                "variance": np.nan,
            },
        }

        cross_model_decoding_statistics = {}
        for ref_model_name, ref_model in iter_dict_with_progress(
            ref_models, desc="Decoding statistics"
        ):
            global_logger.info(f"------------ Decoding statistics for {ref_model_name}")
            # create individual random model for each unique model
            if ref_model.bad_dataset:
                global_logger.warning(
                    f"Model {ref_model_name} has no proper datasets. Skipping..."
                )
                decoding_stats = dummy_decoding_stats.copy()
            else:
                decoding_stats = ref_model.define_decoding_statistics(
                    regenerate=regenerate,
                    n_neighbors=n_neighbors,
                )
            ref_model_decoding_statistics = {ref_model_name: decoding_stats}

            for model_name2, model2 in models.items():
                if ref_model.behavior_data_types != model2.behavior_data_types:
                    do_critical(
                        ValueError,
                        f"Labels between models are not describing the same data types {ref_model.behavior_data_types} and {model2.behavior_data_types}.",
                    )

                if model_name2 == "random":
                    adapt_to_model = ref_model.make_random()
                else:
                    adapt_to_model = model2

                if ref_model.bad_dataset or adapt_to_model.bad_dataset:
                    global_logger.warning(
                        f"Model {model_name2} or {ref_model_name} has no proper datasets. Skipping..."
                    )
                    decoding_stats = dummy_decoding_stats.copy()
                else:
                    global_logger.info(f"Decoding to {model_name2}")

                    neural_data_train_to_embedd = adapt_to_model.get_data(
                        train_or_test="train", type="binarized"
                    )
                    neural_data_test_to_embedd = adapt_to_model.get_data(
                        train_or_test="test", type="binarized"
                    )
                    labels_train = adapt_to_model.get_data(
                        train_or_test="train", type="behavior"
                    )
                    labels_test = adapt_to_model.get_data(
                        train_or_test="test", type="behavior"
                    )

                    to_name = (
                        adapt_to_model.get_metadata("neural", "task_id")
                        if model_name2 != "random"
                        else "random"
                    )

                    adapted_model = (
                        ref_model.adapt(
                            neural_data_train_to_embedd,
                            labels_train,
                            to_name=to_name,
                            regenerate=regenerate,
                        )
                        if adapt
                        else ref_model
                    )

                    decoding_stats = adapted_model.define_decoding_statistics(
                        to_name=adapt_to_model.get_metadata("neural", "task_id"),
                        regenerate=regenerate,
                        neural_data_train_to_embedd=neural_data_train_to_embedd,
                        neural_data_test_to_embedd=neural_data_test_to_embedd,
                        labels_train=labels_train,
                        labels_test=labels_test,
                        n_neighbors=n_neighbors,
                    )
                ref_model_decoding_statistics["to " + model_name2] = decoding_stats

            cross_model_decoding_statistics[ref_model_name] = (
                ref_model_decoding_statistics
            )

        if plot:
            if xticks is not None and len(xticks) != len(ref_models) * len(
                models
            ) + len(ref_models):
                global_logger.warning(
                    f"Plotting one to tone decoding xticks must be a list of length {len(ref_models) * len(models)}, got {len(xticks)}. Using model names instead."
                )
                xticks = None
            else:
                xticks = Models.xticks_from_model_dict(ref_models)

            if len(cross_model_decoding_statistics) == 1 or plot == "single":
                # format decoding ouput for plotting
                for i, (ref_model_name, ref_to_decodings) in enumerate(
                    cross_model_decoding_statistics.items()
                ):
                    xticks_single = (
                        xticks[i * (len(models) + 1) : (i + 1) * (len(models) + 1)]
                        if xticks
                        else None
                    )

                    plot_dict = {}
                    for key, value in ref_to_decodings.items():
                        plot_dict[key] = {
                            "mean": value["rmse"]["mean"],
                            "variance": value["rmse"]["variance"],
                        }
                    Vizualizer.barplot_from_dict(
                        plot_dict,
                        xticks=xticks_single,
                        title=title,
                        ylabel="RMSE (cm)",
                        xlabel="Models",
                        additional_title=additional_title
                        + f" reference - Task {xticks_single[0]}",
                        save_dir=save_dir,
                    )

            elif len(cross_model_decoding_statistics) > 1:
                # format decoding ouput for plotting
                plot_cross_model_decoding_statistics = {}
                for i, (ref_model_name, ref_to_decodings) in enumerate(
                    cross_model_decoding_statistics.items()
                ):
                    plot_ref_model_decoding_statistics = {}
                    for j, (ref_to_name, ref_to_decoding) in enumerate(
                        ref_to_decodings.items()
                    ):
                        plot_ref_model_decoding_statistics[f"{ref_to_name}"] = {
                            "mean": ref_to_decoding["rmse"]["mean"],
                            "variance": ref_to_decoding["rmse"]["variance"],
                        }
                    plot_cross_model_decoding_statistics[ref_model_name] = (
                        plot_ref_model_decoding_statistics
                    )

                Vizualizer.barplot_from_dict_of_dicts(
                    plot_cross_model_decoding_statistics,
                    title=title,
                    xticks=xticks,
                    additional_title=additional_title,
                    save_dir=save_dir,
                )

        return cross_model_decoding_statistics

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
        plot: Union[bool, str] = True,
        additional_title: str = None,
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ):
        """
        Calculate structural indices given distributions and labels inherited from the model.
        """
        model_class, models, plot = self.get_models_check_plot(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            plot=plot,
        )

        structural_indices = {}
        model: CebraOwn
        for model_name, model in models.items():
            global_logger.info(f"Calculating structural indices for {model_name}")
            model_structural_indices = model.structure_index(
                params=copy.deepcopy(params),
                labels=labels,
                to_transform_data=to_transform_data,
                to_2d=to_2d,
                space=space,
                to_name=to_name,
                regenerate=regenerate,
                plot=plot,
                additional_title=additional_title,
                as_pdf=as_pdf,
                plot_save_dir=plot_save_dir,
            )
            if model_structural_indices is None:
                global_logger.error(
                    f"Model {model_name} has no structural indices. Skipping..."
                )
                continue
            structural_indices[model_name] = model_structural_indices

        return structural_indices

    def feature_similarity(
        self,
        manifolds_pipeline: str = "cebra",
        model_naming_filter_include: Union[str, List[str], List[List[str]]] = None,
        model_naming_filter_exclude: Union[str, List[str], List[List[str]]] = None,
        similarity: Literal["pairwise", "inside", "outside"] = "inside",
        out_det_method: Literal["density", "contamination"] = "density",
        remove_outliers: bool = True,
        parallel: bool = True,
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        metric: str = "cosine",
        plot: Union[bool, str] = True,
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        regenerate: bool = False,
        additional_title: str = None,
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ):
        """
        Calculate structural indices given distributions and labels inherited from the model.
        """
        model_class, models, plot = self.get_models_check_plot(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
            plot=plot,
        )

        if len(models) == 0:
            do_critical(
                ValueError,
                f"No models found with filters: {model_naming_filter_include}, {model_naming_filter_exclude}",
            )
        elif len(models) > 1:
            do_critical(
                ValueError,
                f"Found multiple models: {len(models)} with filters: {model_naming_filter_include}, {model_naming_filter_exclude}. Not implemented yet.",
            )

        structural_indices = {}
        model: CebraOwn
        for model_name, model in models.items():
            global_logger.info(f"Calculating feature similarity for {model_name}")
            fsim = model.feature_similarity(
                to_transform_data=to_transform_data,
                labels=labels,
                space=space,
                metric=metric,
                similarity=similarity,
                out_det_method=out_det_method,
                remove_outliers=remove_outliers,
                parallel=parallel,
                to_name=to_name,
                to_2d=to_2d,
                regenerate=regenerate,
                plot=plot,
                additional_title=additional_title,
                as_pdf=as_pdf,
                plot_save_dir=plot_save_dir,
            )
            structural_indices[model_name] = fsim

        return structural_indices

    def get_models_check_plot(
        self,
        labels_name: str = "",
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
        manifolds_pipeline="cebra",
        plot: Union[bool, str, List[str]] = True,
    ):
        models_class, models = self.get_pipeline_models(
            manifolds_pipeline=manifolds_pipeline,
            model_naming_filter_include=model_naming_filter_include,
            model_naming_filter_exclude=model_naming_filter_exclude,
        )
        if plot is not False:
            if isinstance(plot, str):
                plot = [plot]
            if isinstance(plot, list):
                plot = True if "Model" in plot else False
            elif not isinstance(plot, bool):
                do_critical(
                    ValueError, f"plot must be bool or str, got {type(plot)} instead."
                )
        return models_class, models, plot


class ModelsWrapper:
    """
    Meta Class for Models wrapper
    """

    def __init__(self, model_dir, model_settings=None, data_filter: DataFilter = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_settings = model_settings
        self.data_filter: DataFilter = data_filter
        self.models = {}

    def get_models(
        self,
        model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
        model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
    ) -> Dict[str, Model]:
        """
        Get models from the model wrapper class.
        Parameters:
        -----------
            model_naming_filter_include: Union[List[List[str]], List[str], str] = None,
                Filter for model names to include. If None, all models will be included. 3 levels of filtering are possible.
                1. Include all models containing a specific string: "string"
                2. Include all models containing a specific combination of strings: ["string1", "string2"]
                3. Include all models containing one of the string combinations: [["string1", "string2"], ["string3", "string4"]]
            model_naming_filter_exclude: Union[List[List[str]], List[str], str] = None,
                Same as model_naming_filter_include but for excluding models.
        """
        filtered_models = filter_dict_by_properties(
            dictionary=self.models,
            include_properties=model_naming_filter_include,
            exclude_properties=model_naming_filter_exclude,
        )
        return filtered_models


class Model:
    def __init__(
        self, model_dir, model_id, model_settings=None, data_filter: DataFilter = None
    ):
        self.model_dir = model_dir
        self.model_id = model_id
        self.model_settings = model_settings
        self.model_dir.mkdir(exist_ok=True)
        self.data_filter: DataFilter = data_filter
        self.fitted = False
        self.data = None
        self.data_transformation = None
        self.data_split_ratio = None
        self.data_shuffled = None
        self.data_movement_state = None
        self.behavior_data_types = None
        self.name = None
        self.save_path = None

    def define_parameter_save_path(self, model):
        raise NotImplementedError(
            f"define_parameter_save_path not implemented for {self.__class__}"
        )

    def is_fitted(self, model):
        raise NotImplementedError(f"fitted not implemented for {self.__class__}")

    def load_fitted_model(self, model):
        raise NotImplementedError(
            f"load_fitted_model not implemented for {self.__class__}"
        )

    def init_model(self, model_settings_dict):
        if len(model_settings_dict) == 0:
            model_settings_dict = self.model_settings
        initial_model = define_cls_attributes(self, model_settings_dict, override=True)
        initial_model.fitted = False
        return initial_model

    def model_settings_start(self, name, model_settings_dict):
        model = self.init_model(model_settings_dict)
        model.name = name
        return model

    def model_settings_end(self, model):
        model.save_path = self.define_parameter_save_path(model)
        model = self.load_fitted_model(model)
        return model


class PlaceCellDetectors(ModelsWrapper):
    def __init__(self, model_dir, model_id, model_settings=None, **kwargs):
        super().__init__(model_dir, model_settings, **kwargs)
        self.si_model = self.spatial_information(name=model_id)
        self.rate_map = None
        self.time_map = None
        self.dim = None  # Track whether data is 1D or 2D

    def spatial_information(self, name="si", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model_dir = self.model_dir.joinpath("spatial_information")

        model = SpatialInformation(
            model_dir, model_id=name, model_settings=model_settings
        )
        self.models[model.name] = model
        return model

    def get_maps(
        self,
        activity,
        binned_pos,
        category_map,
        fps: Union[float, int] = None,
        smooth=True,
        window_size=2,
        max_bin=None,
    ):
        if self.rate_map is None or self.time_map is None:
            self.rate_map, self.time_map = self.get_rate_map(
                activity,
                binned_pos,
                smooth=smooth,
                window_size=window_size,
                max_bin=max_bin,
                category_map=category_map,
                fps=fps,
            )
        return self.rate_map, self.time_map

    @staticmethod
    def get_time_map(
        binned_pos: np.ndarray,
        max_bin: Union[int, tuple],
        category_map: Dict[Union[int, tuple], int],
        fps: Union[float, int] = None,
        plot: bool = False,
    ):
        """
        Calculates time (occupancy) map for 1D array of binned positions.

        Parameters
        ----------
        binned_pos : np.ndarray
            Binned position data: (n_frames,) for 1D or (n_frames, 2) for 2D.
        fps : float, optional
            Frames per second, used to convert counts to seconds if provided.

        Returns
        -------
        time_map : np.ndarray
            Occupancy map: (bins,) for 1D or (bins_x, bins_y) for 2D.
        """
        example_key = next(iter(category_map))
        if isinstance(example_key, (int, np.integer)):
            bin_shape = 1
        elif isinstance(example_key, (list, tuple, np.ndarray)):
            bin_shape = len(example_key)
        else:
            do_critical(
                ValueError,
                f"category_map key must be an int or a list/tuple of ints, got {type(example_key)}.",
            )

        if len(max_bin) != bin_shape:
            do_critical(
                ValueError,
                f"max_bin must be a single int for 1D or a tuple of two ints for 2D, got {max_bin}.",
            )

        time_map = np.zeros(max_bin, dtype=np.float64)
        if bin_shape == 1:
            for pos, bin_idx in category_map.items():
                if 0 <= bin_idx <= max_bin:
                    # Count occurrences of each binned position
                    time_map[bin_idx] = np.sum(binned_pos == bin_idx)
        elif bin_shape == 2:
            max_x_bins, max_y_bins = max_bin
            time_map = np.zeros((max_x_bins, max_y_bins), dtype=np.float64)
            for (bin_x, bin_y), bin_idx in category_map.items():
                if 0 <= bin_x <= max_x_bins and 0 <= bin_y <= max_y_bins:
                    # Count occurrences of each binned position
                    time_map[bin_x, bin_y] = np.sum(binned_pos == bin_idx)
        else:
            do_critical(
                ValueError,
                f"category_map must be a list of length 2 or lower, got {len(example_key)}.",
            )

        if fps:
            time_map = time_map / fps
        else:
            global_logger.warning("No fps provided, returning frame map in counts.")

        if plot:
            # Plot the time map as a heatmap
            color_bar_label = (
                "Time (counts)" if fps is None else f"Time (s) at {fps} fps"
            )
            if bin_shape == 1:
                plt.figure(figsize=(10, 5))
                plt.bar(range(max_bin), time_map, color="blue")
                plt.xlabel("Position Bin")
                plt.ylabel(f"{color_bar_label}")
                plt.title("Time Map (1D)")
            elif bin_shape == 2:
                plt.imshow(
                    time_map, cmap="viridis", aspect="auto", interpolation="none"
                )
                plt.colorbar(label=f"{color_bar_label}")
                plt.xlabel("Position Bin X")
                plt.ylabel("Position Bin Y")
                plt.title("Time Map (2D)")
            else:
                do_critical(ValueError, "Invalid bin shape for plotting.")
            plt.show()

        return time_map

    @staticmethod
    def get_spike_map(
        activity: np.ndarray,
        binned_pos: np.ndarray,
        max_bin: Union[int, tuple],
        category_map: Dict[Union[int, tuple], int],
        plot: bool = False,
    ):
        """
        Computes the spike map for given neural activity and binned positions.

        Parameters
        ----------
        activity : np.ndarray
            Neural activity: (n_frames, num_cells) or (n_frames,) for one cell.
        pos : np.ndarray
            Binned position data: (n_frames,) for 1D or (n_frames, 2) for 2D.
        max_bin : int or tuple, optional
            Maximum bin index: int for 1D or (max_bin_x, max_bin_y) for 2D.
        category_map : int or tuple, optional
            Number of bins: int for 1D or (bins_x, bins_y) for 2D.

        Returns
        -------
        spike_map : np.ndarray
            Spike map: (num_cells, bins) for 1D or (num_cells, bins_x, bins_y) for 2D.
        """
        activity_2d = np.atleast_2d(activity)
        if activity.ndim > 2:
            do_critical(ValueError, "Activity data  has more than 2 dimensions.")

        num_frames, num_cells = activity_2d.shape
        example_key = next(iter(category_map))
        if isinstance(example_key, (int, np.integer)):
            bin_shape = 1
        elif isinstance(example_key, (list, tuple, np.ndarray)):
            bin_shape = len(example_key)
        else:
            do_critical(
                ValueError,
                f"category_map key must be an int or a list/tuple of ints, got {type(example_key)}.",
            )

        if len(max_bin) != bin_shape:
            do_critical(
                ValueError,
                f"max_bin must be a single int for 1D or a tuple of two ints for 2D, got {max_bin}.",
            )

        # for every frame count the activity of each cell
        if bin_shape == 1:
            spike_map = np.zeros((num_cells, max_bin), dtype=np.float64)
            for pos, bin_idx in category_map.items():
                if 0 <= bin_idx <= max_bin:
                    # Find frames where binned_pos matches this position
                    mask = binned_pos == bin_idx
                    # Sum activity for all neurons at these frames
                    spike_map[:, pos] = np.sum(activity_2d[mask], axis=0)
            # # Old method
            # for frame in prange(num_frames):
            #     rate_map_vec = activity_2d[frame]
            #     pos_at_frame = binned_pos[frame]
            #     spike_map[:, pos_at_frame] += rate_map_vec
            raise NotImplementedError("This method was not tested for 1D data.")
        elif bin_shape == 2:
            max_x_bins, max_y_bins = max_bin
            spike_map = np.zeros((num_cells, max_x_bins, max_y_bins), dtype=np.float64)
            for (bin_x, bin_y), bin_idx in category_map.items():
                if 0 <= bin_x <= max_x_bins and 0 <= bin_y <= max_y_bins:
                    # Find frames where binned_pos matches this position
                    mask = binned_pos == bin_idx
                    # Sum activity for all neurons at these frames
                    spike_map[:, bin_x, bin_y] = np.sum(activity_2d[mask], axis=0)

        else:
            do_critical(
                ValueError,
                f"category_map must be a list of length 2 or lower, got {len(example_key)}.",
            )

        if plot:
            Vizualizer.plot_cell_activites_heatmap(
                spike_map,
                additional_title="(Spike)",
                cbar_title="Spike Count",
            )
        return spike_map

    @staticmethod
    # @njit(parallel=True)
    def get_spike_map_per_laps(cell_neural_data_by_laps, binned_pos_by_lap, max_bin):
        """
        Computes the spike map for all laps.

        Args:
            cell_neural_data_by_laps (list of np.ndarray): A list where each element is a 2D array representing the neural activity for each lap.
            binned_pos_by_lap (list of np.ndarray): A list where each element is a 1D array representing the binned positions for each lap.
            max_bin (int): The maximum bin value for the positions.

        Returns:
            np.ndarray: A 3D array where the first dimension represents laps, the second dimension represents cells, and the third dimension represents bins. The value at (i, j, k) represents the summed activity of cell `j` at bin `k` during lap `i`.
        """
        # count spikes at position
        num_laps = len(cell_neural_data_by_laps)
        num_cells = cell_neural_data_by_laps[0].shape[1]
        cell_lap_activity = np.zeros((num_cells, num_laps, max_bin))
        for lap, (cell_neural_by_lap, lap_pos) in enumerate(
            zip(cell_neural_data_by_laps, binned_pos_by_lap)
        ):

            # this should be the following function, but was changed to the following for numba compatibility
            cells_spikes_at = PlaceCellDetectors.get_spike_map(
                cell_neural_by_lap, lap_pos, max_bin
            )
            ##################### numba compatibility #####################
            # activity_2d = cell_neural_by_lap
            # if activity_2d.ndim == 1:
            #    activity_2d = activity_2d.reshape(-1, 1)
            # elif activity_2d.ndim > 2:
            #    raise ValueError("Activity data has more than 2 dimensions.")
            #
            # num_cells = activity_2d.shape[1]
            ## for every frame count the activity of each cell
            # spike_map = np.zeros((num_cells, max_bin))
            # for frame in prange(len(activity_2d)):
            #    rate_map_vec = activity_2d[frame]
            #    pos_at_frame = lap_pos[frame]
            #    spike_map[:, pos_at_frame] += rate_map_vec
            # counts_at = = spike_map
            ##################### numba compatibility #####################
            cell_lap_activity[:, lap, :] = cells_spikes_at
        raise ValueError(
            "Spike map per laps already calculated. Use get_spike_map_per_laps() to recalculate."
        )
        # get_spike_map_per_laps(cell_neural_data_by_laps, binned_pos_by_lap, max_bin=None, bins=None, range=None):
        """
        Computes the spike map for all laps, supporting 1D or 2D data.

        Parameters
        ----------
        cell_neural_data_by_laps : list of np.ndarray
            Neural activity per lap: each element is (n_frames, num_cells).
        binned_pos_by_lap : list of np.ndarray
            Binned positions per lap: each element is (n_frames,) for 1D or (n_frames, 2) for 2D.
        max_bin : int or tuple, optional
            Maximum bin index: int for 1D or (max_bin_x, max_bin_y) for 2D.
        bins : int or tuple, optional
            Number of bins: int for 1D or (bins_x, bins_y) for 2D.
        range : tuple or list of tuples, optional
            Range for bins: (min, max) for 1D or [(x_min, x_max), (y_min, y_max)] for 2D.

        Returns
        -------
        cell_lap_activity : np.ndarray
            Spike maps: (num_cells, num_laps, bins) for 1D or (num_cells, num_laps, bins_x, bins_y) for 2D.
        """
        num_laps = len(cell_neural_data_by_laps)
        num_cells = cell_neural_data_by_laps[0].shape[1]
        is_2d = binned_pos_by_lap[0].ndim == 2 and binned_pos_by_lap[0].shape[1] == 2

        if is_2d:
            bins = bins or (20, 20)
            max_bin = max_bin or (
                int(np.max([lap[:, 0] for lap in binned_pos_by_lap])) + 1,
                int(np.max([lap[:, 1] for lap in binned_pos_by_lap])) + 1,
            )
            bins = max_bin if bins is None else bins
            cell_lap_activity = np.zeros((num_cells, num_laps, bins[0], bins[1]))
        else:
            max_bin = (
                max_bin or int(np.max([lap.max() for lap in binned_pos_by_lap])) + 1
            )
            bins = bins or max_bin
            cell_lap_activity = np.zeros((num_cells, num_laps, bins))

        for lap, (cell_neural_by_lap, lap_pos) in enumerate(
            zip(cell_neural_data_by_laps, binned_pos_by_lap)
        ):
            cells_spikes_at = PlaceCellDetectors.get_spike_map(
                cell_neural_by_lap, lap_pos, max_bin=max_bin, bins=bins, range=range
            )
            cell_lap_activity[:, lap] = cells_spikes_at
        return cell_lap_activity

    def extract_all_spike_map_per_lap(
        self, cell_ids, neural_data_by_laps, binned_pos_by_laps, max_bin
    ):
        """
        Extracts the spike map for each cell across all laps.

        Args:
            cell_ids (list): A list of cell IDs.
            neural_data_by_laps (list of list of np.ndarray): A list where each element is a list of 2D arrays representing the neural activity for each lap for each cell.
            binned_pos_by_laps (list of list of np.ndarray): A list where each element is a list of 1D arrays representing the binned positions for each lap for each cell.
            max_bin (int): The maximum bin value for the positions.

        Returns:
            np.ndarray: A 4D array where the first dimension represents cells, the second dimension represents laps, the third dimension represents bins. The value at (i, j, k) represents the summed activity of cell `i` at bin `k` during lap `j`.
        """
        cell_lap_activities = np.zeros(
            (len(cell_ids), len(neural_data_by_laps), len(binned_pos_by_laps[0]))
        )
        for cell_id in range(len(cell_ids)):
            cell_neural_data_by_laps = neural_data_by_laps[cell_id]
            cell_lap_activity = self.get_spike_map_per_laps(
                cell_neural_data_by_laps, binned_pos_by_laps, max_bin
            )
            cell_lap_activities[cell_id] = cell_lap_activity
        raise ValueError(
            "Spike map per laps already calculated. Use extract_all_spike_map_per_lap() to recalculate."
        )
        # extract_all_spike_map_per_lap(cell_ids, neural_data_by_laps, binned_pos_by_laps, max_bin=None, bins=None, range=None):
        """
        Extracts spike maps for each cell across all laps.

        Parameters
        ----------
        cell_ids : list
            List of cell IDs.
        neural_data_by_laps : list of list of np.ndarray
            Neural activity per lap for each cell.
        binned_pos_by_laps : list of np.ndarray
            Binned positions per lap.
        max_bin : int or tuple, optional
            Maximum bin index.
        bins : int or tuple, optional
            Number of bins.
        range : tuple or list of tuples, optional
            Range for bins.

        Returns
        -------
        cell_lap_activities : np.ndarray
            Spike maps: (num_cells, num_laps, bins) for 1D or (num_cells, num_laps, bins_x, bins_y) for 2D.
        """
        cell_lap_activities = []
        for cell_id in range(len(cell_ids)):
            cell_neural_data_by_laps = neural_data_by_laps[cell_id]
            cell_lap_activity = PlaceCellDetectors.get_spike_map_per_laps(
                cell_neural_data_by_laps, binned_pos_by_laps, max_bin, bins, range
            )
            cell_lap_activities.append(cell_lap_activity)
        return np.stack(cell_lap_activities, axis=0)
        return cell_lap_activities

    @staticmethod
    def get_spike_maps_per_laps(cell_ids, neural_data, behavior: Datasets):
        """
        Computes the spike map for specified cells across all laps.

        Args:
            cell_ids (list): A list of cell IDs to compute the spike maps for.
            neural_data (np.ndarray): A 2D array representing the neural activity.
            behavior (Datasets): A dataset containing behavioral data including position information.

        Returns:
            dict: A dictionary where keys are cell IDs and values are dictionaries containing the spike maps for each lap.
        """
        cell_ids = make_list_ifnot(cell_ids)

        binned_pos = behavior.position.binned_data
        max_bin = behavior.position.max_bin or max(binned_pos) + 1

        # get neural_data for each cell
        wanted_neural_data = neural_data[:, cell_ids]
        cell_neural_data_by_laps = behavior.split_by_laps(wanted_neural_data)
        binned_pos_by_laps = behavior.split_by_laps(binned_pos)

        # get spike map for each lap
        cell_lap_activities = PlaceCellDetectors.get_spike_map_per_laps(
            cell_neural_data_by_laps, binned_pos_by_laps, max_bin=max_bin
        )

        cell_activity_dict = {}
        for cell_id, cell_lap_activity in zip(cell_ids, cell_lap_activities):
            cell_activity_dict[cell_id] = {}
            cell_activity_dict[cell_id]["lap_activity"] = cell_lap_activity
        raise ValueError(
            "Spike maps per laps already calculated. Use get_spike_maps_per_laps() to recalculate."
        )
        # get_spike_maps_per_laps(cell_ids, neural_data, behavior):
        """
        Computes spike maps for specified cells across all laps.

        Parameters
        ----------
        cell_ids : list
            List of cell IDs.
        neural_data : np.ndarray
            Neural activity: (n_frames, num_cells).
        behavior : Datasets
            Dataset with position.binned_data and split_by_laps method.

        Returns
        -------
        cell_activity_dict : dict
            Dictionary with cell IDs as keys and lap activities as values.
        """
        cell_ids = make_list_ifnot(cell_ids)
        binned_pos = behavior.position.binned_data
        is_2d = binned_pos.ndim == 2 and binned_pos.shape[1] == 2
        max_bin = (
            (int(np.max(binned_pos[:, 0])) + 1, int(np.max(binned_pos[:, 1])) + 1)
            if is_2d
            else int(np.max(binned_pos)) + 1
        )

        wanted_neural_data = neural_data[:, cell_ids]
        cell_neural_data_by_laps = behavior.split_by_laps(wanted_neural_data)
        binned_pos_by_laps = behavior.split_by_laps(binned_pos)

        cell_lap_activities = PlaceCellDetectors.get_spike_map_per_laps(
            cell_neural_data_by_laps, binned_pos_by_laps, max_bin=max_bin
        )

        cell_activity_dict = {}
        for cell_id, cell_lap_activity in zip(cell_ids, cell_lap_activities):
            cell_activity_dict[cell_id] = {"lap_activity": cell_lap_activity}
        return cell_activity_dict
        return cell_activity_dict

    @staticmethod
    def get_rate_map(
        activity,
        binned_pos,
        category_map,
        max_bin: Union[int, tuple],
        fps: Union[float, int] = None,
        smooth: bool = True,
        window_size: int = 2,
        plot: bool = False,
    ):
        """
        Computes spike rate per position per time for 1D or 2D data.

        Parameters
        ----------
        activity : np.ndarray
            Neural activity: (n_frames, num_cells) or (n_frames,) for one cell.
        binned_pos : np.ndarray
            Binned position data: (n_frames,) for 1D or (n_frames, 2) for 2D.
        max_bin : int or tuple, optional
            Maximum bin index.
        bins : int or tuple, optional
            Number of bins.
        range : tuple or list of tuples, optional
            Range for bins.

        Returns
        -------
        rate_map : np.ndarray
            Rate map: (num_cells, bins) for 1D or (num_cells, bins_x, bins_y) for 2D.
        time_map : np.ndarray
            Time map: (bins,) for 1D or (bins_x, bins_y) for 2D.
        """
        spike_map = PlaceCellDetectors.get_spike_map(
            activity, binned_pos, max_bin, category_map, plot=plot
        )
        # smooth and normalize
        # normalize by spatial occupancy
        time_map = PlaceCellDetectors.get_time_map(
            binned_pos, max_bin=max_bin, category_map=category_map, fps=fps, plot=plot
        )
        rate_map_occupancy = spike_map / (time_map + np.spacing(1))
        rate_map_occupancy = np.nan_to_num(rate_map_occupancy, nan=0.0)

        # normalize by activity
        rate_sum = np.nansum(
            rate_map_occupancy,
            axis=(1, 2) if rate_map_occupancy.ndim == 3 else 1,
            keepdims=True,
        ) + np.spacing(1)
        rate_map = rate_map_occupancy / rate_sum

        if smooth:
            axis = (1, 2) if rate_map.ndim == 3 else 1
            for ax in axis:
                # Smooth the rate map along the specified axis
                rate_map = smooth_array(rate_map, window_size=window_size, axis=ax)

        if plot:
            Vizualizer.plot_cell_activites_heatmap(
                rate_map,
                cbar_title="Spike Rate",
            )

        return rate_map, time_map

    @staticmethod
    def get_rate_map_stats(rate_map, position_PDF=None):
        mean_rate = np.mean(rate_map)
        rmap_mean = np.nanmean(rate_map)
        rmap_peak = np.nanmax(rate_map)
        if position_PDF is not None:
            mean_rate_sq = np.ma.sum(np.power(rate_map, 2) * position_PDF)
        # check overall active
        # calculate sparsity of the rate map
        if mean_rate_sq != 0:
            sparsity = mean_rate * mean_rate / mean_rate_sq
            # get selectivity of the rate map
            # high selectivity would be tuning to one position
            max_rate = np.max(rate_map)
            selectivity = max_rate / mean_rate
        rate_map_stats = {}
        rate_map_stats["rmap_mean"] = rmap_mean
        rate_map_stats["rmap_peak"] = rmap_peak
        rate_map_stats["mean_rate_sq"] = mean_rate_sq
        rate_map_stats["sparsity"] = sparsity
        rate_map_stats["selectivity"] = selectivity

        raise ValueError(
            "Rate map statistics already calculated. Use get_rate_map_stats() to recalculate."
        )
        # get_rate_map_stats(rate_map, position_PDF=None):
        """
        Computes statistics for the rate map.

        Parameters
        ----------
        rate_map : np.ndarray
            Rate map: (num_cells, bins) for 1D or (num_cells, bins_x, bins_y) for 2D.
        position_PDF : np.ndarray, optional
            Position probability distribution, same shape as rate_map[0].

        Returns
        -------
        rate_map_stats : dict
            Dictionary containing mean, peak, sparsity, and selectivity of the rate map.
        """
        axis = (1, 2) if rate_map.ndim == 3 else 1
        rmap_mean = np.nanmean(rate_map, axis=axis)
        rmap_peak = np.nanmax(rate_map, axis=axis)
        mean_rate = np.mean(rate_map, axis=axis)
        mean_rate_sq = (
            np.ma.sum(np.power(rate_map, 2) * position_PDF, axis=axis)
            if position_PDF is not None
            else np.nan
        )
        sparsity = (
            mean_rate * mean_rate / (mean_rate_sq + np.spacing(1))
            if mean_rate_sq is not np.nan
            else np.nan
        )
        selectivity = rmap_peak / (mean_rate + np.spacing(1))

        rate_map_stats = {
            "rmap_mean": rmap_mean,
            "rmap_peak": rmap_peak,
            "mean_rate_sq": mean_rate_sq,
            "sparsity": sparsity,
            "selectivity": selectivity,
        }
        return rate_map_stats


class SpatialInformation(Model):
    """
    A class used to model spatial information in a neuroscience context.

    Attributes
    ----------
    name : str
        The name identifier for the model.
    model_settings : dict
        The settings used for the model.

    Methods
    -------
    create_default_model()
        Creates a default model configuration.
    define_parameter_save_path(model)
        Defines the path where model parameters are saved.
    is_fitted(model)
        Checks if the model has been fitted.
    load_fitted_model(model)
        Loads a fitted model from a saved path.
    get_spatial_information(rate_map, time_map, spatial_information_method="opexebo")
        Computes the spatial information rate and content.
    compute_si_zscores(activity, binned_pos, rate_map, time_map, n_tests=500, spatial_information_method="skaggs", fps=None, max_bin=None)
        Computes spatial information and corresponding z-scores.
    """

    def __init__(
        self, model_dir, model_id, model_settings=None, data_filter: DataFilter = None
    ):
        """
        Initializes the SpatialInformation model.

        Parameters
        ----------
        model_dir : Path
            The directory where the model is stored.
        model_id : str
            The identifier for the model.
        model_settings : dict, optional
            The settings for the model (default is None).
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(model_dir, model_id, model_settings, data_filter)
        self.name = "si"
        self.model_settings = model_settings
        self.model_settings_start(self.name, model_settings)
        self.model_settings_end(self)

    def define_parameter_save_path(self, model):
        """
        Defines the path where model parameters are saved.

        Parameters
        ----------
        model : Model
            The model for which the save path is defined.

        Returns
        -------
        Path
            The path where the model parameters are saved.
        """
        # TODO: implement the usage of this function
        save_path = self.model_dir.joinpath(
            f"place_cell_{model.name}_{self.model_id}.npz"
        )
        return save_path

    def is_fitted(self, model):
        """
        Checks if the model has been fitted to the data.

        Parameters
        ----------
        model : Model
            The model to check.

        Returns
        -------
        bool
            True if the model has been fitted, False otherwise.
        """
        # TODO: implement the usage of this function
        return model.save_path.exists()

    def load_fitted_model(self, model):
        """
        Loads a fitted model parameters from from a saved path.

        Parameters
        ----------
        model : Model
            The model to load.

        Returns
        -------
        Model
            The SpatialInformation loaded model.
        """
        fitted_model_path = model.save_path
        if fitted_model_path.exists():
            fitted_model = np.load(fitted_model_path)
            # TODO: load data from model to prevent recalculation
            raise NotImplementedError(
                f"load_fitted_model not implemented for {self.__class__}"
            )
        model.fitted = self.is_fitted(model)
        return model

    @staticmethod
    def get_spatial_information(
        rate_map, time_map=None, spatial_information_method="skaggs"
    ):
        """
        #FIXME: This is old documentation, change it to the new one
        Computes the spatial information rate and content.

        Parameters
        ----------
        rate_map: np.ma.MaskedArray
            Smoothed rate map: n x m array where cell value is the firing rate,
            masked at locations with low occupancy

        time_map: np.ma.MaskedArray
            time map: n x m array where the cell value is the time the animal spent
            in each cell, masked at locations with low occupancy
            Already smoothed

        Returns
        -------
        spatial_information_rate: n x m array where cell value is the float information rate [bits/sec]
        spatial_information_content: n x m array where cell value is the float spatial information content [bits/spike]
        """
        # duration = np.sum(time_map)  # in frames
        ## spacing adds floating point precision to avoid DivideByZero errors
        # position_PDF = time_map / (duration + np.spacing(1))
        ## use position pdf
        # p_spike = rate_map * position_PDF + np.spacing(1)

        p_spike = np.nansum(rate_map, axis=1)
        p_position = np.nansum(rate_map, axis=0)

        # mean_rate = np.sum(p_spike, axis=1) + np.spacing(1)

        # get statistics of the rate map
        # rate_map_stats = get_rate_map_stats(rate_map, position_PDF)

        # if np.sum(mean_rate) == 0:
        #    raise ValueError("Mean firering rate is 0, no brain activity")

        # new axis is added to ensure that the division is done along the right axis
        # log_argument = rate_map / mean_rate[:, np.newaxis]
        p_spike_at_pos = p_spike[:, None] * p_position
        log_argument = rate_map / p_spike_at_pos
        # ensure no number is below 1 before taking the log out of it
        log_argument[log_argument < 1] = 1

        if spatial_information_method == "skaggs":
            inf_rate = np.nansum(rate_map * np.log2(log_argument), axis=1)
            # FIXME: is this correct?
            # inf_rate = np.nansum(
            #    p_spike * np.log2(log_argument), axis=1
            # )
        elif spatial_information_method == "shanon":
            inf_rate = scipy.stats.entropy(pk=log_argument, axis=1)
        elif spatial_information_method == "kullback-leiber":
            inf_rate = scipy.stats.entropy(
                pk=rate_map, qk=p_spike_at_pos[:, np.newaxis], axis=1
            )
        else:
            raise ValueError("Spatial information method not recognized")

        # FIXME: is this correct?
        inf_content = inf_rate * p_spike

        raise NotImplementedError(
            f"SpatialInformation.get_spatial_information not implemented for {spatial_information_method}"
        )
        """
        Computes spatial information rate and content for 1D or 2D rate maps.

        Parameters
        ----------
        rate_map : np.ndarray
            Smoothed rate map: (num_cells, bins) for 1D or (num_cells, bins_x, bins_y) for 2D.
        time_map : np.ndarray, optional
            Time map: (bins,) for 1D or (bins_x, bins_y) for 2D, representing time spent in each bin.
        spatial_information_method : str, optional
            Method to compute spatial information ('skaggs', 'shannon', 'kullback-leibler').

        Returns
        -------
        inf_rate : np.ndarray
            Spatial information rate [bits/sec] per cell.
        inf_content : np.ndarray
            Spatial information content [bits/spike] per cell.
        """
        if rate_map.ndim == 2:
            # 1D case
            p_spike = np.nansum(rate_map, axis=1, keepdims=True)
            p_position = np.nansum(rate_map, axis=0, keepdims=False)
            p_spike_at_pos = p_spike * p_position
            log_argument = rate_map / p_spike_at_pos
            log_argument[log_argument < 1] = 1
        elif rate_map.ndim == 3:
            # 2D case
            p_spike = np.nansum(rate_map, axis=(1, 2), keepdims=True)
            p_position = np.nansum(rate_map, axis=0, keepdims=False)
            p_spike_at_pos = p_spike * p_position
            log_argument = rate_map / p_spike_at_pos
            log_argument[log_argument < 1] = 1
        else:
            raise ValueError("Rate map must be 2D (1D data) or 3D (2D data).")

        if spatial_information_method == "skaggs":
            inf_rate = np.nansum(
                rate_map * np.log2(log_argument),
                axis=(1, 2) if rate_map.ndim == 3 else 1,
            )
        elif spatial_information_method == "shannon":
            inf_rate = stats.entropy(
                log_argument, axis=(1, 2) if rate_map.ndim == 3 else 1
            )
        elif spatial_information_method == "kullback-leibler":
            inf_rate = stats.entropy(
                pk=rate_map, qk=p_spike_at_pos, axis=(1, 2) if rate_map.ndim == 3 else 1
            )
        else:
            raise ValueError("Spatial information method not recognized")

        inf_content = inf_rate / (p_spike.squeeze() + np.spacing(1))

        return inf_rate, inf_content

    def compute_si_zscores(
        self,
        fps,
        activity=None,
        binned_pos=None,
        rate_map=None,
        time_map=None,
        n_tests=500,
        spatial_information_method="skaggs",
        max_bin=None,
    ):
        """
        Computes spatial information and corresponding z-scores.

        Parameters
        ----------
        activity : np.ndarray, optional
            The activity data (default is None).
        binned_pos : np.ndarray, optional
            The binned position data (default is None).
        rate_map : np.ndarray, optional
            The rate map data (default is None).
        time_map : np.ndarray, optional
            The time map data (default is None).
        n_tests : int, optional
            The number of tests for computing z-scores (default is 500).
        spatial_information_method : str, optional
            The method to compute spatial information (default is "skaggs").
        fps : float, optional
            Frames per second (default is None).
        max_bin : int, optional
            The maximum bin (default is None).

        Returns
        -------
        tuple
            A tuple containing:
            - zscore : np.ndarray
                The z-scores for spatial information.
            - si_rate : np.ndarray
                The spatial information rate.
            - si_content : np.ndarray
                The spatial information content.
        """

        rate_map, time_map = PlaceCellDetectors.get_rate_map(
            activity, binned_pos, max_bin=max_bin
        )

        inf_rate, inf_content = self.get_spatial_information(
            rate_map, time_map, spatial_information_method
        )
        si_rate = inf_rate
        si_content = inf_content

        num_cells, len_track = rate_map.shape

        # calculate the information rate for shuffled data:
        si_shuffle = np.zeros((n_tests, num_cells))
        for test_num in trange(n_tests):
            # shuffle the position data
            binned_pos_shuffled = np.roll(
                binned_pos, np.random.choice(np.arange(binned_pos.shape[0]), 1)
            )
            rate_map_shuffled, _ = PlaceCellDetectors.get_rate_map(
                activity, binned_pos_shuffled, max_bin=max_bin
            )

            inf_rate, _ = self.get_spatial_information(
                rate_map_shuffled, spatial_information_method
            )
            si_shuffle[test_num] = inf_rate

        # calculate z-scores for every cell
        stack = np.vstack([si_rate, si_shuffle])
        zscore = scipy.stats.zscore(stack)[0]

        si_rate *= fps
        si_content *= fps
        raise NotImplementedError(
            f"SpatialInformation.compute_si_zscores not implemented for {spatial_information_method}"
        )
        if rate_map is None or time_map is None:
            rate_map, time_map = PlaceCellDetectors.get_rate_map(
                activity, binned_pos, max_bin=max_bin, bins=bins, range=range
            )

        inf_rate, inf_content = self.get_spatial_information(
            rate_map, time_map, spatial_information_method
        )
        si_rate = inf_rate
        si_content = inf_content

        num_cells = rate_map.shape[0]
        si_shuffle = np.zeros((n_tests, num_cells))
        for test_num in trange(n_tests):
            binned_pos_shuffled = np.roll(
                binned_pos, np.random.choice(np.arange(binned_pos.shape[0]), 1), axis=0
            )
            rate_map_shuffled, _ = PlaceCellDetectors.get_rate_map(
                activity, binned_pos_shuffled, max_bin=max_bin, bins=bins, range=range
            )
            inf_rate, _ = self.get_spatial_information(
                rate_map_shuffled, spatial_information_method
            )
            si_shuffle[test_num] = inf_rate

        stack = np.vstack([si_rate, si_shuffle])
        zscore = stats.zscore(stack)[0]

        si_rate *= fps
        si_content *= fps
        return zscore, si_rate, si_content

    def is_fitted(self, model):
        return model.save_path.exists()

    def get_place_cells(self):
        raise NotImplementedError(
            f"get_place_cells not implemented for {self.__class__}"
        )


class Cebras(ModelsWrapper, Model):
    # TODO: restructure this and mode Model to CebraOwn class as well as other functions
    def __init__(
        self, model_dir, model_id, model_settings=None, data_filter: DataFilter = None
    ):
        super().__init__(model_dir, model_settings, data_filter)
        self.model_id = model_id

    def init_model(self, model_settings_dict):
        if len(model_settings_dict) == 0:
            model_settings_dict = self.model_settings
        initial_model = CebraOwn(**model_settings_dict, data_filter=self.data_filter)
        # initial_model = define_cls_attributes(
        #    default_model, model_settings_dict, override=True
        # )
        initial_model.fitted = False
        initial_model.data = None
        initial_model.decoding_statistics = None
        return initial_model

    def model_settings_start(self, name, model_settings_dict):
        model = self.init_model(model_settings_dict)
        model.name = name
        return model

    def model_settings_end(self, model):
        save_path = model.define_parameter_save_path(
            model_dir=self.model_dir, model_id=self.model_id
        )
        model = model.load_fitted_model()
        model.model_id = self.model_id
        self.models[model.name] = model
        return model

    def time(self, name="time", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.temperature = (
            1.12 if kwargs.get("temperature") is None else model.temperature
        )
        model.type = name
        model.conditional = "time" if kwargs.get("time") is None else model.conditional
        model = self.model_settings_end(model)
        return model

    def behavior(self, name="behavior", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.type = name
        model = self.model_settings_end(model)
        return model

    def hybrid(self, name="hybrid", model_settings=None, **kwargs):
        model_settings = model_settings or kwargs
        model = self.model_settings_start(name, model_settings)
        model.hybrid = True if kwargs.get("hybrid") is None else model.hybrid
        model.type = name
        model = self.model_settings_end(model)
        return model

    def create_embeddings(
        self,
        models: Dict[str, Model],
        to_transform_data: Union[np.ndarray, List[np.ndarray]] = None,
        to_2d=False,
        save=False,
        return_labels=False,
    ):
        embeddings = {}
        labels = {}
        for model_name, model in models.items():
            embedding_title = f"{model_name}"
            if return_labels:
                embedding, label = model.create_embedding(
                    model,
                    to_transform_data=to_transform_data,
                    to_2d=to_2d,
                    save=save,
                    return_labels=return_labels,
                )
            else:
                embedding = model.create_embedding(
                    to_transform_data=to_transform_data, to_2d=to_2d, save=save
                )
            if embedding is not None:
                embeddings[embedding_title] = embedding
                if return_labels:
                    labels[embedding_title] = label

        if return_labels:
            return embeddings, labels
        return embeddings


class CebraOwn(CEBRA):
    unimportant_attributes = ["verbose", "max_adapt_iterations"]

    def __init__(
        self,
        model_architecture: str = "offset10-model",
        device: str = "cuda_if_available",
        criterion: str = "infonce",
        distance: str = "cosine",
        conditional: str = "time_delta",
        temperature: float = 1.0,
        temperature_mode: Literal["constant", "auto"] = "constant",
        min_temperature: Optional[float] = 0.1,
        time_offsets: int = 10,
        delta: float = None,
        max_iterations: int = 10000,
        max_adapt_iterations: int = 1500,
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        optimizer: str = "adam",
        output_dimension: int = 3,
        verbose: bool = True,
        num_hidden_units: int = 32,
        pad_before_transform: bool = True,
        hybrid: bool = False,
        optimizer_kwargs: Tuple[Tuple[str, object], ...] = (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-08),
            ("weight_decay", 0),
            ("amsgrad", False),
        ),
        data_filter: DataFilter = None,
    ):
        # TODO: change Cebras class and CebraOwn to make it work when CebraOwn also inherits from Model
        # Model.__init__(self, model_dir, model_id, model_settings=None, **kwargs)
        CEBRA.__init__(
            self,
            model_architecture=model_architecture,
            device=device,
            criterion=criterion,
            distance=distance,
            conditional=conditional,
            temperature=temperature,
            temperature_mode=temperature_mode,
            min_temperature=min_temperature,
            time_offsets=time_offsets,
            delta=delta,
            max_iterations=max_iterations,
            max_adapt_iterations=max_adapt_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            output_dimension=output_dimension,
            verbose=verbose,
            num_hidden_units=num_hidden_units,
            pad_before_transform=pad_before_transform,
            hybrid=hybrid,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.model_type = None
        self.embedding = {
            "train": None,
            "test": None,
        }
        self.name = None
        self.save_path = None
        self.train_info = {
            "train_ids": None,
            "test_ids": None,
            "space": None,
            "transformation": None,
            "split_ratio": None,
            "movement_state": None,
        }
        self.data_filter: DataFilter = data_filter

    def id(
        self,
        get: List[Literal["animal", "date", "task", "model", "stimulus", "id"]] = "id",
    ):
        """
        Returns the model id, which is the name of the model.

        Parameters
        ----------
        get : list of str, optional
            The part of the id to return. Options are "animal", "date", "task", "model", "id". Default is "id".
            Any defined variable name in this function can be used to return the value of that variable.
            If multiple values are provided, they will be joined with an underscore.

        Returns
        -------
        str
            The id of the model or the specified part of the id.
        """
        get = make_list_ifnot(get)
        task_id = self.get_metadata("behavior", "task_id")
        animal, date, task = task_id.split("_")
        model = self.name
        transformation = self.data_transformation
        split_ratio = self.data_split_ratio
        moving_state = self.data_movement_state
        shuffled = "shuffled" if self.data_shuffled else "not_shuffled"
        stimulus = self.get_metadata("behavior", "stimulus_type")
        id = f"{task_id}_{model}"
        id_str = []
        for g in get:
            id_str.append(locals().get(g.lower()))
        return "_".join(id_str) if len(id_str) > 0 else id

    def define_metadata(self, **kwargs):
        """
        Define metadata for the model by setting attributes.

            data_transformation
            data_split_ratio
            data_shuffled
            data_movement_state
            behavior_data_types
            metadata

        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def get_metadata(
        self,
        type: Literal["animal", "session", "neural", "behavior"] = None,
        key: str = None,
    ):
        """
        Get metadata for the model.

        Parameters
        ----------
        type : str, optional
            The type of metadata to get (default is None).
        key : str, optional
            The key of the metadata to get (default is None).

        Returns
        -------
        metadata_info : Any
            The metadata information
        """
        if type is None and key is None:
            metadata_info = self.metadata
        else:
            if type not in self.metadata:
                global_logger.critical(
                    f"Type {type} not found in metadata for {self.__class__}"
                )
                raise ValueError(
                    f"Type {type} not found in metadata for {self.__class__}"
                )

            if type is not None and key is not None:
                if key not in self.metadata[type]:
                    global_logger.error(
                        f"Key {key} not found in metadata for {self.__class__}. Returning None."
                    )
                    metadata_info = None
                else:
                    metadata_info = self.metadata[type][key]
            elif type is not None:
                metadata_info = self.metadata[type]
            elif key is not None:
                global_logger.critical(
                    f"Type must be provided to get metadata from {self.__class__}"
                )
                raise ValueError(
                    f"Type must be provided to get metadata from {self.__class__}"
                )
        return metadata_info

    def define_parameter_save_path(
        self, model_dir: Union[str, Path] = None, model_id: str = None
    ):
        if model_dir is not None:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = self.save_path.parent
        if model_id is None:
            model_id = self.model_id
        self.save_path = self.model_dir.joinpath(
            f"cebra_{self.name}_dim-{self.output_dimension}_model-{model_id}.pt"
        )
        global_logger.debug(
            f"Model save path set to {self.save_path} for {self.__class__}"
        )
        return self.save_path

    def set_name(self, name):
        self.name = name
        global_logger.debug(f"Set model name to {self.name} for {self.__class__}")
        self.define_parameter_save_path()

    def make_random(
        self,
        regenerate: bool = False,
        create_embedding: bool = True,
        verbose: bool = False,
    ):
        random_self = copy.deepcopy(self)
        random_self.set_name(f"{self.name}_random")

        random_self.remove_fitting()
        random_self.load_fitted_model()

        # randomize neural data
        for train_or_test in ["train", "test"]:

            # shuffle embedding data
            emb = random_self.get_data(train_or_test=train_or_test, type="embedding")
            random_self.set_embedding(
                data=Dataset.shuffle(emb), train_or_test=train_or_test
            )

            # shuffle behavior data
            for type in ["idx_train", "idx_test"]:
                idx = random_self.train_info[type]
                random_self.train_info[type] = Dataset.shuffle(idx)

        random_self.train(
            neural_data=random_self.get_data(train_or_test="train"),
            behavior_data=random_self.get_data(train_or_test="train", type="behavior"),
            regenerate=regenerate,
            verbose=verbose,
        )

        if create_embedding:
            train_embedding = random_self.create_embedding(train_or_test="train")
            test_embedding = random_self.create_embedding(train_or_test="test")
            random_self.set_embedding(data=train_embedding, train_or_test="train")
            random_self.set_embedding(data=test_embedding, train_or_test="test")

        return random_self

    @property
    def bad_dataset(self):
        """
        Returns a bad dataset for the model was used.

        Bad is defined by a dataset with different number of frames between neural and behavior data for training.
        """
        neural_data = self.get_data(type="binarized")
        behavior_data = self.get_data(type="behavior")
        if neural_data.shape[0] != behavior_data.shape[0]:
            global_logger.info(
                f"Neural data and behavior data have different number of samples."
            )
            bad = True
        else:
            bad = False
        return bad

    def define_decoding_statistics(
        self,
        neural_data_train_to_embedd: np.ndarray = None,
        neural_data_test_to_embedd: np.ndarray = None,
        labels_train: np.ndarray = None,
        labels_test: np.ndarray = None,
        to_name: str = None,
        n_neighbors: int = None,
        regenerate: bool = False,
    ):
        """
        Decodes the data using the model.

        This function takes the neural data and behavior data, and decodes the neural data using the model.
        It then saves the decoding statistics to a file,
            - either if the model is self-decoding
            - or if a name is provided for the data used for decoding.

        Parameters
        ----------
        neural_data_train_to_embedd : np.ndarray, optional
            The neural data to use for training the model (default is None). If not provided, the training data from the model will be used.
        neural_data_test_to_embedd : np.ndarray, optional
            The neural data to use for testing the model (default is None). If not provided, the testing data from the model will be used.
        labels_train : np.ndarray, optional
            The behavior data to use for training the model (default is None). If not provided, the training data from the model will be used.
        labels_test : np.ndarray, optional
            The behavior data to use for testing the model (default is None). If not provided, the testing data from the model will be used.
        to_name : str, optional
            Name of data used for decoding which is added to the output file name (default is None).
            If None but data is not from self, the decoding statistics will not be saved.
        n_neighbors : int, optional
            The number of neighbors to use for the KNN algorithm (default is None).
            if None, the number of neighbors will be determined by k-fold cross-validation in the decoding function.
        regenerate : bool, optional
            Whether to regenerate the decoding statistics (default is False).

        Returns
        -------
        decoding_statistics : Dict[Dict[str, Any]]
            A dictionary containing the decoding statistics.

        """
        self_decoding = True
        if neural_data_train_to_embedd is not None:
            neural_data_train = self.create_embedding(
                to_transform_data=neural_data_train_to_embedd
            )
            self_decoding = False
        else:
            if self.get_data() is None:
                self.set_data(
                    data=self.create_embedding(
                        to_transform_data=self.get_data(type="binarized")
                    )
                )
            neural_data_train = self.get_data()

        if neural_data_test_to_embedd is not None:
            neural_data_test = self.create_embedding(
                to_transform_data=neural_data_test_to_embedd
            )
            self_decoding = False
        else:
            if self.get_data(train_or_test="test") is None:
                self.set_data(
                    data=self.create_embedding(
                        to_transform_data=self.get_data(
                            train_or_test="test", type="binarized"
                        )
                    ),
                    train_or_test="test",
                )
            neural_data_test = self.get_data(train_or_test="test")

        if labels_train is not None:
            self_decoding = False
        else:
            labels_train = self.get_data(type="behavior")

        if labels_test is not None:
            self_decoding = False
        else:
            labels_test = self.get_data(train_or_test="test", type="behavior")

        # define output file name
        ofname = f"decoding-stats_{self.save_path.stem}"
        if to_name is not None:
            additional_name = f"-to-{to_name}"
            if additional_name not in ofname:
                ofname += f"-to-{to_name}"
            else:
                global_logger.warning(
                    f"Name {to_name} already in decoding statistics file name. Not adding it again."
                )
        ofname += ".npy"

        if not self_decoding and not to_name:
            global_logger.info(
                f"{self.name} not self decoding and no 'to_name' given: Not saving decoding statistics."
            )
            save = False
        else:
            save = True

        if save:
            decoding_stats_output_path = self.model_dir.joinpath("decoding", ofname)
            decoding_statistics = npio(
                decoding_stats_output_path, task="load", file_type="npy"
            )
        else:
            decoding_statistics = None

        decoding_statistics = decoding_statistics
        if decoding_statistics is None or regenerate:
            if neural_data_train is not None and neural_data_train.shape[0] < 10:
                global_logger.error(
                    f"Not enough frames to use for {self.name}. At least 10 are needed, found {neural_data_train.shape[0]}. Skipping"
                )
            else:
                circular_values = self.get_metadata("behavior", "ciruclar_environment")
                max_value = self.get_metadata("behavior", "environment_dimensions")

                decoding_statistics = decode(
                    embedding_train=neural_data_train,
                    embedding_test=neural_data_test,
                    circular_values=circular_values,
                    max_value=max_value,
                    labels_train=labels_train,
                    labels_test=labels_test,
                    n_neighbors=n_neighbors,
                    labels_describe_space=self.labels_describe_space,
                    multiply_by=self.relative_val_scale,
                )
                npio(
                    decoding_stats_output_path,
                    task="save",
                    data=decoding_statistics,
                    file_type="npy",
                )
        else:
            global_logger.debug(
                f"Decoding statistics already calculated. Skipping calculation."
            )
        return decoding_statistics

    @classmethod
    def load(
        cls,
        filename: str,
        backend: Literal["auto", "sklearn", "torch"] = "auto",
        **kwargs,
    ) -> CebraOwn:
        """Load a model from disk.

        Args:
            filename: The path to the file in which to save the trained model.
            backend: A string identifying the used backend.
            kwargs: Optional keyword arguments passed directly to the loader.

        Return:
            The model to load.

        Note:
            Experimental functionality. Do not expect the save/load functionalities to be
            backward compatible yet between CEBRA versions!

            For information about the file format please refer to :py:meth:`cebra.CEBRA.save`.

        Example:

            >>> import cebra
            >>> import numpy as np
            >>> import tempfile
            >>> from pathlib import Path
            >>> tmp_file = Path(tempfile.gettempdir(), 'cebra.pt')
            >>> dataset =  np.random.uniform(0, 1, (1000, 20))
            >>> cebra_model = cebra.CEBRA(max_iterations=10)
            >>> cebra_model.fit(dataset)
            CEBRA(max_iterations=10)
            >>> cebra_model.save(tmp_file)
            >>> loaded_model = cebra.CEBRA.load(tmp_file)
            >>> embedding = loaded_model.transform(dataset)
            >>> tmp_file.unlink()

        """

        supported_backends = ["auto", "sklearn", "torch"]
        if backend not in supported_backends:
            raise NotImplementedError(
                f"Unsupported backend: '{backend}'. Supported backends are: {', '.join(supported_backends)}"
            )

        checkpoint = torch.load(filename, weights_only=False, **kwargs)

        if backend == "auto":
            backend = "sklearn" if isinstance(checkpoint, dict) else "torch"

        if isinstance(checkpoint, dict) and backend == "torch":
            raise RuntimeError(
                f"Cannot use 'torch' backend with a dictionary-based checkpoint. "
                f"Please try a different backend."
            )
        if not isinstance(checkpoint, dict) and backend == "sklearn":
            raise RuntimeError(
                f"Cannot use 'sklearn' backend a non dictionary-based checkpoint. "
                f"Please try a different backend."
            )

        if backend == "sklearn":
            cebra_ = _load_cebra_with_sklearn_backend(checkpoint)
        else:
            cebra_ = _check_type_checkpoint(checkpoint)

        return cebra_

    def compare_params(self, other_model: CebraOwn) -> bool:
        """
        Compare the parameters of the current model with another CebraOwn model.

        Parameters
        ----------
        other_model : CebraOwn
            The other model to compare with.

        Returns
        -------
        bool
            True if the parameters are equal, False otherwise.
        """
        self_params = self.get_params()
        self_params["data_filter"] = (
            self.data_filter.filters if self.data_filter else None
        )
        other_params = other_model.get_params()

        # ensure backcompatibility with old models
        if other_params["data_filter"] == None:
            other_params["data_filter"] = self_params["data_filter"]

        return equal_dicts(self_params, other_params, CebraOwn.unimportant_attributes)

    def load_fitted_model(self):
        fitted_model_path = self.save_path
        if fitted_model_path.exists():
            fitted_model = self.load(fitted_model_path)
            if self.compare_params(fitted_model):
                for unimportant_key in CebraOwn.unimportant_attributes:
                    setattr(
                        fitted_model, unimportant_key, getattr(self, unimportant_key)
                    )

                fitted_model.verbose = self.verbose
                # load_cebra_with_sklearn_backend
                # remove_none_attributes
                for key, value in copy.deepcopy(fitted_model.__dict__).items():
                    if value is None and key in self.__dict__.keys():
                        fitted_model.__dict__.pop(key, None)

                properties_to_copy = list(fitted_model.__dict__.keys())
                # remove data_filter from properties to copy
                if "data_filter" in properties_to_copy:
                    properties_to_copy.remove("data_filter")
                copy_attributes_to_object(
                    propertie_name_list=properties_to_copy,
                    set_object=self,
                    get_object=fitted_model,
                )
                global_logger.info(f"Loaded matching model {fitted_model_path}")
            else:
                global_logger.error(
                    f"Loaded model parameters do not match to initialized model. Not loading {fitted_model_path}"
                )

            self.fitted = self.is_fitted()
        else:
            self.fitted = False
        return self

    @property
    def labels_describe_space(self):
        """Check if the model is trained on behavior data that describes space."""
        return list_vars_in_list(
            self.behavior_data_types, Dataset.labels_describe_space
        )

    @property
    def relative_val_scale(self) -> np.ndarray[float]:
        """Get the relative model value scale for the given task model.

        Args:
            task_model (CebraOwn):

        Raises:
            ValueError: If more than one behavior data was used for training the model.

        Returns:
            Union[int, np.ndarray[float]]:
            The relative model value scale. If the model name contains "relative", it returns the maximum absolute values.
        """

        if "relative" in self.name:
            global_logger.warning(
                f"Detected relative in model name {self.name}. Converting relative performance to absolute using max possible value possible."
            )
            if len(self.behavior_data_types) > 1:
                do_critical(
                    ValueError,
                    f"More than one behavior data was used for training the model {self.name}. Please provide a single behavior data type.",
                )
            else:
                absolute_data = self.behavior_data_types[0]
            borders = Environment.define_border_by_pos(absolute_data)
            val_range = np.diff(borders).flatten()
            return val_range
        else:
            return 1

    def is_fitted(self):
        return sklearn_utils.check_fitted(self)

    def remove_fitting(self):
        if "n_features" in self.__dict__:
            del self.n_features_

    def train(
        self,
        neural_data,
        behavior_data=None,
        regenerate=False,
        verbose=False,
    ) -> CebraOwn:
        self.verbose = verbose
        # remove list if neural data is a list and only one element
        if isinstance(neural_data, list) and len(neural_data) == 1:
            neural_data = neural_data[0]
            behavior_data = behavior_data[0]

        if not self.fitted or regenerate:
            # skip if no neural data available
            if isinstance(neural_data, np.ndarray) and neural_data.shape[0] < 10:
                global_logger.error(
                    f"Not enough frames to use for {self.name}. At least 10 are needed. Skipping"
                )
                global_logger.debug(
                    f"End of ending of datasets was removed to ensure equal length."
                )
            else:
                # train model
                global_logger.info(f"Training  {self.name} model.")
                if self.type == "time":
                    self.fit(neural_data)
                else:
                    if behavior_data is None:
                        error_msg = f"No behavior data given for {self.type} model."
                        global_logger.critical(error_msg)
                        raise ValueError(
                            f"No behavior data types given for {self.type} model."
                        )
                    neural_data, behavior_data = force_equal_dimensions(
                        neural_data, behavior_data
                    )
                    self.fit(neural_data, behavior_data)
                self.fitted = self.is_fitted()
                self.save_procedure(self.save_path)
        else:
            global_logger.info(f"{self.name} model already trained. Skipping.")

        if behavior_data is not None and neural_data.shape[0] != behavior_data.shape[0]:
            global_logger.error(
                f"Neural data and behavior data have different number of frames. Neural data: {neural_data.shape[0]}, Behavior data: {behavior_data.shape[0]}."
            )
            global_logger.debug(
                f"End of ending of datasets was removed to ensure equal length."
            )
        return self

    def save_procedure(self, save_path: Union[str, Path]):
        """
        Save the model to the specified path.

        Parameters
        ----------
        save_path : str or Path, optional
            The path where the model should be saved (default is None).
            If None, the model will be saved to the default save path.
        """
        # convert important information from variable which is an object into dictionary replace it
        if (
            "data_filter" in self.__dict__
            and self.data_filter is not None
            and safe_isinstance(self.data_filter, DataFilter)
        ):
            backup_data_filter = copy.deepcopy(self.data_filter)
            self.data_filter = self.data_filter.filters
            self.save(save_path)
            self.data_filter = backup_data_filter
        else:
            self.save(save_path)

    def adapt(
        self,
        neural_data: np.ndarray,
        behavior_data: np.ndarray = None,
        to_name: Union[str, Path] = None,
        max_adapt_iterations: int = None,
        regenerate: bool = False,
    ) -> CebraOwn:
        """
        Adapt the model to new data.

        Parameters
        ----------
        neural_data : np.ndarray
            The neural data to adapt the model to.
        behavior_data : np.ndarray, optional
            The behavior data to adapt the model to (default is None).
        name : str, optional
            The name of the adapted model (default is None).
        regenerate : bool, optional
            Whether to regenerate the model (default is False).

        Returns
        -------
        CebraOwn
            The adapted model.
        """
        if max_adapt_iterations is not None:
            self.max_adapt_iterations = max_adapt_iterations
        adapted_model = copy.deepcopy(self)
        if to_name is not None:
            adapted_model.set_name(f"{self.name}-to-{to_name}")
            save = True
        else:
            adapted_model.set_name(f"{self.name}_adapted")
            save = False
            global_logger.info("No name given, model will not be saved.")
            global_logger.debug(
                f"Using default name {adapted_model.name} for adapted model."
            )

        if adapted_model.save_path.exists() and not regenerate:
            global_logger.info(
                f"Adapted model {adapted_model.name} already exists. Loading {adapted_model.save_path}"
            )
            adapted_model.load_fitted_model()
            adapt = False
        else:
            adapt = True

        if adapt:
            global_logger.info(f"Adapting model {self.name} to new data.")
            if neural_data.shape[0] != behavior_data.shape[0]:
                global_logger.error(
                    f"Neural data and behavior data have different number of frames. Neural data: {neural_data.shape[0]}, Behavior data: {behavior_data.shape[0]}."
                )
                global_logger.debug(f"Returning None.")
                return None

            adapted_model.fit(neural_data, behavior_data, adapt=True)
            if save:
                global_logger.info(
                    f"Saving adapted model {adapted_model.name} to {adapted_model.save_path}."
                )
                adapted_model.save(adapted_model.save_path)

        return adapted_model

    def create_embedding(
        self,
        session_id=None,
        to_transform_data=None,
        transform_data_labels=None,
        train_or_test="train",
        to_2d=False,
        save=False,
        return_labels=False,
        plot=False,
        markersize=2,
        additional_title="",
        as_pdf=False,
        save_dir=None,
    ):
        embedding = None
        labels = None
        if self.fitted:
            if to_transform_data is None:
                to_transform_data = self.get_data(
                    train_or_test=train_or_test, type="binarized"
                )
                embedding = self.get_data(train_or_test=train_or_test, type="embedding")
                label = self.get_data(train_or_test=train_or_test, type="behavior")
            else:
                label = transform_data_labels
                if label is None:
                    if plot:
                        global_logger.warning(
                            f"WARNING: Proper Plotting of transformed data only possible with provided labels."
                        )
                    label = np.zeros(to_transform_data.shape[0])

            if isinstance(to_transform_data, list) and len(to_transform_data) == 1:
                to_transform_data = to_transform_data[0]

            if isinstance(to_transform_data, np.ndarray):

                if embedding is None:
                    if session_id is not None:
                        # single session embedding from multi-session model
                        global_logger.debug(
                            f"Transforming data for session {session_id}."
                        )
                        embedding = (
                            self.transform(to_transform_data, session_id=session_id)
                            if to_transform_data.shape[0] > 10
                            else None
                        )
                    else:
                        embedding = (
                            self.transform(to_transform_data)
                            if to_transform_data.shape[0] > 10
                            else None
                        )

                if to_2d:
                    if embedding.shape[1] > 2:
                        embedding = sphere_to_plane(embedding)
                    elif embedding.shape[1] == 2:
                        print(f"Embedding is already 2D.")
                if save:
                    raise NotImplementedError("Saving embeddings not implemented yet.")
                    import pickle

                    with open("multi_embeddings.pkl", "wb") as f:
                        pickle.dump(embedding, f)

                if plot:
                    if embedding is None:
                        global_logger.error(
                            f"Embedding is None. Not plotting {self.name} model."
                        )
                    else:
                        plot_labels = {
                            "name": self.name,
                            "labels": label,
                        }
                        task_id = self.get_metadata("neural", "task_id")
                        plot_title = (
                            f"CEBRA Embedding: {task_id} {self.name} - {train_or_test}"
                        )
                        Vizualizer.plot_embedding(
                            embedding=embedding,
                            title=plot_title,
                            embedding_labels=plot_labels,
                            markersize=markersize,
                            additional_title=additional_title,
                            as_pdf=as_pdf,
                            save_dir=save_dir,
                            show=True,
                        )
            else:
                embedding = {}
                # multi-session embedding
                for i, data in enumerate(to_transform_data):
                    embedding_title = f"{self.name}_task_{i}"
                    if return_labels:
                        session_embedding, label = self.create_embedding(
                            self, i, data, to_2d, save, return_labels
                        )
                        if session_embedding is not None:
                            embedding[embedding_title] = session_embedding
                            labels[embedding_title] = label
                    else:
                        session_embedding = self.create_embedding(
                            self, i, data, to_2d, save
                        )
                        if session_embedding is not None:
                            embedding[embedding_title] = session_embedding

        else:
            global_logger.error(f"{self.name} model. Not fitted.")
            global_logger.debug(f"Skipping {self.name} model")

        if return_labels:
            labels = label if labels is None else labels
            return embedding, labels
        return embedding

    def get_loss(self):
        return self.state_dict_["loss"]

    def get_data(
        self,
        train_or_test: Literal["train", "test"] = "train",
        type: Literal[
            "fluorescence", "binarized", "embedding", "behavior"
        ] = "embedding",
    ) -> np.ndarray:
        """
        Get the data for the model.

        Parameters
        ----------
        train_or_test : str
            The type of data to get (either "train" or "test").
        type : str
            The type f data to get
            types:
                - "fluorescence": fluorescence data
                - "binarized": Binarized data
                - "embedding": Embedding data
                - "behavior": behavior data

        Returns
        -------
        np.ndarray
            The data for the model.
        """
        space = self.train_info["space"]
        shuffle = self.train_info["shuffle"]

        if type == "embedding":
            if space != type:
                global_logger.warning(
                    f"Data type {type} not matching with training data space {space}. Be cautious."
                )
            data = self.embedding[train_or_test]

        elif type in ["fluorescence", "binarized", "behavior"]:
            ids = (
                self.train_info["idx_train"]
                if train_or_test == "train"
                else self.train_info["idx_test"]
            )
            transformation = self.train_info["transformation"]

            objects = (
                self.neural_data_objects
                if type != "behavior"
                else self.behavior_data_objects
            )
            if len(objects) > 1:
                raise NotImplementedError(
                    "Loading data from multiple neural objects not implemented yet."
                )
                self.get_data_types("behavior")
            else:
                obj = next(iter(objects.values()))

            if type in ["fluorescence", "binarized"]:
                if space != type:
                    global_logger.warning(
                        f"Data type {type} not matching with training data space {space}. Be cautious."
                    )
                if type == "fluorescence":
                    all_data = obj.get_process_data(type="unprocessed")
                else:
                    all_data = obj.get_transdata(transformation=transformation)

            elif type == "behavior":
                # handle transformation
                all_data = obj.get_transdata(transformation=transformation)
                if shuffle:
                    all_data = Dataset.shuffle(all_data)

            data = Dataset.filter_by_idx(all_data, ids)
        else:
            raise ValueError(
                f"Unknown data type {type}. Please use 'fluorescence', 'binarized', 'embedding' or 'behavior'."
            )

        return data

    def set_train_info(
        self,
        idx_train: List[int],
        idx_test: List[int],
        space: Literal["fluorescence", "binarized"],
        transformation: str = None,
        movement_state: str = None,
        split_ratio: float = None,
        shuffle: bool = None,
    ) -> None:
        """
        Set the train information for the model.
        """
        information = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "space": space,
            "transformation": transformation,
            "movement_state": movement_state,
            "split_ratio": split_ratio,
            "shuffle": shuffle,
        }
        self.train_info = information

    def set_train_objects(
        self,
        neural_objects: List[NeuralDataset],
        behavior_objects: List[BehaviorDataset] = [],
    ):
        """
        Set the train objects for the model.
        """
        self.neural_data_objects = neural_objects
        self.behavior_data_objects = behavior_objects

    def get_data_types(self, type: str = "neural"):
        types = []
        objects = (
            self.neural_data_objects if type == "neural" else self.behavior_data_objects
        )
        for obj in objects:
            types.append(obj.asdf)
        return types

    def set_embedding(self, data: np.ndarray, train_or_test: str = "train"):
        self.embedding[train_or_test] = data

    def prepfanalysis(
        self,
        folder_name: str,
        ofname_start: str,
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        labels_needed: bool = False,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        additional_title: Optional[str] = "",
    ) -> Tuple[str, Path, np.ndarray, Optional[np.ndarray]]:
        """
        Prepares the analysis of the model.


        Parameters
        ----------
        folder_name : str
            Name of the folder to save the output file in.
        ofname_start : str
            Start of the output file name.
        use_raw : bool, optional
            Whether to use the raw data (default is False).
        labels : Union[np.ndarray, Dict[str, np.ndarray]], optional
            Labels for the data needed for determining Structure (default is None). If None, the labels from the model are used.
        to_transform_data : np.ndarray, optional
            Data to transform (default is None). If None, the data from the model is used.
        to_name : str, optional
            Name of the data to transform (default is None). If None, the data from the model is used.
        to_2d : bool, optional
            Whether to transform the data to 2D (default is False).
        additional_title : str, optional
            Additional title for the output file (default is None).

        Returns
        -------
        Tuple[str, Path, np.ndarray, Optional[np.ndarray]]
            The additional title, output file path, data and labels.
        """
        additional_title = additional_title or f"{self.save_path.stem}"
        ofname = f"{ofname_start}_{self.save_path.stem}"
        if space in ["fluorescence", "binarized"] and to_transform_data is not None:
            do_critical(
                ValueError,
                f"Cannot use {space} data and provide data to transform. Raw data is based on the model training data.",
            )
        elif space in ["fluorescence", "binarized"]:
            data = self.get_data(train_or_test="train", type=space)
            labels = self.get_data(train_or_test="train", type="behavior")
            additional_title += f" - {space}"
            ofname += f"_{space}"
        elif space == "embedding":
            if to_transform_data is None:
                return_labels = True if labels is None else True
                data = self.create_embedding(to_2d=to_2d, return_labels=return_labels)
                additional_title += " - Embedding"
                if return_labels:
                    labels = data[1]
                    data = data[0]
            else:
                if labels is None and labels_needed:
                    do_critical(
                        ValueError,
                        "No labels provided for data to transform but needed for analysis.",
                    )
                data = self.create_embedding(
                    to_transform_data=to_transform_data,
                    to_2d=to_2d,
                )
                additional_title += f" to {to_name}"
                ofname += f"_to-{to_name}"
        else:
            do_critical(
                ValueError,
                f"Unknown space {space}. Please use 'fluorescence', 'binarized' or 'embedding'.",
            )

        additional_title += f" {self.train_info['movement_state']}"
        # define output file name
        # check if parameter sweep is performed
        opath = self.save_path.parent.joinpath(folder_name, ofname + ".npz")
        return additional_title, opath, data, labels

    def structure_index(
        self,
        params: Dict[str, Union[int, bool, List[int]]],
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: bool = True,
        additional_title: str = None,
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ) -> Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]:
        """
        Computes the structural index for a Distribution.

        The structural index is a measure of the structure of the neural data.

        Parameters
        ----------
        params : Dict[str, Union[int, bool, List[int]]]
            Parameters for the structural index calculation. If n_neighbors is a list, a parameter sweep is performed.
            Keywords:
                "n_bins": 3, - number of bins for the histogram
                "n_neighbors": 15, - number of neighbors for the KNN algorithm
                "discrete_label": False, - whether the labels are discrete or continuous
                "num_shuffles": 0, - number of shuffles to perform
                "verbose": True, -  whether to print verbose output
        use_raw : bool, optional
            Whether to use the raw data (default is False).
        labels : Union[np.ndarray, Dict[str, np.ndarray]], optional
            Labels for the data needed for determining Structure (default is None). If None, the labels from the model are used.
        to_transform_data : np.ndarray, optional
            Data to transform (default is None). If None, the data from the model is used.
        to_name : str, optional
            Name of the data to transform (default is None). If None, the data from the model is used.
        regenerate : bool, optional
            Whether to regenerate the structural index (default is False).
        to_2d : bool, optional
            Whether to transform the data to 2D (default is False).
        plot : bool, optional
            Whether to plot the structural index (default is True).

        Returns:
        -------
        Dict[str, Dict[str, Dict[str, Union[float, np.ndarray]]]]
            The structural index for the data.
            The structure index is a measure of the structure of the neural data.
            The structure index is a dictionary with the following keys:
                - "mean": mean structure index
                - "variance": variance structure index
                - "distribution": distribution of the structure index
                - "n_neighbors": number of neighbors used for the KNN algorithm
                - "n_bins": number of bins used for the histogram
        """

        additional_title, opath, data, labels = self.prepfanalysis(
            folder_name="structure_index",
            ofname_start="structural_indices",
            space=space,
            labels=labels,
            labels_needed=True,
            to_transform_data=to_transform_data,
            to_name=to_name,
            to_2d=to_2d,
            additional_title=additional_title,
        )
        if data.shape[0] != labels.shape[0]:
            global_logger.error(
                f"Neural data and behavior data have different number of frames. Neural data: {data.shape[0]}, Behavior data: {labels.shape[0]}. Returning None"
            )
            return None

        struc_ind = structure_index(
            data=data,
            labels=labels,
            params=params,
            additional_title=additional_title,
            save_path=opath,
            plot=plot,
            plot_save_dir=plot_save_dir
            or self.save_path.parent.parent.joinpath("figures"),
            as_pdf=as_pdf,
            regenerate=regenerate,
        )
        return struc_ind

    def feature_similarity(
        self,
        metric: str = "cosine",
        similarity: Literal["pairwise", "inside", "outside"] = "inside",
        out_det_method: Literal["density", "contamination"] = "density",
        remove_outliers: bool = True,
        parallel: bool = True,
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        labels: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        to_transform_data: Optional[np.ndarray] = None,
        to_name: str = None,
        to_2d: bool = False,
        regenerate: bool = False,
        plot: bool = True,
        additional_title: str = None,
        as_pdf: bool = False,
        plot_save_dir: Optional[Path] = None,
    ):
        """
        Computes the feature similarity for a Distribution.

        The feature similarity is a measure of the similarity of population vectors to each other, inside groups or between groups.

        Parameters
        ----------
        metrics:
            euclidean, wasserstein, kolmogorov-smirnov, chi2, kullback-leibler, jensen-shannon, energy, mahalanobis, cosine
            the compare_distributions is also removing outliers on default.

        category map:
            maps discrete labels to multi dimensional position vectors

        """
        additional_title, opath, data, labels = self.prepfanalysis(
            folder_name="feature_similarity",
            ofname_start="feature_similarity",
            space=space,
            labels=labels,
            labels_needed=True,
            to_transform_data=to_transform_data,
            to_name=to_name,
            to_2d=to_2d,
            additional_title=additional_title,
        )

        if len(self.behavior_data_objects) > 1:
            do_critical(
                ValueError,
                "Feature similarity not implemented for multiple behavior data objects.",
            )
        else:
            behavior_data_object = next(iter((self.behavior_data_objects.values())))

        bin_size, min_bins, max_bins = (
            behavior_data_object.define_binsize_minbins_maxbins()
        )
        binned_labels = bin_array(
            labels, bin_size=bin_size, min_bin=min_bins, max_bin=max_bins
        )
        category_map = behavior_data_object.category_map
        encoded_labels, _ = encode_categorical(binned_labels, category_map=category_map)

        fsim = feature_similarity(
            data=data,
            labels=encoded_labels,
            category_map=category_map,
            max_bin=behavior_data_object.max_bin,
            metric=metric,
            out_det_method=out_det_method,
            remove_outliers=remove_outliers,
            parallel=parallel,
            similarity=similarity,
            additional_title=additional_title,
            save_path=opath,
            plot=plot,
            plot_save_dir=plot_save_dir
            or self.save_path.parent.parent.joinpath("figures"),
            as_pdf=as_pdf,
            regenerate=regenerate,
        )
        return fsim

    def tuning_map(
        self,
        train_or_test: Literal["train", "test"] = "train",
        space: Literal["fluorescence", "binarized", "embedding"] = "binarized",
        behavior_data_type: str = "position",
        plot: bool = False,
        cell_indices: Optional[np.ndarray] = None,
    ):
        """
        Computes the tuning map for the model.
        The tuning map is a measure of the tuning of the neural data to the behavior data.

        Parameters
        ----------
        train_or_test : Literal["train", "test"]
            The type of data to use for the tuning map (default is "train").
        space : Literal["fluorescence", "binarized", "embedding"]
            The type of data to use for the tuning map (default is "binarized").
        behavior_data_type : str
            The type of behavior data to use for the tuning map (default is "position").
        cell_indices : Optional[np.ndarray], optional
            Boolean or index array to filter specific cells. If None, all cells are included. Default is None.


        Returns
        -------
        np.ndarray
            The tuning map for the model.
        """
        all_activity = self.get_data(train_or_test=train_or_test, type=space)
        activity = (
            all_activity[:, cell_indices] if cell_indices is not None else all_activity
        )

        beh_obj = self.behavior_data_objects[behavior_data_type]
        train_idx = self.train_info["idx_train"]
        filtered_binned_pos = self.data_filter.filter(
            beh_obj.unfiltered_binned_data, behavior_data_type
        )[train_idx]
        rate_map, time_map = PlaceCellDetectors.get_rate_map(
            activity=activity,
            binned_pos=filtered_binned_pos,
            smooth=True,
            window_size=2,
            max_bin=beh_obj.max_bin,
            category_map=beh_obj.category_map,
            fps=self.get_metadata("neural", "fps"),
            plot=plot,
        )
        tuning_map = rate_map.reshape(rate_map.shape[0], -1)
        return tuning_map

    def to_df(
        self,
        train_or_test: Literal["train", "test"] = "train",
        space: Literal["embedding", "binarized", "fluorescence"] = "embedding",
        behavior_data_type: str = "position",
        return_activity_type: Literal["activity", "tuning", "none"] = "activity",
        cell_indices: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Construct a pandas DataFrame summarizing session-level metadata and activity data.

        Parameters
        ----------
        train_or_test : Literal["train", "test"], optional
            Dataset partition to retrieve data from; valid options are 'train' or 'test'. Default is 'train'.
        type : Literal["embedding", "binarized", "fluorescence"], optional
            Data representation type to retrieve. Options are 'embedding', 'binarized', or 'fluorescence'. Default is 'embedding'.
        behavior_data_type : str, optional
            The type of behavioral data used for constructing the tuning map. Only relevant if `get_tuning_map` is True. Default is 'position'.
        return_activity_type : Literal["activity", "tuning", "none"], optional
            Specifies whether to return standard activity data or tuning map.
            If cell_filter is provided, the tuning map will be based on the filtered cells.
        cell_filter : Optional[np.ndarray], optional
            Boolean or index array to filter specific cells. If None, all cells are included. Default is None.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing one row with the following columns:
                - 'animal': Animal identifier.
                - 'date': Session date as a formatted string.
                - 'condition': Experimental or behavioral condition.
                - 'task': Full task string identifier.
                - 'task_name': Task name parsed from the task identifier.
                - 'task_number': Task number parsed from the task identifier.
                - 'model': Model identifier.
                - 'id': Unique session identifier.
                - 'activity': Activity data or tuning map (as a single-item list).

        Notes
        -----
        - The function automatically parses the task identifier to extract task name and task number using regular expressions.
        - If `get_tuning_map` is True, the activity data is derived from the tuning map specific to the indicated behavioral data type and representation.
        - All metadata fields are extracted using internal methods for consistency.
        - The output DataFrame is structured for easy concatenation across sessions or animals.
        """

        task = self.id("task")
        # Extract task prefix and number
        # automatically detect number in task name independent of name using regex check for number in front or back of task name
        task_name, task_number = search_split(r"\d+", task)

        match return_activity_type:
            case "tuning":
                activity = self.tuning_map(
                    train_or_test=train_or_test,
                    space=space,
                    behavior_data_type=behavior_data_type,
                    cell_indices=cell_indices,
                )
            case "activity":
                all_activity = self.get_data(train_or_test=train_or_test, type=space)
                activity = (
                    all_activity[:, cell_indices]
                    if cell_indices is not None
                    else all_activity
                )
            case "none":
                activity = None
            case _:
                raise ValueError(
                    f"Unknown return_activity_type {return_activity_type}. Please use 'activity', 'tuning' or 'none'."
                )
        condition = self.get_metadata("animal", "condition")
        df = pd.DataFrame(
            {
                "animal": self.id("animal"),
                "date": num_to_date(int(self.id("date"))),
                "condition": condition,
                "task": task,
                "task_name": task_name,
                "task_number": int(task_number),
                "model": self.id("model"),
                "id": self.id(),
                "activity": [activity],
                "cell_indices": [cell_indices] if cell_indices is not None else None,
            }
        )
        return df


def _load_cebra_with_sklearn_backend(cebra_info: Dict) -> "CebraOwn":
    """Loads a CEBRA model with a Sklearn backend.

    Args:
        cebra_info: A dictionary containing information about the CEBRA object,
            including the arguments, the state of the object and the state
            dictionary of the model.

    Returns:
        The loaded CEBRA object.

    Raises:
        ValueError: If the loaded CEBRA model was not already fit, indicating that loading it is not supported.
    """
    required_keys = ["args", "state", "state_dict"]
    missing_keys = [key for key in required_keys if key not in cebra_info]
    if missing_keys:
        raise ValueError(
            f"Missing keys in data dictionary: {', '.join(missing_keys)}. "
            f"You can try loading the CEBRA model with the torch backend."
        )

    args, state, state_dict = (
        cebra_info["args"],
        cebra_info["state"],
        cebra_info["state_dict"],
    )
    cebra_ = CebraOwn(**args)

    for key, value in state.items():
        setattr(cebra_, key, value)

    state_and_args = {**args, **state}

    if not sklearn_utils.check_fitted(cebra_):
        raise ValueError(
            "CEBRA model was not already fit. Loading it is not supported."
        )

    if cebra_.num_sessions_ is None:
        model = cebra.models.init(
            args["model_architecture"],
            num_neurons=state["n_features_in_"],
            num_units=args["num_hidden_units"],
            num_output=args["output_dimension"],
        ).to(state["device_"])

    elif isinstance(cebra_.num_sessions_, int):
        model = nn.ModuleList(
            [
                cebra.models.init(
                    args["model_architecture"],
                    num_neurons=n_features,
                    num_units=args["num_hidden_units"],
                    num_output=args["output_dimension"],
                )
                for n_features in state["n_features_in_"]
            ]
        ).to(state["device_"])

    criterion = cebra_._prepare_criterion()
    criterion.to(state["device_"])

    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), criterion.parameters()),
        lr=args["learning_rate"],
        **dict(args["optimizer_kwargs"]),
    )

    solver = cebra.solver.init(
        state["solver_name_"],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        tqdm_on=args["verbose"],
    )
    solver.load_state_dict(state_dict)
    solver.to(state["device_"])

    cebra_.model_ = model
    cebra_.solver_ = solver

    return cebra_


def _check_type_checkpoint(checkpoint):
    if not isinstance(checkpoint, CebraOwn):
        raise RuntimeError(
            "Model loaded from file is not compatible with "
            "the current CEBRA version."
        )
    if not sklearn_utils.check_fitted(checkpoint):
        raise ValueError("CEBRA model is not fitted. Loading it is not supported.")

    return checkpoint
