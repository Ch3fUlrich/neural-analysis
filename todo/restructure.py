"""
Data Restructuring Module

This module reorganizes experimental data directories to match the SERBRA
pipeline's expected structure.

The SERBRA pipeline requires a specific folder structure to automatically load
data. This module helps reorganize existing data into the required format:

    DON-XXXXXX/              (Animal ID)
      └── YYYYMMDD/          (Session date)
          └── TaskName/      (Task identifier)
              ├── 002P-F/    (Neural data - 2P Femtonics)
              │   └── tif/
              │       └── suite2p/
              │           └── plane0/
              └── TR-BSL/    (Behavioral data - Open field)

Key Functions:
    restructure_animal_dirs: Main function to restructure multiple animal directories
    restructure_animal_dir: Restructures a single animal's directory
    restructure_date_dir: Organizes data within a date-specific folder
    restructure_task_dir: Organizes task-specific data files
    move_task_into_date_folder: Moves task folders into date-organized structure
    move_file_to_folder: Moves a single file to a target folder

Naming Structures:
    The module enforces specific naming conventions for folders:
    - Animals: DON-XXXXXX format (6-8 digits)
    - Dates: YYYYMMDD format
    - Photon recordings: 001P-I (1P Inscopix), 002P-F (2P Femtonics), 002P-T (2P Thorlabs)
    - Behavioral setups: TRD-2P (treadmill), TR-BSL (open field), etc.

Workflow:
    1. Identify animal directories based on naming convention
    2. Within each animal, identify or create date folders
    3. Within each date, organize task folders
    4. Within each task, move files to appropriate output folders (neural/behavioral)

Dependencies:
    - pathlib: Path manipulation
    - numpy: Array operations
    - tqdm: Progress bars
    - Helper: Custom helper functions for file operations

Example:
    >>> from restructure import restructure_animal_dirs
    >>>
    >>> # Restructure all animals in a directory
    >>> restructure_animal_dirs(
    ...     path="/path/to/data",
    ...     photon_type="2p",              # "1p" or "2p"
    ...     rec_output="femtonics",        # "femtonics", "thorlabs", or "inscopix"
    ...     behavior_rec_type="openfield", # "openfield", "wheel", "cam"
    ...     neural_related=".*002P-F.*",   # regex for neural files
    ...     location_related=".*TR-BSL.*"  # regex for behavioral files
    ... )

Notes:
    - Always backup data before restructuring
    - File naming must contain animal ID, date, and setup information
    - The module creates folders as needed but does not delete existing ones
"""

from typing import Union, Dict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from Helper import (
    init_path_checks,
    extract_date_from_filename,
    num_to_date,
    global_logger,
    search_filedir,
    regex_search,
)
import shutil

# Naming patterns for different folder types
naming_structure = {
    "animal": r"^DON-\d{6,8}$",  # Animal IDs: DON-XXXXXX
    "date": r"^\d{8}$",  # Dates: YYYYMMDD
    "photon_rec": r"^00[123]P-[IFT]$",  # Photon recordings: 001P-I, 002P-F, etc.
    "rec_output": r"^-[IFT]$",  # Recording outputs: -I, -F, -T
}

behavior_naming_structure = {
    "wheel": r"^TRD-2P$",  # Treadmill/wheel setup
}

cam_naming_structure = {
    "vr": "0000VR",  # VR camera
    "cam": "0000CM",  # Standard camera
    "top": "0000BSM",  # Top-view camera
    "openfield": "TR-BSL",  # Open field behavioral setup
}

rec_outputs = {
    "inscopix": "-I",  # Inscopix 1P imaging
    "femtonics": "-F",  # Femtonics 2P imaging
    "thorlabs": "-T",  # Thorlabs 2P imaging
}

photon_types = {
    "1p": "001P",  # 1-photon imaging
    "2p": "002P",  # 2-photon imaging
}

# Analysis output folders (should not be processed as data folders)
analysis_outputs = [
    "Bayesian_decoder",
    "Opexebo_cell_analysis",
    "models",
    "figures",
    "output",
]

# Combine all neural, behavioral, and camera folder names
neural_output_folder_names = [
    photon_type + rec_output
    for photon_type in photon_types.values()
    for rec_output in rec_outputs.values()
]
behavior_output_folder_names = [
    behavior_naming_structure[behavior] for behavior in behavior_naming_structure
]
cam_output_folder_names = [cam_naming_structure[cam] for cam in cam_naming_structure]

# Folders that should not be treated as data folders
forbidden_names = (
    neural_output_folder_names
    + behavior_output_folder_names
    + cam_output_folder_names
    + analysis_outputs
)


def move_file_to_folder(fname: Union[str, Path], folder: Union[str, Path]) -> None:
    """Move a file to a specified folder, creating the folder if needed.

    Parameters:
    ----------
        fname: str or Path
            Path to the file to move

        folder: str or Path
            Destination folder path

    Returns:
    -------
        None

    Notes:
    -----
        Creates the destination folder if it doesn't exist and logs the move operation.
    """
    fname = Path(fname)
    folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    new_path = folder / fname.name
    fname.rename(new_path)
    global_logger.info(f"Moved {fname} to {new_path}")


def move_task_into_date_folder(task_dir: Union[str, Path], date: str = None) -> None:
    """Move a task directory into a date-organized folder structure.

    Searches for date information in filenames within the task directory,
    creates a date folder in the parent directory, and moves the task into it.

    Parameters:
    ----------
        task_dir: str or Path
            Path to the task directory to move

        date: str, optional
            Date in YYYYMMDD format. If None, will be extracted from filenames.

    Returns:
    -------
        None

    Raises:
    ------
        ValueError: If no date is found or multiple dates are found in filenames

    Notes:
    -----
        This ensures task folders are organized under date folders as required
        by the SERBRA pipeline structure.
    """
    task_dir = init_path_checks(task_dir, check="dir")

    # Search for date inside file names
    file_names = [f.name for f in task_dir.iterdir() if f.is_file()]
    if date is None:
        dates = np.unique(
            [
                extract_date_from_filename(fname)
                for fname in file_names
                if extract_date_from_filename(fname)
            ]
        )
        if len(dates) == 0:
            raise ValueError(f"No date found in the task directory {task_dir}.")
        elif len(dates) > 1:
            raise ValueError(
                f"Multiple dates found in the task directory {task_dir}: {dates}."
            )
        else:
            date = dates[0]

    # Create a date folder in parent directory and move task folder into it
    date_folder = task_dir.parent / date
    date_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task_dir), str(date_folder))


def restructure_task_dir(
    task_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure a task directory to match the SERBRA format.

    Organizes files within a task directory by moving them into appropriate
    subfolders (neural data, behavioral data, etc.) based on regex patterns.

    Parameters:
    ----------
        task_dir: str or Path
            Path to the task directory

        task_data_locations: dict
            Dictionary mapping folder names to regex patterns for identifying files.
            Keys are target folder names (e.g., "002P-F", "TR-BSL")
            Values are regex patterns to match files (with format string placeholders)

    Returns:
    -------
        None

    Example:
    -------
        >>> task_data_locations = {
        ...     "002P-F": ".*{animal_id}.*{date}.*002P-F.*",
        ...     "TR-BSL": ".*{animal_id}.*{date}.*TR-BSL.*"
        ... }
        >>> restructure_task_dir("/path/to/DON-123456/20210101/Task1", task_data_locations)

    Notes:
    -----
        - Creates target folders if they don't exist
        - Files matching patterns are moved to corresponding folders
        - Unmatched files remain in the task directory
    """
    task_dir = init_path_checks(task_dir, check="dir")

    # Get the list of files in the task directory
    files = search_filedir(
        path=task_dir,
        type="file",
    )
    if len(files) == 0:
        print(f"No files found in the task directory {task_dir}.")
        return

    # Extract metadata from directory structure
    task_name = task_dir.name
    date = task_dir.parent.name
    animal_id = task_dir.parent.parent.name

    # Move files to appropriate folders based on regex patterns
    for folder_name, category in task_data_locations.items():
        folder_path = task_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Format the regex pattern with actual metadata values
        formated_category = category.format(
            animal_id=animal_id, date=date, behavior_setup=folder_name, task=task_name
        )

        # Find files matching the pattern and move them
        files_to_move = regex_search(files, formated_category)
        for file in files_to_move:
            move_file_to_folder(file, folder_path)


def restructure_date_dir(
    date_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure the date directory to match the CEBRA format.

    Parameters
    ----------
        date_dir (Union[str, Path]): Path to the date directory.
        task_data_locations (Dict[str, str]): Dictionary containing the task data locations and their corresponding regex patterns.
    """
    date_dir = init_path_checks(date_dir, check="dir")
    # get the list of task folders in the date directory
    folders = search_filedir(
        path=date_dir,
        exclude_regex=forbidden_names,
        type="dir",
    )
    for task_folder in folders:
        restructure_task_dir(
            task_dir=task_folder,
            task_data_locations=task_data_locations,
        )


def restructure_animal_dir(
    animal_dir: Union[str, Path],
    task_data_locations: Dict[str, str],
) -> None:
    """Restructure the animal directory to match the CEBRA format.

    Parameters
    ----------
        animal_dir (Union[str, Path]): Path to the animal directory.
        task_data_locations (Dict[str, str]): Dictionary containing the task data locations and their corresponding regex patterns.

    """
    animal_dir = init_path_checks(animal_dir, check="dir")

    # check for task folders and if found, move them into date folders
    task_folders = search_filedir(
        path=animal_dir, type="dir", exclude_regex=forbidden_names
    )
    if len(task_folders) > 0:
        for task_folder in task_folders:
            move_task_into_date_folder(task_folder)

    # get the list of folders in the directory based on animal naming structure
    folders = search_filedir(
        path=animal_dir,
        include_regex=naming_structure["date"],
        exclude_regex=None,
        type="dir",
    )

    filtered_folders = []
    if len(folders) > 0:
        # checks if foldernames are real dates
        for date_folder in folders:
            date = num_to_date(date_folder.name)
            if date is None:
                global_logger.warning(
                    f"Folder {date_folder} in {animal_dir} is not a valid date. Skipping."
                )
            else:
                filtered_folders.append(date_folder)
        folders = filtered_folders

    if len(folders) == 0:
        raise ValueError(f"No date folders found in the animal directory {animal_dir}.")

    for date_folder in folders:
        restructure_date_dir(
            date_dir=date_folder,
            task_data_locations=task_data_locations,
        )


def restructure_animal_dirs(
    path: Union[str, Path],
    photon_type: str,
    rec_output: str,
    behavior_rec_type: str,
    neural_related: str,
    location_related: str,
) -> None:
    neural_rec_output_folder = photon_types[photon_type] + rec_outputs[rec_output]
    behavior_rec_output_folder = cam_naming_structure[behavior_rec_type]
    # get list of folders in the directory based on animal naming structure
    folders = search_filedir(
        path=path,
        include_regex=naming_structure["animal"],
        exclude_regex=None,
        type="dir",
    )
    task_data_locations = {
        neural_rec_output_folder: neural_related,
        behavior_rec_output_folder: location_related,
    }
    # iterate through each animal directory
    for animal_dir in tqdm(folders, desc="Restructuring animal directories"):
        restructure_animal_dir(animal_dir, task_data_locations)
