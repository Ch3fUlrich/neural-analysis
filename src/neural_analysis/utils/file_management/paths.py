import os
import re
from pathlib import Path
from typing import Union, List

def init_path_checks(path: Union[str, Path], check: str = "dir") -> Path:
    p = Path(path)
    if check == "dir" and not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    if check == "file" and not p.is_file():
        raise ValueError(f"{path} is not a valid file.")
    return p

def search_filedir(path: Union[str, Path], type: str = "file", include_regex: str = None, exclude_regex: tuple = None) -> list:
    p = Path(path)
    if not p.exists():
        return []
    results = []
    if type == "dir":
        for item in p.iterdir():
            if item.is_dir():
                if include_regex and not re.search(include_regex, item.name):
                    continue
                if exclude_regex and item.name in exclude_regex:
                    continue
                results.append(item)
    elif type == "file":
        for item in p.iterdir():
            if item.is_file():
                results.append(item)
    return results

def regex_search(paths: list, pattern: str) -> list:
    return [p for p in paths if re.search(pattern, p.name)]

def get_directories(path: str, regex_search: str = None) -> List[str]:
    if not os.path.exists(path):
        return []
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if regex_search:
        dirs = [d for d in dirs if re.search(regex_search, d)]
    return dirs

def get_files(path: str, ending: str = None, regex_search: str = None) -> List[str]:
    if not os.path.exists(path):
        return []
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if ending:
        files = [f for f in files if f.endswith(ending)]
    if regex_search:
        files = [f for f in files if re.search(regex_search, f)]
    return files
