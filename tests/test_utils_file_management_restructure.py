from pathlib import Path
from typing import TYPE_CHECKING

from neural_analysis.utils.file_management.restructure import (
    move_file_to_folder,
    move_task_into_date_folder,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_move_file_to_folder(tmp_path: Path):
    source_file = tmp_path / "test.txt"
    source_file.write_text("hello")
    target_dir = tmp_path / "target"

    move_file_to_folder(source_file, target_dir)

    assert (target_dir / "test.txt").exists()
    assert not source_file.exists()


def test_move_task_into_date_folder(tmp_path: Path):
    # Setup /animal/task/file_with_date
    animal_dir = tmp_path / "DON-123456"
    task_dir = animal_dir / "Task1"
    task_dir.mkdir(parents=True)

    data_file = task_dir / "data_20210101_file.txt"
    data_file.write_text("hello")

    move_task_into_date_folder(task_dir)

    # Should move to /animal/20210101/Task1
    expected_task_dir = animal_dir / "20210101" / "Task1"
    assert expected_task_dir.exists()
    assert (expected_task_dir / "data_20210101_file.txt").exists()
    assert not task_dir.exists()
