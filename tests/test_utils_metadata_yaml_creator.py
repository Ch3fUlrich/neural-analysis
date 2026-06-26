from neural_analysis.utils.metadata.yaml_creator import (
    create_dict,
    remove_none_from_dict,
)


def test_create_dict():
    d = create_dict(a="val", b="n/a", c="?", d="")
    assert d == {"a": "val", "b": None, "c": None, "d": None}

def test_remove_none_from_dict():
    d = {"a": 1, "b": None, "c": {"d": None, "e": 2}}

    # Non-recursive
    clean_d = remove_none_from_dict(d)
    assert clean_d == {"a": 1, "c": {"d": None, "e": 2}}

    # Recursive
    clean_d_rec = remove_none_from_dict(d, recursive=True)
    assert clean_d_rec == {"a": 1, "c": {"e": 2}}
