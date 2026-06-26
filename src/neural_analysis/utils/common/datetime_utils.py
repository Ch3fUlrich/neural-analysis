import datetime
import re


def extract_date_from_filename(fname: str) -> str | None:
    match = re.search(r"(\d{8})", fname)
    return match.group(1) if match else None

def num_to_date(date_string: str) -> str | None:
    try:
        datetime.datetime.strptime(date_string, '%Y%m%d')
        return date_string
    except ValueError:
        return None
