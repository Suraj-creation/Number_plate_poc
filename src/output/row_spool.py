from __future__ import annotations

import csv
import json
import re
from collections import Counter
from tempfile import TemporaryFile


class RowSpool:
    """Replay observations from disk while retaining only a bounded preview."""

    def __init__(self, preview_rows: int = 2000):
        self.stream = TemporaryFile(mode="w+t", encoding="utf-8")
        self.preview_rows = preview_rows
        self.preview = []
        self.fields = dict.fromkeys(("source", "frame_index", "timestamp_ms", "recognition_status"))
        self.counts = Counter()
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()

    def add(self, rows):
        for row in rows:
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            self.fields.update(dict.fromkeys(row))
            self.count += 1
            self.counts[row.get("recognition_status")] += 1
            if len(self.preview) < self.preview_rows:
                self.preview.append(row)

    def __iter__(self):
        self.stream.seek(0)
        for line in self.stream:
            yield json.loads(line)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str) and re.match(r"^\s*[=+@-]|^[\t\r\n]", value):
        return "'" + value
    return value
