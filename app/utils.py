from json import load, dump
from typing import Iterator, Iterable
from os import PathLike
from csv import reader

type Path = PathLike | str


def tskv_reader(path: Path) -> Iterator[dict[str, str]]:
    with open(path, 'r', encoding='utf-8') as file:
        for row in reader(file, delimiter='\t'):
            yield {(kv := o.split('=', 1))[0]: kv[1] for o in row}


def merge_iterators(iterators: Iterable[Iterator]) -> Iterator:
    for it in iterators:
        for obj in it:
            yield obj


def json_read(path: Path):
    with open(path, 'r') as file:
        return load(file)


def json_write(obj, path: Path):
    with open(path, 'w') as file:
        return dump(obj, file)
