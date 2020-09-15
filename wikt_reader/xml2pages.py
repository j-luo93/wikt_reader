from argparse import ArgumentParser
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger
from tqdm import tqdm

from lxml import cssselect, etree


def cache(out_path_name: str, load_func):

    def decorator(func: Callable):

        @wraps(func)
        def wrapped(*args, **kwargs):
            sign = signature(func)
            out_path_str = sign.bind(*args, **kwargs).arguments[out_path_name]
            out_path = Path(out_path_str)

            if out_path.exists():
                logger.info(f'{out_path} already exists, directly loading from it.')
                return load_func(out_path_str)

            return func(*args, **kwargs)

        return wrapped

    return decorator


@cache('out_path', pd.read_pickle)
def read_xml(in_path: str, out_path: str):
    """Read xml file using `lxml.etree.parse`. Wiktionary dump contains special namespaces, so they are used to extract the pages. See https://stackoverflow.com/questions/6860013/xhtml-namespace-issues-with-cssselect-in-lxml."""
    logger.info(f'Loading xml from {in_path}.')
    xml_file = etree.parse(open(in_path, 'rb'))
    nsmap = xml_file.getroot().nsmap.copy()

    # CSS selector cannot process no-prefix namspace.
    nsmap['xhtml'] = nsmap[None]
    del nsmap[None]

    def get_selector(element: str):
        return cssselect.CSSSelector(f'xhtml|{element}', namespaces=nsmap)

    logger.info('Extracting pages.')
    page_selector = get_selector('page')
    pages = page_selector(xml_file)
    logger.info(f'Extracted {len(pages)} in total.')

    title_selector = get_selector('title')
    text_selector = get_selector('text')

    logger.info('Extracting titles and texts from pages.')
    records = list()
    for page in tqdm(pages):
        title = title_selector(page)[0].text
        text = text_selector(page)[0].text
        records.append({'title': title, 'text': text})
    df = pd.DataFrame(records)
    df.to_pickle(out_path)
    logger.info(f'Pages saved to {out_path}.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('xml_path', type=str, help='Path to the xml file.')
    parser.add_argument('out_name', type=str, help='Name for the output file without suffix.')
    args = parser.parse_args()
    if '.' in args.out_name:
        raise ValueError(f'{args.out_name} should not include "." or suffixes.')

    out_path_str = f'processed/{args.out_name}.pkl'
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    read_xml(args.xml_path, out_path_str)
