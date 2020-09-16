from __future__ import annotations

import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable, List

import pandas as pd
from loguru import logger
from lxml import cssselect, etree
from tqdm import tqdm


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
def read_xml(in_path: str, out_path: str) -> pd.DataFrame:
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
    ns_selector = get_selector('ns')

    logger.info('Extracting titles and texts from pages.')
    records = list()
    for page in tqdm(pages):
        ns = int(ns_selector(page)[0].text)
        # Only normal pages are preserved, excluding special pages.
        if ns == 0:
            title = title_selector(page)[0].text
            text = text_selector(page)[0].text
            records.append({'title': title, 'text': text})
    df = pd.DataFrame(records)
    logger.info(f'Extracted {len(df)} dictionary entries.')
    df = df.dropna()
    logger.info(f'Empty pages dropped, ending up with {len(df)} pages.')
    df.to_pickle(out_path)
    logger.info(f'Pages saved to {out_path}.')
    return df


header_pat = re.compile(r'^==+([^=]+?)==+$')


@dataclass
class Node:
    depth: int
    header: str
    text: str
    children: List[Node] = field(default_factory=list)

    def to_element(self) -> etree.Element:
        """Convert node to an element with proper sub-elements."""
        sub_elements = [child.to_element() for child in self.children]
        return E('section', self.text, *sub_elements, depth=str(self.depth), header=self.header, )


def get_page_as_xml(text: str) -> str:
    root = Node(0, 'ROOT', '')
    stack: List[Node] = [root]
    for line in text.split('\n'):
        # Skip empty lines.
        if line:
            match = header_pat.search(line)
            # Deal with headers.
            if match:
                # Get number of equation signs, i.e., depth.
                num_eq = 0
                while line[num_eq] == '=':
                    num_eq += 1

                # Keep closing the brackets if current depth is greater than or equal to the last one on the stack.
                while stack and num_eq <= stack[-1].depth:
                    node = stack.pop()

                # Start the new bracket.
                parent_node = stack[-1]
                new_node = Node(num_eq, match.group(1), '')
                stack.append(new_node)
                parent_node.children.append(new_node)
            else:
                stack[-1].text += line + '\n'
    xml = etree.tostring(root.to_element(), encoding=str)
    return xml


@cache('out_path', pd.read_pickle)
def build_trees(raw_pages: Optional[pd.DataFrame], out_path: str) -> pd.Series:
    xmls = raw_pages['text'].progress_apply(get_page_as_xml)
    page_xmls = pd.DataFrame({'title': raw_pages['title'], 'text': xmls})
    page_xmls.to_pickle(out_path)
    logger.info(f'Page trees saved to {out_path}.')
    return page_xmls


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('in_path', type=str, help='Path to the xml file.')
    parser.add_argument('out_name', type=str, help='Name for the output file without suffix.')
    args = parser.parse_args()
    if '.' in args.out_name:
        raise ValueError(f'{args.out_name} should not include "." or suffixes.')

    tqdm.pandas()

    # Create folder for caching/storing processed files.
    folder = Path('./processed')
    folder.mkdir(parents=True, exist_ok=True)

    # # Get raw pages first.
    # raw_path = f'{folder}/{args.out_name}.raw.pkl'
    # raw_pages = read_xml(args.in_path, raw_path)

    # Build trees out of texts.
    xml_path = f'{folder}/{args.out_name}.xml.pkl'
    # page_xmls = build_trees(raw_pages, xml_path)
    page_xmls = build_trees(None, xml_path)

    # page_tree = raw_pages['text'].progress_apply(get_page_as_tree)
