from __future__ import annotations

import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable, List, Optional

import pandas as pd
from loguru import logger
from lxml import cssselect, etree
from lxml.builder import E
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
def read_xml(in_path: str, out_path: str, namespaces=(0, 118)) -> pd.DataFrame:
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
        # By default, only normal pages and reconstruction pages are preserved, excluding special pages.
        if ns in namespaces:
            title = title_selector(page)[0].text
            text = text_selector(page)[0].text
            records.append({'title': title, 'namespace': ns, 'text': text})
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
        return E('section', self.text, *sub_elements, depth=str(self.depth), header=self.header)


def get_page_as_xml(text: str) -> str:
    root = Node(0, 'ROOT', '')
    stack: List[Node] = [root]
    for line in text.split('\n'):
        # Strip whitespaces first.
        line = line.strip()
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
def build_trees(raw_pages: Optional[pd.DataFrame], out_path: str) -> pd.DataFrame:
    logger.info(f'Building trees for the texts.')
    xmls = raw_pages['text'].progress_apply(get_page_as_xml)
    page_xmls = pd.DataFrame({'title': raw_pages['title'], 'namespace': raw_pages['namespace'], 'text': xmls})
    page_xmls.to_pickle(out_path)
    logger.info(f'Page trees saved to {out_path}.')
    return page_xmls


@cache('out_path', pd.read_pickle)
def extract_lang_sections(page_xmls: pd.DataFrame, out_path: str) -> pd.DataFrame:

    lang_selector = cssselect.CSSSelector('section[depth="2"]')

    def extract(xml: str) -> str:
        """Extract sections with language headers."""
        xml = etree.XML(xml)
        lang_sections = lang_selector(xml)
        return [(etree.tostring(ls, encoding=str), ls.get('header')) for ls in lang_sections]

    logger.info('Extracting language sections.')
    lang_tuples = page_xmls['text'].progress_apply(extract)
    page_xmls['extra'] = lang_tuples

    page_xmls = page_xmls.explode('extra')
    page_xmls = page_xmls.dropna(subset=['extra'])
    page_xmls = page_xmls.reset_index(drop=True)
    logger.info(f'Extracted {len(page_xmls)} language sections.')

    lang_xmls = page_xmls[['title', 'namespace']]
    lang_xmls[['lang_section', 'lang']] = pd.DataFrame(page_xmls['extra'].tolist())
    lang_xmls.to_pickle(out_path)
    logger.info(f'Language sections saved to {out_path}.')
    return lang_xmls


@cache('out_path', pd.read_pickle)
def extract_desc_sections(lang_xmls: pd.DataFrame, out_path: str) -> pd.DataFrame:

    desc_selector = cssselect.CSSSelector('section[header="Descendants"]')

    def extract(xml: str) -> List[str]:
        xml = etree.XML(xml)
        desc_sections = desc_selector(xml)
        return [etree.tostring(ds, encoding=str) for ds in desc_sections]

    logger.info(f'Extracting descendant sections.')
    desc_sections = lang_xmls['lang_section'].progress_apply(extract)

    lang_xmls['desc_section'] = desc_sections

    desc_xmls = lang_xmls.explode('desc_section').dropna(subset=['desc_section'])
    desc_xmls = desc_xmls.reset_index(drop=True)[['title', 'namespace', 'lang', 'desc_section']]
    logger.info(f'Extracted {len(desc_xmls)} descendant sections in total.')
    desc_xmls.to_pickle(out_path)
    logger.info(f'Descendant sections saved to {out_path}.')
    return desc_xmls


@cache('out_path', pd.read_pickle)
def extract_pairs(desc_xmls: pd.DataFrame, out_path: str) -> pd.DataFrame:
    templates = desc_xmls['desc_section'].str.extractall(r'\{\{(?P<template>.+?)\}\}')
    tmpl_name = templates['template'].progress_apply(lambda tmpl: tmpl.split('|')[0])
    tmpl_to_keep = tmpl_name.isin({'desc', 'l', 'desctree'})
    remaining_templates = templates[tmpl_to_keep].reset_index(level=1)
    pairs = pd.merge(desc_xmls, remaining_templates, left_index=True, right_index=True)

    def extract(tmpl: str) -> List[str, str]:
        ret = list()
        for seg in tmpl.split('|')[1:]:
            if '=' not in seg:
                ret.append(seg)
                if len(ret) == 2:
                    return ret

    logger.info('Extracting descendant languages.')
    extra = pairs['template'].progress_apply(extract)
    pairs['extra'] = extra
    pairs = pairs.dropna(subset=['extra'])
    pairs = pairs.reset_index(drop=True)
    pairs[['desc_lang', 'desc_token']] = pd.DataFrame(pairs['extra'].tolist())
    pairs = pairs[['title', 'namespace', 'lang', 'desc_lang', 'desc_token']]
    logger.info(f'Extracting {len(pairs)} pairs in total.')
    pairs.to_pickle(out_path)
    logger.info(f'Pairs saved to {out_path}.')
    return pairs


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

    # Get raw pages first.
    raw_path = f'{folder}/{args.out_name}.raw.pkl'
    raw_pages = read_xml(args.in_path, raw_path)

    # Build trees out of texts.
    xml_path = f'{folder}/{args.out_name}.xml.pkl'
    page_xmls = build_trees(raw_pages, xml_path)

    # Extract language sections.
    lang_path = f'{folder}/{args.out_name}.lang.pkl'
    lang_xmls = extract_lang_sections(page_xmls, lang_path)

    # Extract descendant sections.
    desc_path = f'{folder}/{args.out_name}.desc.pkl'
    desc_xmls = extract_desc_sections(lang_xmls, desc_path)

    # Extract pairs of (desc_lang, desc_token).
    pair_path = f'{folder}/{args.out_name}.pair.pkl'
    pairs = extract_pairs(desc_xmls, pair_path)
    breakpoint()  # BREAKPOINT(j_luo)
