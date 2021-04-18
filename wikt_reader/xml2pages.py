"""This is the main script that converts an xml Wiktionary dump into several pickle files (of dataframes), including:
    1. *.raw.pkl: all raw pages
    2. *.xml.pkl: all pages with tree structures analyzed, stored in xml format
    3. *.lang.pkl: all language sections
    4. *.desc.pkl: all descendant sections
    5. *.pair.pkl: all cognate pairs

Note that you have to provide a list of namespaces to extract. By default, namespace 0 (normal pages) and
118 (reconstruction pages) are extracted. See the <namespace> tag in the xml Wiktionary dump to find out
what each namespace means.
"""

from __future__ import annotations

import re
import typing
from argparse import ArgumentParser
from dataclasses import dataclass, field
from inspect import signature
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd
from loguru import logger
from lxml import cssselect, etree
from lxml.builder import E
from numpy.lib.type_check import _is_type_dispatcher
from tqdm import tqdm

from wikt_reader.utils import cache


@cache('out_path', pd.read_pickle)
def read_xml(xml_dump_path: str, out_path: str, namespaces: Tuple[int, ...] = (0, 118)) -> pd.DataFrame:
    """Reads xml file using `lxml.etree.parse`.

    Wiktionary dump contains special namespaces, so they are used to extract the pages. See https://stackoverflow.com/questions/6860013/xhtml-namespace-issues-with-cssselect-in-lxml.

    Args:
        xml_dump_path (str): path to the xml Wiktionary dump file.
        out_path (str): path to save the output dataframe to.
        namespaces (Tuple[int, ...], optional): namespaces to extract. See the documentation of this file for more details. Defaults to (0, 118).

    Returns:
        pd.DataFrame: a dataframe with all the contents of the xml dump.
    """
    logger.info(f'Loading xml from {xml_dump_path}.')
    xml_file = etree.parse(open(xml_dump_path, 'rb'))  # type: ignore
    nsmap = xml_file.getroot().nsmap.copy()

    # CSS selector cannot process no-prefix (`None`) namespace. It is converted to a "xhtml" namespace instead.
    nsmap['xhtml'] = nsmap[None]
    del nsmap[None]

    def get_selector(element: str):
        return cssselect.CSSSelector(f'xhtml|{element}', namespaces=nsmap)

    # Extract all pages.
    logger.info('Extracting pages.')
    page_selector = get_selector('page')
    pages = page_selector(xml_file)
    logger.info(f'Extracted {len(pages)} in total.')

    # Extract titles and texts with proper namespaces.
    title_selector = get_selector('title')
    text_selector = get_selector('text')
    ns_selector = get_selector('ns')

    logger.info('Extracting titles and texts from pages.')
    records = list()
    for page in tqdm(pages):
        ns = int(ns_selector(page)[0].text)  # This is the namespace.
        if ns in namespaces:
            title = title_selector(page)[0].text
            text = text_selector(page)[0].text
            records.append({'title': title, 'namespace': ns, 'text': text})
    df = pd.DataFrame(records)

    # Clean up, save and return.
    logger.info(f'Extracted {len(df)} dictionary entries.')
    df = df.dropna()
    logger.info(f'Empty pages dropped, ending up with {len(df)} pages.')
    df.to_pickle(out_path)
    logger.info(f'Pages saved to {out_path}.')
    return df


header_pat = re.compile(r'^==+([^=]+?)==+$')


@cache('out_path', pd.read_pickle)
def build_trees(raw_pages: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Builds tree structures for the raw pages (in Wikitext format) and converts them to xml.

    Args:
        raw_pages (pd.DataFrame): the input saved as a dataframe.
        out_path (str): the path to save the output to

    Returns:
        pd.DataFrame: the dataframe with the new xml page (with tree structures in Wikitext).
    """

    @dataclass
    class Node:
        """Represents one node in the tree."""
        depth: int
        header: str
        text: str
        children: List[Node] = field(default_factory=list)

        def to_element(self) -> etree.Element:  # type: ignore
            """Convert node to an element with proper sub-elements."""
            sub_elements = [child.to_element() for child in self.children]
            return E('section', self.text, *sub_elements, depth=str(self.depth), header=self.header)

    def get_page_as_xml(text: str) -> str:
        """Convert the text from Wikitext format to xml format."""
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
        xml = etree.tostring(root.to_element(), encoding=str)  # type: ignore
        return xml

    logger.info(f'Building trees for the texts.')
    xmls = raw_pages['text'].progress_apply(get_page_as_xml)
    page_xmls = pd.DataFrame({'title': raw_pages['title'], 'namespace': raw_pages['namespace'], 'text': xmls})
    page_xmls.to_pickle(out_path)
    logger.info(f'Page trees saved to {out_path}.')
    return page_xmls


@cache('out_path', pd.read_pickle)
def extract_lang_sections(page_xmls: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Extract all the languages sections.

    Args:
        page_xmls (pd.DataFrame): the input dataframe with all pages stored in xml format (after building tree structures).
        out_path (str): the path to save the output to.

    Returns:
        pd.DataFrame: the dataframe with all extracted language sections.
    """

    lang_selector = cssselect.CSSSelector('section[depth="2"]')

    def extract(xml: str) -> str:
        """Extract sections with language headers."""
        xml = etree.XML(xml)  # type: ignore
        lang_sections = lang_selector(xml)
        return [(etree.tostring(ls, encoding=str), ls.get('header')) for ls in lang_sections]  # type: ignore

    logger.info('Extracting language sections.')
    lang_tuples = page_xmls['text'].progress_apply(extract)
    page_xmls['extra'] = lang_tuples

    page_xmls = page_xmls.explode('extra')  # type: ignore
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
    parser.add_argument('xml_dump_path', type=str, help='Path to the xml Wiktionary dump.')
    parser.add_argument('out_prefix', type=str, help='Prefix for the output files.')
    parser.add_argument('--namespaces', type=int, nargs='+', default=(0, 118),
                        help='Namespaces to keep. The exact meanings of different namespaces are provided in the <namespaces> tag.')
    args = parser.parse_args()

    # Make sure `out_prefix` is properly formatted.
    if '.' in args.out_prefix:
        raise ValueError(f'{args.out_prefix} should not include "." or suffixes.')

    # Track pandas progress.
    tqdm.pandas()

    # Create folder for caching/storing processed files.
    out_folder = Path('./processed')
    out_folder.mkdir(parents=True, exist_ok=True)

    # Get raw pages first.
    raw_path = f'{out_folder}/{args.out_prefix}.raw.pkl'
    raw_pages = read_xml(args.xml_dump_path, raw_path, namespaces=args.namespaces)

    # Build trees out of texts.
    xml_path = f'{out_folder}/{args.out_prefix}.xml.pkl'
    page_xmls = build_trees(raw_pages, xml_path)

    # Extract language sections.
    lang_path = f'{out_folder}/{args.out_prefix}.lang.pkl'
    lang_xmls = extract_lang_sections(page_xmls, lang_path)

    # Extract descendant sections.
    desc_path = f'{out_folder}/{args.out_prefix}.desc.pkl'
    desc_xmls = extract_desc_sections(lang_xmls, desc_path)

    # Extract pairs of (desc_lang, desc_token).
    pair_path = f'{out_folder}/{args.out_prefix}.pair.pkl'
    pairs = extract_pairs(desc_xmls, pair_path)
