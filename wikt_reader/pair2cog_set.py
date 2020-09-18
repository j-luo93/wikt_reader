import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Set, Tuple

import pandas as pd
import pycountry
from loguru import logger
from lupa import LuaError, LuaRuntime
from lxml import etree
from networkx import DiGraph, descendants
from tqdm import tqdm

extract_token_pat = re.compile(r'^Reconstruction:.+?/\*?(.+)$')


def extract_token(text: str) -> str:
    return extract_token_pat.match(text).group(1)


def remove_reconstruction(item: Tuple[str, str]):
    title, ns = item
    # 118 indicates this page is for reconstructed tokens.
    if ns == 118:
        return extract_token(title)
    return title


@dataclass
class Token:
    lang: str
    form: str
    title: str = field(init=False)

    def __post_init__(self):
        if self.lang == 'prs':
            self.lang = 'fa'
        try:
            self.title = make_entry_name(self.lang, self.form)
        except LuaError:
            self.title = self.form

    def __hash__(self):
        return hash((self.lang, self.title))

    def __eq__(self, other):
        return self.lang == other.lang and self.title == other.title


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('mod_xml_path', type=str, help='Path to the pickled xml for modules.')
    parser.add_argument('pair_path', type=str, help='Path to the pickled cognate pair data.')
    parser.add_argument('common_ancestor', type=str,
                        help='The language code (Wiktionary standard) for the common ancestor.')
    args = parser.parse_args()

    tqdm.pandas()

    # Read module data and keep the language data.
    mods = pd.read_pickle(args.mod_xml_path)
    lang_data_pat = r'^Module:languages/data[23x].*(?<!documentation)$'
    lang_data_mask = mods['title'].str.match(lang_data_pat)
    lang_data = mods[lang_data_mask].reset_index(drop=True)

    # Clean up tags and only keep the raw lua code.
    logger.info(f'Extracting code from {args.mod_xml_path}.')
    lang_data['code'] = lang_data['text'].progress_apply(lambda xml: etree.XML(xml).text)

    # Write lua code to files.
    folder = Path('lualib')
    folder.mkdir(exist_ok=True)
    lib_files = list()

    import_code = """
    local m = {}
    """
    import_block = """
    new_m = require "{}"
    if type(new_m) == "table" then
        for k, v in pairs(new_m) do
            m[k] = v
        end
    end
    """
    for title, lua_code in zip(lang_data['title'], lang_data['code']):
        out_path = folder / '_'.join(title.split('/')[1:])
        import_code += import_block.format(out_path)
        with out_path.with_suffix('.lua').open('w') as fout:
            fout.write(lua_code)

    # Write main lua code that defines useful helper functions for the main Python program.
    mw_code = """
    ustring = require 'ustring/ustring'
    mw = {
        ustring = ustring
    }
    """ + import_code + """
    export = {
        m = m
    }

    function do_entry_name_or_sort_key_replacements(text, replacements)
        if replacements.from then
            for i, from in ipairs(replacements.from) do
                local to = replacements.to[i] or ""
                text = mw.ustring.gsub(text, from, to)
            end
        end

        if replacements.remove_diacritics then
            text = mw.ustring.toNFD(text)
            text = mw.ustring.gsub(text,
                '[' .. replacements.remove_diacritics .. ']',
                '')
            text = mw.ustring.toNFC(text)
        end

        return text
    end

    function export.makeEntryName(lang, text)
        local langData = m[lang]
        text = mw.ustring.match(text, "^[¿¡]?(.-[^%s%p].-)%s*[؟?!;՛՜ ՞ ՟？！︖︕।॥။၊་།]?$") or text

        if langData[1] == "ar" then
            local U = mw.ustring.char
            local taTwiil = U(0x640)
            local waSla = U(0x671)
            -- diacritics ordinarily removed by entry_name replacements
            local Arabic_diacritics = U(0x64B, 0x64C, 0x64D, 0x64E, 0x64F, 0x650, 0x651, 0x652, 0x670)

            if text == waSla or mw.ustring.find(text, "^" .. taTwiil .. "?[" .. Arabic_diacritics .. "]" .. "$") then
                return text
            end
        end

        if type(langData.entry_name) == "table" then
            text = do_entry_name_or_sort_key_replacements(text, langData.entry_name)
        end

        return text
    end

    return export
    """

    with (folder / 'mw.lua').open('w') as fout:
        fout.write(dedent(mw_code))

    lua = LuaRuntime()
    export = lua.require('lualib/mw')
    # This function is used to obtain the page title for any token. Note that this process is language specific.
    make_entry_name = export['makeEntryName']
    # This is the language code used by `make_entry_name`.
    data = export['m']
    name2code = dict()
    for code, d in data.items():
        names = [d[1]]
        if 'aliases' in d:
            names += list(d['aliases'].values())
        for name in names:
            name2code[name] = code

    # Get cognate pair data.
    pairs = pd.read_pickle(args.pair_path)
    logger.info('Converting language names to codes.')
    pairs['lang'] = pairs['lang'].progress_apply(lambda name: name2code[name.strip()])
    logger.info('Dealing with reconstructed titles.')
    pairs['title'] = pairs[['title', 'namespace']].progress_apply(remove_reconstruction, axis=1)
    logger.info('Dealing with reconstructed descendant tokens.')
    pairs['desc_token'] = pairs['desc_token'].str.replace(r'^\*', '')

    # Construct a graph representing all cognate pairs.
    logger.info('Constructing graphs for all tokens.')
    g = DiGraph()
    for lang, title, desc_lang, desc_token in tqdm(zip(pairs['lang'], pairs['title'], pairs['desc_lang'], pairs['desc_token']), total=len(pairs)):
        token = Token(lang, title)
        desc_token = Token(desc_lang, desc_token)
        g.add_node(token)
        g.add_node(desc_token)
        g.add_edge(token, desc_token)

    # Get trees.
    logger.info('Getting all trees from the graph.')
    cog_sets: Dict[Token, Set[Token]] = dict()
    for node in tqdm(g.nodes):
        if node.lang == args.common_ancestor:
            cog_sets[node] = descendants(g, node)

    # Get the final output.
    logger.info(f'Getting the final output.')
    records = list()
    for token, cognates in tqdm(cog_sets.items()):
        for cog in cognates:
            records.append({args.common_ancestor: token.form, 'desc_lang': cog.lang, 'desc_form': cog.form})
    cog_set_df = pd.DataFrame(records)
    out_path = f'processed/{args.common_ancestor}.tsv'
    cog_set_df.to_csv(out_path, index=None, sep='\t')
    logger.info(f'Final output saved to {out_path}.')
