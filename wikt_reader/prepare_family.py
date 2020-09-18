from argparse import ArgumentParser
from functools import lru_cache
from textwrap import indent

import pandas as pd
from loguru import logger
from pycountry import languages
from tqdm import tqdm


@lru_cache(maxsize=None)
def standardize_code(code: str) -> str:
    if len(code) == 2:
        return languages.get(alpha_2=code).alpha_3
    else:
        return code


def to_multiline(msg: str, obj: object) -> str:
    """Convert the string representation of an object into a multiline text, useful for loggers."""
    return '\n' + msg + '\n' + indent(str(obj), '    ')


if __name__ == "__main__":
    tqdm.pandas()

    parser = ArgumentParser()
    parser.add_argument('form_path', type=str, help='Path to the forms.')
    parser.add_argument('lang_path', type=str, help='Path to the language metdata file.')
    parser.add_argument('out_path', type=str, help='Path to save the data frame.')
    parser.add_argument('--family', type=str, help='Which (sub)family to keep.')
    parser.add_argument('--common_ancestor', type=str, help='Language code for the common ancestor language.')
    parser.add_argument('--family_level', type=str,
                        choices=['family', 'subfamily'], default='subfamily',
                        help='Which level of family relation to specify.')
    args = parser.parse_args()

    # Prepare data.
    cog_df = pd.read_csv(args.form_path, sep='\t', error_bad_lines=False)
    lang_df = pd.read_csv(args.lang_path, sep='\t')
    lang_cog_df = pd.merge(lang_df, cog_df, left_on='glotto_code',
                           right_on='Glottocode')
    query = f'{args.family_level} == "{args.family}"'
    lang_cog_df = lang_cog_df.query(query)

    family = pd.read_csv(f'processed/{args.common_ancestor}.tsv', sep='\t', keep_default_na=False)
    family['lang_code'] = family['desc_lang'].progress_apply(standardize_code)
    merged = pd.merge(family, lang_cog_df,
                      left_on=['lang_code', 'desc_form'],
                      right_on=['iso_code', 'Word_Form'],
                      how='inner')
    out_df = merged[[args.common_ancestor, 'lang_code', 'Word_Form', 'rawIPA', 'IPA']]
    out_df = out_df.drop_duplicates()
    out_df.to_csv(args.out_path, sep='\t', index=None)
    lang_stats = out_df['lang_code'].value_counts()
    logger.info(to_multiline('Stats for languages:', lang_stats))
    logger.info(f'Output saved to {args.out_path}, with {len(out_df)} entries in total.')
