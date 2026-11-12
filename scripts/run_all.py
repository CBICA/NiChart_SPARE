# This wrapper takes standard NiChart_DLMUSE/merged demographics CSV files and handles preprocessing steps necessary for SPARE, then invokes SPARE.

import argparse
import pandas as pd
import sys
import os
import tempfile
import re
import joblib
from pathlib import Path

parser = argparse.ArgumentParser(description="Run many SPARE predictions in one invocation (CVM or non-CVM). Preprocess CSV and run spare_scores")
parser.add_argument('-i', '--input', required=True, help='Input CSV file')
parser.add_argument('-demog','--demographics_csv', required=False, help='Optional demographics CSV to merge with input csv')
parser.add_argument('-o', '--output', required=True, help='Output CSV file')
parser.add_argument('--category', default='misc', help='Category to run. Values: "cvm", "misc"')
parser.add_argument('--harmonize', default=False, action='store_true', help='Pass to run harmonized SPARE models only (default: only run non-harmonized).')
args, unknown_args = parser.parse_known_args()

if not args.input.lower().endswith('.csv'):
    print("Error: input file must be a CSV file.")
    sys.exit(1)
if not os.path.exists(args.input):
    print(f"Error: Input file '{args.input}' does not exist.")
    sys.exit(1)
if args.demographics_csv is not None:
    if not args.demographics_csv.lower().endswith('.csv'):
        print("Error: demographics file must be a CSV file.")
        sys.exit(1)
    if not os.path.exists(args.demographics_csv):
        print(f"Error: Input file '{args.demographics_csv}' does not exist.")
        sys.exit(1)

def list_filtered_files(directory, forbidden_substrings):
    """
    Return all filenames in `directory` that do NOT contain
    any of the substrings in `forbidden_substrings`.
    """
    files = []
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
        if os.path.isfile(full):
            if not any(sub in name for sub in forbidden_substrings):
                files.append(full)
    return files

def merge_dfs_on_mrid(df_list):
    """
    Merge a list of dataframes on the 'MRID' column.
    Returns a single merged dataframe.
    """
    if not df_list:
        raise ValueError("No dataframes provided.")

    # Start with the first dataframe
    merged = df_list[0]

    # Iteratively merge all remaining
    for df in df_list[1:]:
        merged = merged.merge(df, on="MRID", how="outer")

    return merged

def merge_csvs_on_mrid(csv_paths):
    """
    Load all CSVs in `csv_paths` and merge them on the 'MRID' column.
    Returns a single merged dataframe.
    """
    if not csv_paths:
        print("No CSV paths were passed to MRID-based merge. Did SPARE produce any output?")
        return pd.DataFrame()

    # Load the first one as the base
    merged = pd.read_csv(csv_paths[0])

    # Iteratively merge all remaining
    for path in csv_paths[1:]:
        df = pd.read_csv(path)
        merged = merged.merge(df, on="MRID", how="outer")

    return merged

def extract_spare_tag(path):
    """
    Given a filename like SPARE-BA-RAW-xyz.joblib,
    return the tag after SPARE-, e.g. 'BA'.

    The tag is everything between "SPARE-" and the -Harmonized/-RAW marker, so
    a multi-part name keeps all of its parts. Stopping at the first hyphen
    collapsed the five SPARE-CVM-<CONDITION>-Harmonized models onto the single
    tag "CVM": four were then skipped by the duplicate-tag guard below, and the
    one survivor's score was published under a name that does not say which
    condition it predicts.
    """
    name = os.path.basename(path)
    m = re.search(r"SPARE-(.+?)-(?:Harmonized|RAW)\b", name, flags=re.IGNORECASE)
    if m:
        # CVM-T2D -> CVM_T2D, so the tag is usable as a column-name fragment.
        return m.group(1).replace('-', '_')
    # Names carrying neither marker, e.g. SPARE-MDD.joblib.
    m = re.search(r"SPARE-([A-Za-z0-9]+)[-.]", name)
    return m.group(1) if m else None


# Every inference mode NiChart_SPARE can be asked for. SPARE names its output
# column after the mode, not after the model, so two models sharing a mode also
# share a column name and only the tag tells them apart. A mode missing from
# this list keeps its generic name and the results collide in the final outer
# merge as SPARE_<MODE>_x / SPARE_<MODE>_y -- which is what happened to the
# CVM models, whose mode was never listed here.
INFERENCE_MODES = ('CL', 'RG', 'CVM', 'AD', 'BA', 'HT', 'T2B', 'SM', 'BMI')


def rename_spare_columns(df, tag):
    """
    Rename SPARE/GT columns using the provided tag (e.g., 'MDD').
    Returns a new dataframe.
    """

    mapping = {}

    for mode in INFERENCE_MODES:
        # SPARE_<mode> -> SPARE_<tag>
        if f"SPARE_{mode}" in df.columns:
            mapping[f"SPARE_{mode}"] = f"SPARE_{tag}"

        # SPARE_<mode>_decision_function -> SPARE_<tag>_decision_function
        if f"SPARE_{mode}_decision_function" in df.columns:
            mapping[f"SPARE_{mode}_decision_function"] = f"SPARE_{tag}_decision_function"

        # GT_<mode> -> <tag>_GT_<mode>
        if f"GT_{mode}" in df.columns:
            mapping[f"GT_{mode}"] = f"{tag}_GT_{mode}"

    return df.rename(columns=mapping)

def debug_column(df: pd.DataFrame, col: str, name: str = "df", n: int = 5) -> None:
    """Print dtype and sample-value diagnostics for one column."""
    if col not in df.columns:
        print(f"[{name}] column '{col}' not found")
        return

    s = df[col]
    print(f"[{name}] {col}: dtype={s.dtype}, n={len(s)}, nulls={s.isna().sum()}")

    # If dtype is object, it can still hide mixed python types (int, str, etc.)
    if s.dtype == object:
        type_counts = s.map(lambda x: type(x).__name__).value_counts()
        print(f"[{name}] {col}: underlying python types -> {type_counts.to_dict()}")

    # Show repr() of a few values so you can SEE quotes/no-quotes
    sample = s.dropna().head(n)
    print(f"[{name}] {col}: sample values -> {[repr(v) for v in sample]}")


def require_mrid(df, source):
    """Fail with a useful message rather than a bare KeyError on merge."""
    if 'MRID' not in df.columns:
        print(f"Error: '{source}' has no 'MRID' column. Found: {list(df.columns)[:15]}")
        sys.exit(1)
    dupes = df['MRID'].duplicated().sum()
    if dupes:
        examples = df.loc[df['MRID'].duplicated(), 'MRID'].head(5).tolist()
        print(f"WARNING: '{source}' contains {dupes} duplicate MRID(s), e.g. {examples}. "
              "Rows will be multiplied by the merge and scores will be duplicated.")


def merge_demographics(df_input, df_demog):
    """Merge demographics onto the input, resolving columns present in both.

    A harmonized DLMUSE CSV already carries Age/Sex: the harmonizer merges the
    covariates in and passes them through to its output. A plain merge therefore
    produces Age_x/Age_y and Sex_x/Sex_y, and the plain 'Sex' that everything
    downstream expects stops existing — which looks to the caller like the
    bizarre "Required 'Sex' column not found" error on a file that visibly has
    a Sex column.

    Demographics win where they carry a value; the input fills any gaps. Both
    sources normally originate from the same participants.csv, so this is a
    no-op in the common case and simply makes the duplication harmless.
    """
    overlap = [c for c in df_demog.columns if c != 'MRID' and c in df_input.columns]
    if overlap:
        print(f"Note: these columns exist in both the input and the demographics CSV "
              f"and will be taken from the demographics file: {overlap}")

    merged = pd.merge(
        df_input, df_demog, on='MRID', how='inner', suffixes=('', '__demog')
    )

    for col in overlap:
        dup = f"{col}__demog"
        if dup in merged.columns:
            merged[col] = merged[dup].combine_first(merged[col])
            merged.drop(columns=[dup], inplace=True)

    # Nothing should survive with the suffix; drop any stragglers so they can
    # never be mistaken for ROI features further down.
    leftovers = [c for c in merged.columns if c.endswith('__demog')]
    if leftovers:
        merged.drop(columns=leftovers, inplace=True)

    return merged


# Accepted spellings for sex. Anything outside these is an error rather than a
# silent default: the previous `1 if x == 'M' else 0` quietly recorded every
# unrecognised value -- 'm', 'Male', 1/2 codes, blanks -- as female.
_SEX_MALE = {'m', 'male', '1', '1.0'}
_SEX_FEMALE = {'f', 'female', '2', '2.0'}


def sex_to_male_flag(value):
    key = str(value).strip().lower()
    if key in _SEX_MALE:
        return 1
    if key in _SEX_FEMALE:
        return 0
    return None


df_original = pd.read_csv(args.input)
debug_column(df_original, "MRID", "dlmuse-csv")
require_mrid(df_original, args.input)

if args.demographics_csv is not None:
    print("Detected optional demographics file. Merging files to ensure all data is available for SPARE.")
    df_demog = pd.read_csv(args.demographics_csv)
    debug_column(df_demog, "MRID", "demog-csv")
    require_mrid(df_demog, args.demographics_csv)
    print(f"Demographic dataframe passed to SPARE runall has columns: {df_demog.columns}")
    len_original = len(df_original)
    len_demog = len(df_demog)
    print(f"Length of original data: { len_original }")
    print(f"Length of demographic data: { len_demog }")
    input_mrids = set(df_original['MRID'])
    df_original = merge_demographics(df_original, df_demog)
    len_merged = len(df_original)
    print(f"Length of merged data: { len_merged }")
    if len_merged < len_original:
        dropped = sorted(input_mrids - set(df_original['MRID']))
        print(f"WARNING: Merged CSV has fewer entries than the original data. "
              f"{len(dropped)} subject(s) had no demographics row and were dropped, "
              f"e.g. {dropped[:10]}. Check demographics and input CSVs for completeness.")

if 'Sex' not in df_original.columns:
    # Name the likely cause instead of just the symptom.
    split = sorted(c for c in df_original.columns if c.startswith(('Sex_', 'Sex')) and c != 'Sex')
    hint = (f" Columns {split} are present, which means the input and demographics "
            f"CSVs both carried a Sex column and the merge renamed them."
            if split else f" Available columns: {list(df_original.columns)[:15]}")
    print(f"Error: Required 'Sex' column not found in the CSV file.{hint}")
    sys.exit(1)

if 'Age' not in df_original.columns:
    print("WARNING: no 'Age' column found. Most SPARE models expect Age as a "
          "covariate or regression target and will likely fail or return "
          "meaningless values without it.")
else:
    bad_age = df_original['Age'].isna().sum() + (
        pd.to_numeric(df_original['Age'], errors='coerce').isna()
        & df_original['Age'].notna()
    ).sum()
    if bad_age:
        offenders = df_original.loc[
            pd.to_numeric(df_original['Age'], errors='coerce').isna(), 'MRID'
        ].head(10).tolist()
        print(f"WARNING: {bad_age} row(s) have a missing or non-numeric Age, "
              f"e.g. {offenders}. These subjects' scores may be unusable.")

print(f"Original dataframe passed to SPARE runall has columns: {df_original.columns}")

sex_flags = df_original['Sex'].map(sex_to_male_flag)
if sex_flags.isna().any():
    unrecognised = sorted({repr(v) for v in df_original.loc[sex_flags.isna(), 'Sex'].unique()})
    offenders = df_original.loc[sex_flags.isna(), 'MRID'].head(10).tolist()
    print(f"Error: unrecognised 'Sex' value(s): {', '.join(unrecognised[:10])}. "
          f"Expected M/F (or male/female, or the 1=male/2=female coding). "
          f"Affected MRIDs e.g. {offenders}.")
    sys.exit(1)
df_original['Sex_M'] = sex_flags.astype(int)

#if 'DL_MUSE_Volume_702' in df_original.columns:
#    df_original['702'] = df_original['DL_MUSE_Volume_702']
#if 'H_DL_MUSE_Volume_702' in df_original.columns:
#    df_original['702'] = df_original['H_DL_MUSE_Volume_702']

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
    temp_csv_path = tmp_file.name
    df_original.to_csv(temp_csv_path, index=False)

## Here's where we filter out models

non_cvm_models = list_filtered_files('/spare_score/Models/final_models', forbidden_substrings=['HYPERTENSION', 'DIABETES', 'OBESITY', 'SMOKING', 'HYPERLIPIDEMIA', 'MDD', 'PSY'])
cvm_models = list_filtered_files('/spare_score/Models/withDLWMLS', forbidden_substrings=[])

model_location_dict = {'cvm': cvm_models, 'misc': non_cvm_models}

if args.category not in model_location_dict:
    print(f"Error: unknown --category '{args.category}'. "
          f"Expected one of: {sorted(model_location_dict)}")
    sys.exit(1)

all_tmp_csvs = []
all_successful_tmp_csvs = []
successful_tags = []
exit_codes = []
encountered_spare_tags = []
print("See https://huggingface.co/nichart/SPARE-ALL/tree/main for all available model files...")
num_models_attempted = 0
num_models_succeeded = 0
for model in model_location_dict[args.category]:
    print(f"Scanning model {model}")
    if 'harmonized' in model.lower():
        if not args.harmonize:
            print("Skipping (this model is for harmonized data)...")
            continue
    else:
        if args.harmonize:
            print("Skipping (this model is for unharmonized data)...")
            continue
    model_joblib = joblib.load(model)
    inference_mode = model_joblib['meta_data']['spare_type']
    spare_tag = extract_spare_tag(model)
    if spare_tag is None:
        # Without a tag the output columns would be named SPARE_None.
        print(f"Skipping (could not parse a SPARE tag from {os.path.basename(model)})...")
        continue
    if spare_tag in encountered_spare_tags:
        print("Skipping due to duplicate spare tag...")
        continue
    else:
        encountered_spare_tags.append(spare_tag)

    print(f"Spare tag {spare_tag}, inference mode {inference_mode}")
    num_models_attempted += 1
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_out:
        temp_out_path = tmp_out.name
        all_tmp_csvs.append(temp_out_path)

        command_parts = ['NiChart_SPARE', '-kv', 'MRID', '-i', temp_csv_path, '-o', temp_out_path, '-t', inference_mode, '-m', model] + unknown_args
        command = ' '.join(f"{part}" if ' ' in part else part for part in command_parts)

        print(f"Running inference using SPARE model from file {model}")

        exit_code = os.system(command)
        exit_codes.append(exit_code)
        if os.WEXITSTATUS(exit_code) > 0:
            print(f"Nonzero exit code encountered while executing SPARE model {model}")
        else:
            successful_tags.append(spare_tag)
            all_successful_tmp_csvs.append(temp_out_path)
            num_models_succeeded += 1

if num_models_attempted == 0:
    print("No relevant models were found for this combination of requested SPARE scores and harmonization status. This is not necessarily an error, please see documentation.")
    sys.exit(0)

if num_models_succeeded == 0:
    # Previously this fell through to merge_dfs_on_mrid([]), which raises
    # ValueError and exits with a traceback instead of a diagnosis.
    print("Error: every SPARE model failed; no output was produced. "
          "See the per-model messages above.")
    sys.exit(1)

dfs = []
for index, csv in  enumerate(all_successful_tmp_csvs):
    df = pd.read_csv(csv)
    df = rename_spare_columns(df, successful_tags[index])
    dfs.append(df)


df_merged = merge_dfs_on_mrid(dfs)
# index=False: the default wrote an unnamed index column into every result CSV.
df_merged.to_csv(args.output, index=False)

if any([os.WEXITSTATUS(exit_code) > 0 for exit_code in exit_codes]):
    print("Some SPARE scores did not succeed, but those that did have been written to the specified output.")
    # sys.exit() takes a process exit status, but os.system() returns a wait
    # status -- exiting with 256 (SIGCHLD-encoded 1) truncates to 0 and reports
    # success. Exit 1 so a partial failure is actually visible to the caller.
    sys.exit(1)
