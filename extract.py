#!/usr/bin/env python3
#   The MIT License (MIT)
#  Copyright (c) 2020. Ian Buttimer
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
import os
import time
from collections import namedtuple
from enum import Enum
from http import HTTPStatus
from typing import Union, AnyStr, Tuple, List
from zipfile import ZipFile, ZipExtFile
import pandas as pd
import requests
import sys
import io
import re
import happybase
from typing.io import IO

from misc.arg_ctrl import ArgCtrl

BASE_URL = 'https://cli.fusio.net/cli/climate_data/webdata/'
INTER_REQ_DELAY = 10  # default inter-request delay to avoid swamping host
DFLT_SAVE_FOLDER = './'
DFLT_INFO_PATH = './info.csv'


class Comparator(Enum):
    LT = 1  # less than
    LT_EQ = 2  # less than or equal
    EQ = 3  # equal
    GT_EQ = 4  # greater than or equal
    GT = 5  # greater


class FilterArg:
    def __init__(self, value, comparitor: Union[Comparator, None]):
        self._value = value
        self._comparitor = comparitor

    def __str__(self) -> str:
        return super().__str__() + f" {self._value}, {self._comparitor}"

    @property
    def value(self):
        return self._value

    @property
    def comparitor(self) -> Comparator:
        return self._comparitor


STATION_COUNTY = 'County'
STATION_NUM = 'Station Number'
STATION_NAME = 'name'
STATION_HEIGHT = 'Height (m)'
STATION_EAST = 'Easting'
STATION_NORTH = 'Northing'
STATION_LAT = 'Latitude'
STATION_LNG = 'Longitude'
STATION_OPEN = 'Open Year'
STATION_CLOSE = 'Close Year'

MAX_YEAR = 9999  # value representing max year
MIN_YEAR = 0  # value representing min year
NOT_CLOSED = MAX_YEAR  # year value representing not closed


def load_file(filepath_or_buffer: Union[str, ZipExtFile, IO[AnyStr]], count: int = None) -> []:
    """
    Load a file
    :param filepath_or_buffer: uri str or file-like object
    :param count: Max number of lines to read or None to read all
    :return:
    """
    lines = []

    if count is not None:
        def add_line(ln) -> bool:
            lines.append(ln)
            nonlocal count
            count -= 1
            return count == 0
    else:
        def add_line(ln) -> bool:
            lines.append(ln)
            return False

    if not isinstance(filepath_or_buffer, str):
        for line in filepath_or_buffer:
            if add_line(line):
                break
    else:
        with open(filepath_or_buffer, 'r') as fin:
            for line in fin:
                if add_line(line):
                    break
    return lines


def load_csv(filepath_or_buffer: Union[str, ZipExtFile, IO[AnyStr]], read_args: dict = None, filters=None,
             drop=None) -> pd.DataFrame:
    """
    Load a csv file
    :param filepath_or_buffer: uri str or file-like object
    :param read_args: pandas.read_csv() arguments
    :param filters: filter conditions
    :param drop: list of columns to drop
    :return:
    """
    if read_args is None:
        read_args = {}
    df = None
    if not isinstance(filepath_or_buffer, ZipExtFile):
        read_from, result = get_filepath_or_buffer(filepath_or_buffer)
        if result.status != HTTPStatus.OK:
            read_from = None
    else:
        read_from = filepath_or_buffer

    if read_from is not None:
        df = df_filter(
            pd.read_csv(read_from, **read_args), filters=filters)

    if drop is not None and len(drop):
        df.drop(labels=drop, inplace=True)

    return df


RequestResult = namedtuple('RequestResult', ['status', 'payload'])


def download_file(uri: str, save_to: str = None):
    stream = True if uri.endswith('zip') else False
    response = requests.get(uri, stream=stream, headers={'user-agent': 'weather-data-extract/0.0.1'})
    if response.ok:
        content = response.content
        if save_to is not None:
            open(save_to, 'wb').write(content)
        result = RequestResult(HTTPStatus.OK, content)
    else:
        try:
            response.raise_for_status()
        except requests.HTTPError as http_err:
            error_msg = f'HTTP error: {http_err}'
        except Exception as err:
            error_msg = f'Error: {err}'
        else:
            error_msg = ''
        result = RequestResult(response.status_code, error_msg)

    return result, stream


def get_contents_or_path(uri: str, end_on_err=True) -> Tuple[Union[io.StringIO, io.BytesIO, str], RequestResult]:
    filepath_or_buffer = uri
    result = RequestResult(HTTPStatus.NOT_FOUND, None)
    if isinstance(uri, str):
        if uri.startswith('http'):
            content_result, stream = download_file(uri)
            if content_result.status == HTTPStatus.OK:
                if not stream:
                    filepath_or_buffer = io.StringIO(content_result.payload.decode('utf8'))
                else:
                    filepath_or_buffer = io.BytesIO(content_result.payload)
                result = RequestResult(HTTPStatus.OK, None)
            else:
                result = content_result

        elif uri.startswith('file://'):
            file_path = get_file_path(uri)
            msg = None
            status = HTTPStatus.NOT_FOUND
            if not os.path.exists(file_path):
                msg = f"'{file_path}' does not exist"
            elif not os.path.isfile(file_path):
                msg = f"'{file_path}' is not a file"
            else:
                filepath_or_buffer = file_path
                status = HTTPStatus.OK

            if end_on_err and status != HTTPStatus.OK:
                error(msg)
            else:
                result = RequestResult(status, msg)
        else:
            error(f"Unknown uri, '{uri}'")

    return filepath_or_buffer, result


def get_filepath_or_buffer(uri: Union[str, list], end_on_err=True) -> \
        Tuple[Union[io.StringIO, io.BytesIO, str], RequestResult]:
    filepath_or_buffer = uri
    result = RequestResult(HTTPStatus.NOT_FOUND, None)

    if isinstance(uri, str):
        uri_list = [uri]
    else:
        uri_list = uri

    for entry in uri_list:
        filepath_or_buffer, result = get_contents_or_path(entry, end_on_err=end_on_err)
        if result.status == HTTPStatus.OK:
            break  # have result we need

    return filepath_or_buffer, result


def get_file_path(uri):
    """
    Get the filepath from uri for pandas.read_csv(). It expected full paths such as 'file://localhost/path/to/csv' but
    this allows relative paths as well 'file://../path/to/csv'
    :param uri:
    :return:
    """
    return uri if uri.startswith('file://localhost') else uri[len('file://'):]


DfFilter = namedtuple('DfFilter', ['column', 'op', 'value'])


def df_filter(df: pd.DataFrame, filters=None):
    """
    Filter a DataFrame
    :param df: DataFrame to filter
    :param filters: filter conditions
    :return:
    """
    if filters is None:
        filters = []
    if isinstance(filters, DfFilter):
        filters = [filters]

    sub_df = df
    if sub_df is not None and len(filters):
        print(f"Pre-filter length: {len(sub_df)}")
        for filter_by in filters:
            if filter_by.op == 'subset_by_val':
                sub_df = subset_by_val(sub_df, filter_by.column, filter_by.value)
            elif filter_by.op == 'subset_by_isin':
                # needs list-like
                sub_df = sub_df[sub_df[filter_by.column].isin(filter_by.value.value)]
        print(f"Post-filter length: {len(sub_df)}")

    return sub_df


def get_stations(base_uri, open_year: FilterArg = None, close_year: FilterArg = None, county: str = None,
                 number: Union[int, list] = None):
    print(f"Loading all stations data")

    df = load_csv(f"{base_uri}StationDetails.csv", {
        'skiprows': 1,
        'converters': {STATION_CLOSE: lambda yr: int(yr) if len(yr) > 0 else NOT_CLOSED}
    })

    # correct for errors in data
    # - Castlemagner has close year of 0 when is should be 2002
    index = df[df[STATION_NAME] == 'CASTLEMAGNER'].index[0]
    if df.loc[index, STATION_CLOSE] == 0:
        df.loc[index, STATION_CLOSE] = 2002
    # - remove any commas in station name
    df[STATION_NAME] = df[STATION_NAME].apply(lambda name: name.replace(",", ""))

    filters = []
    if county is not None:
        filters.append(DfFilter(STATION_COUNTY, 'subset_by_val', FilterArg(county, Comparator.EQ)))
    if open_year is not None:
        filters.append(DfFilter(STATION_OPEN, 'subset_by_val', open_year))
    if close_year is not None:
        filters.append(DfFilter(STATION_CLOSE, 'subset_by_val', close_year))
    if number is not None:
        if isinstance(number, int):
            filters.append(DfFilter(STATION_NUM, 'subset_by_val', FilterArg(number, Comparator.EQ)))
        else:
            filters.append(DfFilter(STATION_NUM, 'subset_by_isin', FilterArg(number, None)))

    sub_df = df_filter(df, filters=filters)

    return sub_df


def subset_by_val(df: pd.DataFrame, column: str, filter_value: FilterArg):
    """
    Subset a DataFrame
    :param df: Dataframe
    :param column: Column to filter on
    :param filter_value: filter condition
    :return:
    """
    if filter_value.value == 'max':
        target_value = df[column].max()
    elif filter_value.value == 'min':
        target_value = df[column].min()
    else:
        target_value = filter_value.value
    if isinstance(target_value, list):
        def lt(val):
            return val < min(target_value)

        def lt_eq(val):
            return val <= min(target_value)

        def eq(val):
            return val in target_value

        def gt_eq(val):
            return val >= max(target_value)

        def gt(val):
            return val > max(target_value)
    else:
        def lt(val):
            return val < target_value

        def lt_eq(val):
            return val <= target_value

        def eq(val):
            return val == target_value

        def gt_eq(val):
            return val >= target_value

        def gt(val):
            return val > target_value
    if filter_value.comparitor == Comparator.LT:
        def value_filter(val):
            return lt(val)
    elif filter_value.comparitor == Comparator.LT_EQ:
        def value_filter(val):
            return lt_eq(val)
    elif filter_value.comparitor == Comparator.EQ:
        def value_filter(val):
            return eq(val)
    elif filter_value.comparitor == Comparator.GT:
        def value_filter(val):
            return gt_eq(val)
    elif filter_value.comparitor == Comparator.GT_EQ:
        def value_filter(val):
            return gt(val)
    else:
        def value_filter(val):
            return True
    return df[df[column].apply(value_filter)]


DATA_DATE = 'date'  # Date and Time (utc)
DATA_IND = 'ind'  # Indicator column
DATA_IND_RAIN = 'irain'  # Precipitation Amount Indicator
DATA_RAIN = 'rain'  # Precipitation Amount (mm)
DATA_IND_TEMP = 'itemp'  # Air Temperature Indicator
DATA_TEMP = 'temp'  # Air Temperature (C)
DATA_IND_WETB = 'iwetb'  # Wet Bulb Air Temperature Indicator
DATA_WETB = 'wetb'  # Wet Bulb Air Temperature (C)
DATA_DEWPT = 'dewpt'  # Dew Point Air Temperature (C)
DATA_VAPPR = 'vappr'  # Vapour Pressure (hPa)
DATA_RHUM = 'rhum'  # Relative Humidity (%)
DATA_MSL = 'msl'  # Mean Sea Level Pressure (hPa)
DATA_IND_WDSP = 'iwdsp'  # Mean Hourly Wind Speed Indicator
DATA_WDSP = 'wdsp'  # Mean Hourly Wind Speed (knot)
DATA_IND_WDDIR = 'iwddir'  # Predominant Hourly wind Direction Indicator
DATA_WDDIR = 'wddir'  # Predominant Hourly wind Direction (degree)
DATA_WW = 'ww'  # Synop Code Present Weather
DATA_W = 'w'  # Synop Code Past Weather
DATA_SUN = 'sun'  # Sunshine duration (hours)
DATA_VIS = 'vis'  # Visibility (m)
DATA_CLHT = 'clht'  # Cloud Ceiling Height (100's of ft) - 999 if none
DATA_CLAMT = 'clamt'  # Cloud Amount

ReadParam = namedtuple('ReadParam', ['filename', 'read_args'])


def save_station_data(base_uri: str, file_list: list, save_to: str, args: dict) -> (bool, str):
    """
    Retrieve the data for the specified station
    :param base_uri: Base uri
    :param file_list: Station zip filename
    :param save_to:
    :param args: Arguments
    :return:
    """
    ok = False
    saved_to = None
    for entry in file_list:
        saved_to = os.path.join(save_to, f"{entry}.zip")
        result, _ = download_file(f"{base_uri}{entry}.zip", save_to=saved_to)
        ok = result.status == HTTPStatus.OK
        if ok:
            break
    return ok, saved_to


def load_station_data(base_uri: str, filename: Union[ReadParam, list], args: dict, lines=None, filters=None,
                      drop=None) -> Tuple[Union[pd.DataFrame, List], AnyStr]:
    """
    Retrieve the data for the specified station, as a DataFrame or list of strings
    :param base_uri: Base uri
    :param filename: Station zip filename
    :param args: Arguments
    :param lines: Number of lines to read; if specified a list of strings is returned
    :param filters: filter conditions
    :param drop: list of columns to drop
    :return: DataFrame of 'lines' is not specified, otherwise list of strings
    """
    if isinstance(filename, ReadParam):
        file_list = [filename]
    else:
        file_list = filename
    loops = len(file_list)

    df = None
    read_file = None
    for entry in file_list:
        zip_url = f"{base_uri}{entry.filename}.zip"
        read_from, result = get_filepath_or_buffer(zip_url, end_on_err=False)
        if result.status == HTTPStatus.OK:
            with ZipFile(read_from) as zipped:
                print(f"Retrieved {zip_url}")
                with zipped.open(f"{entry.filename}.csv") as csv_file:
                    read_file = entry.filename
                    if lines is None:
                        df = load_csv(csv_file, entry.read_args, filters=filters, drop=drop)
                    else:
                        df = load_file(csv_file, lines)
                    break
        else:
            print(f"Not available {zip_url}")

        loops -= 1
        inter_request_delay(loops > 0, base_uri, args['delay'])

    return df, read_file


def save_data(base_uri, station_ids: list, station_names: list, args: dict):
    """
    Save the data for the specified stations
    :param base_uri: Base uri
    :param station_ids: List of station ids
    :param station_names: List of station names
    :param args: Arguments
    :return:
    """
    for sid in range(len(station_ids)):
        station_num = station_ids[sid]
        print(f"Saving data for station {station_num}: {station_names[sid]}")

        ok, saved_to = save_station_data(base_uri, [f"hly{station_num}", f"dly{station_num}"], args['folder'], args)
        if ok:
            print(f"Saved as '{saved_to}'")
        else:
            print(f"Not saved")

        loop = sid < len(station_ids) - 1
        inter_request_delay(loop, base_uri, args['delay'])


DFLT_READ_ARGS = {
    'skiprows': 0,
    'parse_dates': [DATA_DATE],
    'header': 0,
    'converters': {},
    'names': [DATA_DATE],
    'nrows': None
}


def handle_ws_float(value):
    return float(value) if len(value.strip()) > 0 else 0.0


def handle_ws_int(value):
    return int(value) if len(value.strip()) > 0 else 0


def get_data_columns(base_uri, station_ids: list, station_names: list, args: dict) -> (list, dict):
    """
    Retrieve the data for the specified stations
    :param base_uri: Base uri
    :param station_ids: List of station ids
    :param station_names: List of station names
    :param args: Arguments
    :return: Tuple of
            - list of all column names which take the form '<name>_<station num>'
            - dict of the pandas.read_csv arguments; key is station number, value is arguments
    """
    column_list = DFLT_READ_ARGS['names'].copy()
    read_args = {}
    for sid in range(len(station_ids)):
        station_num = station_ids[sid]
        print(f"Examining data for station {station_num}: {station_names[sid]}")

        lines, read_file = load_station_data(base_uri, [
            ReadParam(f"hly{station_num}", None), ReadParam(f"dly{station_num}", None)], args, lines=100)

        if lines is not None:
            skiprows = 0
            column_names = []
            converters = {}
            for line in lines:
                line = line.decode('utf-8').strip()
                if len(line) == 0 or line[0:1].isupper() or re.search(r":\s+-", line):
                    # skip leading, empty and column legend lines
                    skiprows += 1
                else:
                    # heading row
                    columns = line.split(",")
                    ind = False  # have ind column flag
                    for col in columns:
                        if col == DATA_DATE:
                            continue
                        elif col == DATA_IND:
                            ind = True
                            continue
                        elif ind:
                            column_names.append(f"{DATA_IND}_{col}_{station_num}")
                            ind = False
                        this_column = f"{col}_{station_num}"
                        column_names.append(this_column)

                        # add converters if required
                        if col in [DATA_RAIN, DATA_TEMP, DATA_WETB, DATA_DEWPT, DATA_VAPPR, DATA_MSL, DATA_SUN]:
                            converters[this_column] = handle_ws_float
                        elif col in [DATA_RHUM, DATA_WDDIR, DATA_VIS, DATA_WDSP, DATA_WW, DATA_W, DATA_CLHT,
                                     DATA_CLAMT]:
                            converters[this_column] = handle_ws_int
                    break

            these_args = DFLT_READ_ARGS.copy()
            these_args['skiprows'] = skiprows
            these_args['names'] = [x for x in DFLT_READ_ARGS['names']]
            these_args['names'].extend(column_names)
            these_args['converters'] = {k: v for k, v in converters.items()}
            these_args['nrows'] = None if 'nrows' not in args else args['nrows']

            read_args[station_num] = {'args': these_args, 'filename': read_file}

            column_list.extend(column_names)

        inter_request_delay(sid < len(station_ids) - 1, base_uri, args['delay'])

    return column_list, read_args


def get_data(base_uri, station_ids: list, station_names: list, args: dict):
    """
    Retrieve the data for the specified stations
    :param base_uri: Base uri
    :param station_ids: List of station ids
    :param station_names: List of station names
    :param args: Arguments
    :return:
    """

    data_cache = {}
    for sid in range(len(station_ids)):
        station_num = station_ids[sid]
        print(f"Loading data for station {station_num}: {station_names[sid]}")

        data_cache[station_num], read_file = load_station_data(base_uri, [
            ReadParam(f"hly{station_num}", {
                'skiprows': 23,
                'parse_dates': [DATA_DATE],
                'header': 0,
                'names': [DATA_DATE, DATA_IND_RAIN, DATA_RAIN, DATA_IND_TEMP, DATA_TEMP, DATA_IND_WETB, DATA_WETB,
                          DATA_DEWPT, DATA_VAPPR, DATA_RHUM, DATA_MSL, DATA_IND_WDSP, DATA_WDSP, DATA_IND_WDDIR,
                          DATA_WDDIR,
                          DATA_WW, DATA_W, DATA_SUN, DATA_VIS, DATA_CLHT, DATA_CLAMT],
                'converters': {DATA_VAPPR: handle_ws_float, DATA_RHUM: handle_ws_int, DATA_WDDIR: handle_ws_int},
                'nrows': None if 'nrows' not in args else args['nrows']
            }),
            ReadParam(f"dly{station_num}", {
                'skiprows': 9,
                'parse_dates': [DATA_DATE],
                'header': 0,
                'names': [DATA_DATE, DATA_IND_RAIN, DATA_RAIN],
                'nrows': None if 'nrows' not in args else args['nrows']
            })], args)

        if data_cache[station_num] is not None:
            print(f"Loaded {len(data_cache[station_num])} rows")

        inter_request_delay(sid < len(station_ids) - 1, base_uri, args['delay'])

    return data_cache


def inter_request_delay(condition, uri, delay):
    if condition and uri.startswith('http'):
        time.sleep(delay)


def get_station_data(base_uri, station_id: int, station_name: str, read_args: dict, args: dict, filters=None,
                     drop=None):
    """
    Retrieve the data for the specified station
    :param base_uri: Base uri
    :param station_id: station ids
    :param station_name: station name
    :param read_args: pandas.read_csv() arguments
    :param args: Arguments
    :param filters: filter conditions
    :param drop: list of columns to drop
    :return:
    """

    print(f"Loading data for station {station_id}: {station_name}")

    df, read_file = load_station_data(base_uri, [ReadParam(read_args['filename'], read_args['args'])], args,
                                      filters=filters, drop=drop)

    if df is not None:
        print(f"Loaded {len(df)} rows")

    return df


def analyse_station_data(base_uri, station_id: int, station_name: str, read_args: dict, args: dict,
                         query_rm_file: bool, filters=None):
    """
    Analyse the data for the specified station
    :param base_uri: Base uri
    :param station_id: station ids
    :param station_name: station name
    :param read_args: pandas.read_csv() arguments
    :param args: Arguments
    :param query_rm_file: Query file deletion if exists
    :param filters: filter conditions
    :return:
    """

    if query_rm_file and os.path.exists(args['info']):
        choice = ''
        while not choice == 'y' and not choice == 'n':
            choice = input(f"'{args['info']}' exists\nDelete it [y/n]: ").lower()
        if choice == 'y':
            os.remove(args['info'])

    df, read_file = load_station_data(base_uri, [ReadParam(read_args['filename'], read_args['args'])], args,
                                      filters=filters)

    if df is not None:
        mode = 'a' if os.path.isfile(args['info']) else 'w'
        with open(args['info'], mode) as fout:
            if mode == 'w':
                fout.write("station_id,station_name,filename,"
                           "min_date,max_date,"
                           "columns,type,"
                           "col_na_count,col_null_count,col_empty_count\n")

            na_str = ''
            null_str = ''
            empty_str = ''
            columns_str = DATA_DATE  # always present
            for col in df.columns:
                na_str = f"{na_str}{';' if len(na_str) else ''}{col}={df[col].isna().sum()}"
                null_str = f"{null_str}{';' if len(null_str) else ''}{col}={df[col].isnull().sum()}"
                empty_str = f"{empty_str}{';' if len(empty_str) else ''}{col}={len(df[df[col] == ''])}"
                # get column name ex station id
                match = re.match(r"^(.*)_\d+$", col)
                if match:
                    columns_str = f"{columns_str}{';' if len(columns_str) else ''}{match.group(1)}"
            # get file type
            match = re.match(r"([A-Za-z]+)\d+", read_args['filename'])
            typ = match.group(1) if match else ''

            fout.write(f"{station_id},{station_name},{read_args['filename']},"
                       f"{df[DATA_DATE].min()},{df[DATA_DATE].max()},"
                       f"{columns_str},{typ},"
                       f"{na_str},{null_str},{empty_str}\n")


def station_analysis_summary(args: dict):
    create = True
    for typ in ['hly', 'dly']:
        # read station analysis csv
        filepath = args['info']
        if not filepath.startswith('file://'):
            filepath = f"file://{args['info']}"
        df = load_csv(filepath, read_args={
            'parse_dates': ['min_date', 'max_date'],
        }, filters=DfFilter('type', 'subset_by_val', FilterArg(typ, Comparator.EQ)))

        df.drop(['col_na_count', 'col_null_count', 'col_empty_count'], axis=1,
                inplace=True)  # just cluttering up the view

        # work out which columns appear for each station
        column_set = set()

        def update_column_set(cols_u):
            nonlocal column_set
            column_set = column_set.union(set(cols_u))

        df['columns'].apply(lambda cols_l: update_column_set(cols_l.split(";")))

        for col in column_set:
            df[col] = False

        cols_idx = list(df.columns).index('columns')
        for row_idx in range(len(df)):
            cols = df.iloc[row_idx, cols_idx].split(';')
            for col in cols:
                col_col_idx = list(df.columns).index(col)
                df.iloc[row_idx, col_col_idx] = True

        # save results
        col_width = 0
        for col in column_set:
            col_width = max(col_width, len(col))

        with open(args['station_summary'], "w" if create else "a") as fhout:
            create = False

            heading = f"Summary for {typ} data"
            fhout.write(f"{heading}\n{'-' * len(heading)}\n\n")

            fhout.write(f"Earliest readings start date: {df['min_date'].min()}\n")
            fhout.write(f"Latest readings start date: {df['min_date'].max()}\n")
            fhout.write(f"Earliest readings end date: {df['max_date'].min()}\n")
            fhout.write(f"Latest readings end date: {df['max_date'].max()}\n")
            fhout.write(f"Max readings coverage date range: {df['min_date'].max()} to {df['max_date'].min()}\n\n")

            have_all = []
            fhout.write(f"Station data availability:\n")
            for col in column_set:
                have_col_df = df[df[col]]
                have_col_cnt = len(have_col_df)
                if have_col_cnt == len(df):
                    have_all.append(col)
                    matches = ''
                else:
                    matches = f", {have_col_df['station_id'].tolist()}"
                fhout.write(f"{col:{col_width}s}: {have_col_cnt:2d} of {len(df):2d}{matches}\n")

            fhout.write(f"\nCommon columns: {have_all}\n")
            fhout.write(f"Station ids: {df['station_id'].tolist()}\n")
            if typ == 'hly':
                fhout.write(f"Estimated dataset size: "
                            f"{(df['max_date'].min() - df['min_date'].max()).days * 24} rows\n\n")


TABLE_PREFIX = "weather"
DATA_TABLE = "data"
COLUMN_FAMILY = "cf"


def get_row_name(timestamp: pd.Timestamp):
    return f"r-{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}{timestamp.hour:02d}"


def get_row_key(key: str):
    return f"{COLUMN_FAMILY}:{key}"


def get_row_values(value_series: pd.Series):
    value_dict = value_series.astype(str).to_dict()
    return {f"{get_row_key(key)}": val for (key, val) in value_dict.items()}


def connection_hbase(args: dict, autoconnect=True):
    """
    Get a hbase connection
    :param args: Arguments
    :param autoconnect:
    :return:
    """
    return happybase.Connection(host=args['thrift'], port=args['port'], autoconnect=autoconnect,
                                table_prefix=TABLE_PREFIX)


def create_table_hbase(table_name: str, args: dict, connection: happybase.connection = None):
    """
    Create hbase table
    :param table_name: Name of table
    :param args: Arguments
    :param connection: hbase connection
    :return:
    """
    if connection is None:
        connection = connection_hbase(args)

    if bytearray(table_name, 'utf-8') not in connection.tables():
        connection.create_table(DATA_TABLE, {COLUMN_FAMILY: dict()})  # use defaults

    return connection


def bytes_to_str(array):
    return array if isinstance(array, str) else "".join(map(chr, array))


def save_to_hbase(data: pd.DataFrame, station: int, row_template: dict, args: dict,
                  connection: happybase.connection = None, close: bool = False):
    """
    Save station data to hbase
    :param row_template:
    :param data: Dataframe
    :param station: Station number
    :param args: Arguments
    :param connection: hbase connection
    :param close: close connection when done flag
    :return:
    """
    if connection is None:
        connection = connection_hbase(args)

    print(f"Saving data from station {station} to hbase")

    count = 0
    total = 0
    table = connection.table(DATA_TABLE)
    with table.batch(batch_size=1000) as b:
        for row in data.iterrows():
            row_name = get_row_name(row[1][DATA_DATE])
            row_values = get_row_values(row[1])

            current_row = table.row(row_name)
            if len(current_row) == 0:
                # row not in db, save everything
                save_row = row_template.copy()
                keys_to_set = save_row.keys()
            else:
                # row exists so only save what is needed
                save_row = {bytes_to_str(key): bytes_to_str(value) for key, value in current_row.items()}
                keys_to_set = set(row_values.keys()) - {get_row_key(DATA_DATE)}

            # new row value with updated values
            save_row = {key: f"{row_values[key] if key in keys_to_set and key in row_values else value}"
                        for key, value in save_row.items()}

            b.put(row_name, save_row)
            count = progress("Row", total, count)

    print(f"Saved {count} rows")

    if close:
        connection.close()


def progress(cmt, total, current, step=100):
    current += 1
    if current % step == 0 or total == current:
        percent = "" if total == 0 else f"({current * 100 / total:.1f}%)"
        print(f"{cmt}: {current} {percent}", flush=True, end='\r' if total > current or total == 0 else '\n')
    return current


def get_station_config(cfg):
    args = {}
    for key, arg in [('station_number', 'number'), ('station_county', 'county'),
                     ('station_open_year', 'open_year'), ('station_close_year', 'close_year')]:
        if key in cfg:
            args[arg] = None if cfg[key] == 'none' else cfg[key]
    for arg in ['open_year', 'close_year']:
        if arg in args and args[arg] is not None:
            bad_arg = True
            if isinstance(args[arg], int):
                args[arg] = FilterArg(args[arg], Comparator.EQ)
                bad_arg = False
            elif isinstance(args[arg], str):
                # process comparative argument e.g. <=1244
                match = re.match(r"([<>=]+)\s*(\d{4})", args[arg])
                if match:
                    for comp, comparator in [('<', Comparator.LT), ('<=', Comparator.LT_EQ), ('=', Comparator.EQ),
                                             ('>=', Comparator.GT_EQ), ('>', Comparator.GT)]:
                        if match.group(1) == comp:
                            args[arg] = FilterArg(match.group(2), comparator)
                            bad_arg = False
                            break

            if bad_arg:
                warning(f"Malformed argument '{arg}', ignoring")

    return args


def get_readings_config(cfg):
    args = {}
    for key, arg in [('reading_stations', 'number')]:
        if key in cfg:
            args[arg] = None if cfg[key] == 'none' else cfg[key]

    return args


def warning(msg):
    print(f"Warning: {msg}")


def error(msg):
    sys.exit(f"Error: {msg}")


def ignore_arg_warning(args_namespace, arg_lst):
    for arg in arg_lst:
        if arg in args_namespace:
            warning(f"Ignoring '{arg}' argument")


def arg_error(arg_parser, msg):
    arg_parser.print_usage()
    sys.exit(msg)


def main():
    arg_ctrl = ArgCtrl(os.path.basename(sys.argv[0]))
    arg_ctrl.add_option('n', 'nrows', 'Number of rows of file to read', has_value=True, typ=int)
    arg_ctrl.add_option('t', 'thrift', f'Address of Thrift host; default {happybase.DEFAULT_HOST}', has_value=True,
                        dfl_value=happybase.DEFAULT_HOST)
    arg_ctrl.add_option('p', 'port', f'Host port; default {happybase.DEFAULT_PORT}', has_value=True,
                        dfl_value=happybase.DEFAULT_PORT)
    arg_ctrl.add_option('d', 'delay', f'Inter-request delay (sec); default {INTER_REQ_DELAY}', has_value=True,
                        dfl_value=INTER_REQ_DELAY, typ=int)
    arg_ctrl.add_option('f', 'folder', f'Folder to save files to; default {DFLT_SAVE_FOLDER}', has_value=True,
                        dfl_value=DFLT_SAVE_FOLDER)
    arg_ctrl.add_option('i', 'info', f'Path to save station data info to; default {DFLT_INFO_PATH}', has_value=True,
                        dfl_value=DFLT_INFO_PATH)
    arg_ctrl.add_option('u', 'uri', f'Uri for data; default {BASE_URL}', has_value=True, dfl_value=BASE_URL)
    arg_ctrl.add_option('b', 'begin', 'Minimum date for readings; yyyy-mm-dd', has_value=True, typ="date=%Y-%m-%d")
    arg_ctrl.add_option('e', 'end', 'Maximum date for readings; yyyy-mm-dd', has_value=True, typ="date=%Y-%m-%d")
    arg_ctrl.add_option('s', 'save', f'Save files', dfl_value=False)
    arg_ctrl.add_option('a', 'analyse', f'Analyse files', dfl_value=False)
    arg_ctrl.add_option('l', 'load', f'Upload data to hbase', dfl_value=False)
    arg_ctrl.add_option('v', 'verbose', f'Verbose mode', dfl_value=False)

    app_cfg = arg_ctrl.get_app_config(sys.argv[1:], set_defaults=False)

    if app_cfg['verbose']:
        print(f"{app_cfg}")

    # load station details
    if app_cfg['load']:  # load data to hbase
        stations_args = get_readings_config(app_cfg)
    else:
        stations_args = get_station_config(app_cfg)
    stations = get_stations(app_cfg['uri'], **stations_args)

    if app_cfg['save']:  # save files to local
        save_data(BASE_URL, stations[STATION_NUM].tolist(), stations[STATION_NAME].tolist(), app_cfg)

    if app_cfg['analyse'] or app_cfg['load']:  # analyse or load data to hbase
        column_list, read_args = get_data_columns(app_cfg['uri'], stations[STATION_NUM].tolist(),
                                                  stations[STATION_NAME].tolist(), app_cfg)

        station_filters = []
        if app_cfg['load']:  # load data to hbase

            # only save columns required
            save_col_list = []
            for save_col in app_cfg['reading_columns']:
                for col in column_list:
                    if col.startswith(save_col):
                        save_col_list.append(col)
            drop_col_list = [value for value in column_list if value not in save_col_list]

            row_template = {get_row_key(x): "" for x in save_col_list}

            if app_cfg['begin'] is not None:
                station_filters.append(DfFilter(DATA_DATE, 'subset_by_val',
                                                FilterArg(app_cfg['begin'], Comparator.GT_EQ)))
            if app_cfg['end'] is not None:
                station_filters.append(DfFilter(DATA_DATE, 'subset_by_val',
                                                FilterArg(app_cfg['end'], Comparator.LT_EQ)))

            connection = create_table_hbase(DATA_TABLE, app_cfg)
        else:
            row_template = None
            connection = None
            drop_col_list = None

        query_rm_file = True
        station_range = range(len(stations[STATION_NUM]))
        for index in station_range:
            station_num = stations[STATION_NUM].iloc[index]
            station_name = stations[STATION_NAME].iloc[index]

            if station_num not in read_args:
                print(f"Ignoring station {station_num}, data not available")
                continue
            print(f"Processing station {station_num} ({index + 1}/{station_range.stop})")

            if app_cfg['load']:  # load data to hbase
                station_drop_col = []
                for col in drop_col_list:
                    if col.endswith(str(station_name)):
                        station_drop_col.append(col)

                data = get_station_data(app_cfg['uri'], station_num, station_name, read_args[station_num], app_cfg,
                                        filters=station_filters, drop=station_drop_col)
            else:  # analyse data
                analyse_station_data(app_cfg['uri'], station_num, station_name, read_args[station_num],
                                     app_cfg, query_rm_file)
                query_rm_file = False
                data = None

            loop_end = index == station_range.stop - 1
            inter_request_delay(not loop_end, app_cfg['uri'], app_cfg['delay'])

            if app_cfg['load']:  # load data to hbase
                save_to_hbase(data, station_num, row_template, app_cfg, connection=connection, close=loop_end)

    if app_cfg['analyse']:  # analyse data
        station_analysis_summary(app_cfg)


if __name__ == "__main__":
    main()
