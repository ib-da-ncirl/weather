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
import datetime
import os
import time
from collections import namedtuple
from enum import Enum
from http import HTTPStatus
from typing import Union, AnyStr, Tuple, Any, List
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
INTER_REQ_DELAY = 20  # default inter-request delay to avoid swamping host


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


def load_csv(filepath_or_buffer: Union[str, ZipExtFile, IO[AnyStr]], read_args: dict, filters=None) -> pd.DataFrame:
    """
    Load a csv file
    :param filepath_or_buffer: uri str or file-like object
    :param read_args: pandas.read_csv() arguments
    :param filters: filter conditions
    :return:
    """
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

    filters = []
    if county is not None:
        filters.append(DfFilter(STATION_COUNTY, 'subset_by_val', FilterArg(county, Comparator.EQ)))
    if open_year is not None:
        filters.append(DfFilter(STATION_OPEN, 'subset_by_val', FilterArg(open_year, Comparator.EQ)))
    if close_year is not None:
        filters.append(DfFilter(STATION_CLOSE, 'subset_by_val', FilterArg(close_year, Comparator.EQ)))
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
    if filter_value.comparitor == Comparator.LT:
        def value_filter(val):
            return val < target_value
    elif filter_value.comparitor == Comparator.LT_EQ:
        def value_filter(val):
            return val <= target_value
    elif filter_value.comparitor == Comparator.EQ:
        def value_filter(val):
            return val == target_value
    elif filter_value.comparitor == Comparator.GT:
        def value_filter(val):
            return val > target_value
    elif filter_value.comparitor == Comparator.GT_EQ:
        def value_filter(val):
            return val >= target_value
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


def load_station_data(base_uri: str, filename: Union[ReadParam, list], args: dict, lines=None, filters=None) -> \
        Tuple[Union[pd.DataFrame, List], AnyStr]:
    """
    Retrieve the data for the specified station, as a DataFrame or list of strings
    :param base_uri: Base uri
    :param filename: Station zip filename
    :param args: Arguments
    :param lines: Number of lines to read; if specified a list of strings is returned
    :param filters: filter conditions
    :return: DataFrame of 'lines' is not specified, otherwise list of strings
    """
    if isinstance(filename, ReadParam):
        file_list = [filename]
    else:
        file_list = filename

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
                        df = load_csv(csv_file, entry.read_args, filters=filters)
                    else:
                        df = load_file(csv_file, lines)
                    break

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

        if sid < len(station_ids) - 1:
            time.sleep(int(args['delay']))


DFLT_READ_ARGS = {
    'skiprows': 0,
    'parse_dates': [DATA_DATE],
    'header': 0,
    'converters': {},
    'names': [DATA_DATE],
    'nrows': None
}


def handle_ws_float(value):
    return float(value) if len(value.strip()) > 0 else 0


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
                        if col == DATA_VAPPR:
                            converters[this_column] = handle_ws_float
                        elif col in [DATA_RHUM, DATA_WDDIR]:
                            converters[this_column] = handle_ws_int
                    break
            these_args = DFLT_READ_ARGS.copy()
            these_args['skiprows'] = skiprows
            these_args['names'] = [x for x in DFLT_READ_ARGS['names']]
            these_args['names'].extend(column_names)
            these_args['converters'] = {k: v for k, v in converters.items()}
            these_args['nrows'] = int(args['nrows']) if args['nrows'] is not None else None

            read_args[station_num] = {'args': these_args, 'filename': read_file}

            column_list.extend(column_names)

        inter_request_delay(sid < len(station_ids) - 1, args['delay'])

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
                'nrows': None if 'nrows' not in args else int(args['nrows'])
            }),
            ReadParam(f"dly{station_num}", {
                'skiprows': 9,
                'parse_dates': [DATA_DATE],
                'header': 0,
                'names': [DATA_DATE, DATA_IND_RAIN, DATA_RAIN],
                'nrows': None if 'nrows' not in args else int(args['nrows'])
            })], args)

        if data_cache[station_num] is not None:
            print(f"Loaded {len(data_cache[station_num])} rows")

        inter_request_delay(sid < len(station_ids) - 1, args['delay'])

    return data_cache


def inter_request_delay(condition, delay):
    if condition:
        time.sleep(delay)


def get_station_data(base_uri, station_id: int, station_name: str, read_args: dict, args: dict, filters=None):
    """
    Retrieve the data for the specified station
    :param base_uri: Base uri
    :param station_id: station ids
    :param station_name: station name
    :param read_args: pandas.read_csv() arguments
    :param args: Arguments
    :param filters: filter conditions
    :return:
    """

    print(f"Loading data for station {station_id}: {station_name}")

    df, read_file = load_station_data(base_uri, [ReadParam(read_args['filename'], read_args['args'])], args,
                                      filters=filters)

    if df is not None:
        print(f"Loaded {len(df)} rows")

    return df


def error(msg):
    sys.exit(f"Error: {msg}")


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
    :param data: Dataframe
    :param station: Station number
    :param args: Arguments
    :return:
    """
    if connection is None:
        connection = connection_hbase(args)

    print(f"Saving data from station {station} to hbase")

    count = 0
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
            count += 1

    print(f"Saved {count} rows")

    if close:
        connection.close()


def main():
    arg_ctrl = ArgCtrl(os.path.basename(sys.argv[0]), dflt_config=None)
    arg_ctrl.add_option('n', 'nrows', 'Number of rows of file to read', has_value=True)
    arg_ctrl.add_option('t', 'thrift', f'Address of Thrift host; default {happybase.DEFAULT_HOST}', has_value=True,
                        dfl_value=happybase.DEFAULT_HOST)
    arg_ctrl.add_option('p', 'port', f'Host port; default {happybase.DEFAULT_PORT}', has_value=True,
                        dfl_value=happybase.DEFAULT_PORT)
    arg_ctrl.add_option('d', 'delay', f'Inter-request delay (sec); default {happybase.DEFAULT_PORT}', has_value=True,
                        dfl_value=INTER_REQ_DELAY)
    arg_ctrl.add_option('f', 'folder', f'Folder to save files to', has_value=True, dfl_value='./')
    arg_ctrl.add_option('s', 'save', f'Save files')
    arg_ctrl.add_option('u', 'upload', f'Upload data to hbase')

    app_cfg = arg_ctrl.get_app_config(sys.argv[1:])
    if 'delay' in app_cfg:
        app_cfg['delay'] = int(app_cfg['delay'])

    uri = 'file://../data/'
    # uri = BASE_URL
    stations = get_stations(uri, number=[532, 5704])
    # county='Cork', open_year=YearArg(1990, Comparator.LT_EQ),
    # close_year=YearArg(NOT_CLOSED, Comparator.EQ))

    if 'save' in app_cfg:
        save_data(BASE_URL, stations[STATION_NUM].tolist(), stations[STATION_NAME].tolist(), app_cfg)

    if 'upload' in app_cfg:

        column_list, read_args = get_data_columns(uri, stations[STATION_NUM].tolist(), stations[STATION_NAME].tolist(),
                                                  app_cfg)
        row_template = {get_row_key(x): "" for x in column_list}

        connection = create_table_hbase(DATA_TABLE, app_cfg)

        station_range = range(len(stations[STATION_NUM]))
        for index in station_range:
            station_num = stations[STATION_NUM].iloc[index]
            station_name = stations[STATION_NAME].iloc[index]

            data = get_station_data(uri,    #BASE_URL,
                                    station_num, station_name, read_args[station_num], app_cfg,
                                    filters=[
                                        DfFilter(DATA_DATE, 'subset_by_val', FilterArg(datetime.datetime(1990, 1, 1),
                                                                                       Comparator.GT_EQ)),
                                        DfFilter(DATA_DATE, 'subset_by_val', FilterArg(datetime.datetime(1990, 12, 31),
                                                                                       Comparator.LT_EQ))
                                    ])

            loop_end = index == station_range.stop - 1
            inter_request_delay(not loop_end, app_cfg['delay'])

            save_to_hbase(data, station_num, row_template, app_cfg, connection=connection, close=loop_end)


if __name__ == "__main__":
    main()
