#  The MIT License (MIT)
#  Copyright (c) 2019-2020. Ian Buttimer
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
import getopt
import os
import sys
from collections import namedtuple
from typing import Union

import pkg_resources
import yaml

ConfigOpt = namedtuple('ConfigOpt', ['short', 'long', 'desc', 'has_value', 'req', 'dfl_value', 'type'])


class ArgCtrl:

    def __init__(self, name: str, dflt_config: Union[str, None] = 'config.yaml'):
        self._opts = {}
        self.add_option('h', 'help', 'Display usage')
        if dflt_config is not None:
            self.add_option('c', 'config', 'Specify path to configuration script', has_value=True,
                            dfl_value=dflt_config)
        self._name = name

    def add_option(self, short_name: str, long_name: str, description: str, has_value: bool = False,
                   required: bool = False, dfl_value=None, type='str'):
        """
        Add an option
        :param short_name: Short name
        :param long_name: Long name
        :param description: Description
        :param has_value: Has value flag
        :param required: Is required flag
        :param dfl_value: Default value, if not present
        :param type: type to convert command line value to
        :return:
        """
        if short_name in self._opts.keys():
            self.usage()
            raise ValueError(f"Option already exists: {short_name}")

        self._opts[short_name] = ConfigOpt(short_name if not has_value else f"{short_name}:",
                                           long_name if not has_value else f"{long_name}=",
                                           description, has_value, required, dfl_value, type)

    @property
    def options(self):
        return self._opts

    def get_short_opts(self) -> str:
        """
        Get string of short options
        :return:
        """
        opts_lst = ''
        for o_key in self._opts.keys():
            opts_lst += self._opts[o_key].short
        return opts_lst

    def get_long_opts(self) -> list:
        """
        Get list of log options
        :return:
        """
        opts_lst = []
        for o_key in self._opts.keys():
            if self._opts[o_key].long is not None:
                opts_lst.append(self._opts[o_key].long)
        return opts_lst

    def get_short_opt(self, o_key) -> str:
        """
        Get the short option (as appears in command line)
        :param o_key: Option key
        :return:
        """
        short_opt = ''
        if o_key in self._opts.keys():
            short_opt = f"-{self._opts[o_key].short}"
            if short_opt.endswith(':'):
                short_opt = short_opt[:-1]
        return short_opt

    def get_long_opt_name(self, o_key) -> str:
        """
        Get the long option name
        :param o_key: Option key
        :return:
        """
        long_opt = ''
        if o_key in self._opts.keys():
            long_opt = self._opts[o_key].long
            if long_opt.endswith('='):
                long_opt = long_opt[:-1]
        return long_opt

    def get_long_opt(self, o_key) -> str:
        """
        Get the long option (as appears in command line)
        :param o_key: Option key
        :return:
        """
        long_opt = self.get_long_opt_name(o_key)
        if len(long_opt):
            long_opt = f"--{long_opt}"
        return long_opt

    def usage(self):
        print(f'Usage: {self._name}')
        lines = []
        short_len = 0
        long_len = 0
        for o_key, opt_info in self._opts.items():
            if opt_info.has_value:
                short_opt = opt_info.short[:-1] + ' <value>'
            else:
                short_opt = opt_info.short
            if opt_info.long.has_value:
                long_opt = opt_info.long[:-1] + ' <value>'
            else:
                long_opt = opt_info.long
            short_len = max(short_len, len(short_opt))
            long_len = max(long_len, len(long_opt))
            lines.append((short_opt, long_opt, opt_info.desc))
        for line in lines:
            print(f' -{line[0]:{short_len}.{short_len}s}|--{line[1]:{long_len}.{long_len}s} : {line[2]}')
        print()

    def get_app_config(self, args: list):
        """
        Get the application config
        Note: command line options override config file options
        :param args: command line args
        :return:
        """
        try:
            opts, args = getopt.getopt(args, self.get_short_opts(), self.get_long_opts())
        except getopt.GetoptError as err:
            print(err)
            self.usage()
            sys.exit(2)

        if 'c' in self._opts:
            app_cfg_path = self._opts['c'].dfl_value    # default config file
        else:
            app_cfg_path = None

        cmd_line_args = {}
        # set default arguments
        for o_key, opt_info in {x: self._opts[x] for x in self._opts if x not in ['h', 'c']}.items():
            if opt_info.has_value:
                cmd_line_args[opt_info.long[:-1]] = opt_info.dfl_value
        # read command line
        exclude_opts = ['h', 'c']
        for opt, arg in opts:
            if opt == self.get_short_opt('h') or opt == self.get_long_opt('h'):
                self.usage()
                sys.exit()
            elif opt == self.get_short_opt('c') or opt == self.get_long_opt('c'):
                if app_cfg_path is not None:
                    app_cfg_path = arg  # use specified config file
            else:
                # read arguments from command line
                for o_key, opt_info in {x: self._opts[x] for x in self._opts if x not in exclude_opts}.items():
                    if opt == self.get_short_opt(o_key) or opt == self.get_long_opt(o_key):
                        exclude_opts.append(o_key)
                        raw_arg_key = opt_info.long[:-1] if opt_info.has_value else opt_info.long
                        cmd_line_args[raw_arg_key] = arg
                        break

        # load app config
        if app_cfg_path is not None:
            app_cfg = load_yaml(app_cfg_path)
        else:
            app_cfg = {}

        app_cfg.update(cmd_line_args)

        exclude_opts = ['h', 'c']
        for opt, arg in app_cfg.items():
            for o_key, opt_info in {x: self._opts[x] for x in self._opts if x not in exclude_opts}.items():
                if opt == self.get_long_opt_name(o_key):
                    exclude_opts.append(o_key)
                    if arg is not None:
                        if opt_info.type == 'int':
                            app_cfg[opt] = int(arg)
                        elif opt_info.type == 'float':
                            app_cfg[opt] = float(arg)
                        elif opt_info.type.startswith('date='):
                            splits = opt_info.type.split('=')
                            if len(splits) == 2:
                                app_cfg[opt] = datetime.datetime.strptime(arg, splits[1])
                    break

        return app_cfg


def load_yaml(yaml_path, key=None):
    """
    Load yaml file and return the configuration dictionary
    :param yaml_path: path to the yaml configuration file
    :param key: configuration key to return; default is all keys
    :return: configuration dictionary
    :rtype: dict
    """
    # verify path
    if not os.path.exists(yaml_path):
        raise ValueError(f'Invalid path: {yaml_path}')
    if not os.path.isfile(yaml_path):
        raise ValueError(f'Not a file path: {yaml_path}')

    with open(rf'{yaml_path}') as file:
        # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
        if pkg_resources.get_distribution("PyYAML").version.startswith('5'):
            # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            configs = yaml.load(file, Loader=yaml.FullLoader)
        else:
            configs = yaml.load(file)
        if key is not None:
            config_dict = configs[key]
        else:
            config_dict = configs

    return config_dict
