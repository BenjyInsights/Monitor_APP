# setup.py #
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2026  Benjamín Sánchez Calza
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Legacy setup.py — Configuration is primarily in pyproject.toml.

This file is provided for backward compatibility and to support
older setuptools workflows. Modern installations should use:

    pip install -e .

which will read pyproject.toml automatically.
"""

from setuptools import setup

if __name__ == "__main__":
    # Minimal setup() call — all metadata is in pyproject.toml
    setup()