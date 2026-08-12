# utils/cpu_info.py
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

import cpuinfo

class CPUInfo:
    """
    Class to get the CPU information.

    The information is obtained using the cpuinfo library.
    """
    _cpu_info = cpuinfo.get_cpu_info()
    _cpu_vendor = _cpu_info.get("vendor_id_raw")
    _cpu_cores = int(_cpu_info.get("count") / 2)
    _cpu_threads = _cpu_info.get("count")

    @classmethod
    def get_cpu_vendor(cls) -> str:
        """
        Get the CPU vendor.

        Returns:
            str: The CPU vendor.
        """
        return cls._cpu_vendor
    
    @classmethod
    def get_cpu_cores(cls) -> int:
        """
        Get the number of CPU cores.

        Returns:
            int: The number of CPU cores.
        """
        return cls._cpu_cores
    
    @classmethod
    def get_cpu_threads(cls) -> int:
        """
        Get the number of CPU threads.

        Returns:
            int: The number of CPU threads.
        """
        return cls._cpu_threads