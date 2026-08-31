# moniaenergy/utils/carbon_emissions.py #
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
carbon_emissions.py — Carbon intensity data from local CSV + optional external APIs.

Provides country and continent-level carbon intensity (gCO2/kWh) data with
support for optional external sources:
  - **CSV (default)**: Ember 2025 provisional + legacy 2022 continent data.
  - **Electricity Maps API**: Real-time carbon intensity via HTTP (requires API key).

Graceful fallback ensures that if the external API is unavailable, the CSV
data is used automatically without disrupting training.

Environment Variables
----------------------
ELECTRICITY_MAPS_API_KEY : str
    API key for Electricity Maps (optional).
    When set, enables real-time queries via HTTP.
    If unset or query fails, falls back to CSV data.

ELECTRICITY_MAPS_API_URL : str
    Override the Electricity Maps API endpoint.
    Default: https://api.electricitymap.org/v3/carbon-intensity/latest

CARBON_EMISSIONS_CSV_PATH : str
    Override the path to the 2025 CSV file.
    Default: data/2025_Country_Carbon_Emissions.csv

Examples
--------
>>> from moniaenergy.utils.carbon_emissions import CarbonEmissions
>>>
>>> # Query with automatic fallback: API → CSV → World Average
>>> factor = CarbonEmissions.get_country_factor("Spain")
>>> print(f"Spain: {factor} gCO2/kWh")

>>> # Force CSV-only (e.g., offline training):
>>> data = CarbonEmissions.get_ce_data(use_external_api=False)

>>> # Check which source was used
>>> CarbonEmissions.set_external_api_enabled(False)
"""

from __future__ import annotations

import csv
import json
import sys
import logging
import os
import urllib.error
import urllib.request
from typing import Optional, Any


logger = logging.getLogger(__name__)

# Absolute path to the data/ directory — resolved from this file's location so
# the module works regardless of the caller's current working directory.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_MODULE_DIR, "..", "..", "..", "data"))


class CarbonEmissions:
    """Provides country and continent-level carbon intensity data.

    Supports dual-source fallback:
      1. **Electricity Maps API** (optional, requires API key via env var).
         Real-time carbon intensity queries with millisecond latency.
      2. **CSV files** (always available).
         2025 country-level dataset (Ember) + legacy 2022 continent data.

    Graceful degradation: If API is unavailable, outdated, or misconfigured,
    falls back to CSV data automatically without blocking training.

    Attributes:
        _country_file_path (str): Path to 2025 country-level CSV.
        _continent_file_path (str): Path to legacy 2022 continent CSV.
        _external_api_enabled (bool): Whether to attempt API queries.
        _api_cache (dict): Cache of successful API queries (per process lifetime).
    """

    _country_file_path = os.environ.get(
        "CARBON_EMISSIONS_CSV_PATH",
        os.path.join(_DEFAULT_DATA_DIR, "2025_Country_Carbon_Emissions.csv")
    )
    _continent_file_path = os.path.join(_DEFAULT_DATA_DIR, "2022_Continent_Carbon_Emissions.csv")

    _country_data: Optional[dict] = None
    _continent_data: Optional[dict] = None
    _external_api_enabled: bool = bool(os.environ.get("ELECTRICITY_MAPS_API_KEY"))
    _api_url: str = os.environ.get(
        "ELECTRICITY_MAPS_API_URL",
        "https://api.electricitymap.org/v3/carbon-intensity/latest"
    )
    _api_key: Optional[str] = os.environ.get("ELECTRICITY_MAPS_API_KEY")
    _api_cache: dict[str, Optional[float]] = {}

    @classmethod
    def _read_csv(cls, path: str) -> dict:
        """Read a carbon intensity CSV into {entity: gCO2/kWh} dict.

        Parameters
        ----------
        path : str
            Path to CSV file with columns "Entity" and
            "Carbon intensity of electricity - gCO2/kWh".

        Returns
        -------
        dict
            Mapping {entity_name: carbon_intensity_gco2_per_kwh or None}.
        """
        data: dict = {}
        try:
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entity = row["Entity"].strip()
                    try:
                        factor = float(row["Carbon intensity of electricity - gCO2/kWh"])
                    except (ValueError, KeyError):
                        factor = None  # type: ignore[assignment]
                    data[entity] = factor
        except FileNotFoundError:
            pass  # caller decides whether to exit or fall back
        return data

    @classmethod
    def _query_electricity_maps_api(cls, country: str) -> Optional[float]:
        """Query real-time carbon intensity from Electricity Maps API.

        Parameters
        ----------
        country : str
            Country name or ISO-3166-1 alpha-2 code
            (e.g., "Spain", "ES", "United States", "US").

        Returns
        -------
        float or None
            Carbon intensity in gCO2/kWh, or None if API is unavailable,
            country is not found, or API query fails.
        """
        if not cls._external_api_enabled or not cls._api_key:
            return None

        # Check cache first
        cache_key = country.lower()
        if cache_key in cls._api_cache:
            return cls._api_cache[cache_key]

        try:
            headers = {
                "auth-token": cls._api_key,
                "Content-Type": "application/json",
            }
            req = urllib.request.Request(
                f"{cls._api_url}?countryCode={country}",
                headers=headers,
                method="GET"
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                # Expected response: {"carbonIntensity": <float>}
                intensity = data.get("carbonIntensity")
                if intensity is not None:
                    cls._api_cache[cache_key] = float(intensity)
                    logger.debug(
                        f"[CarbonEmissions] API hit for {country}: "
                        f"{intensity} gCO2/kWh"
                    )
                    return float(intensity)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Country not found in API
                logger.debug(
                    f"[CarbonEmissions] API: country '{country}' not found (404)"
                )
            else:
                logger.warning(
                    f"[CarbonEmissions] API error ({e.code}): {e.reason}"
                )
        except Exception as e:
            logger.debug(
                f"[CarbonEmissions] API query failed for '{country}': {e}"
            )

        # Cache None to avoid repeated failed queries
        cls._api_cache[cache_key] = None
        return None

    @classmethod
    def _load_country_data(cls, use_external_api: bool | None = None) -> dict:
        """Load country-level 2025 data, fall back to continent 2022 data.

        Parameters
        ----------
        use_external_api : bool, optional
            If True, attempts external API query mode for each country.
            If None (default), uses class-level setting.

        Returns
        -------
        dict
            {country: carbon_intensity or None}.
        """
        if cls._country_data is None:
            data = cls._read_csv(cls._country_file_path)
            if not data:
                # Fall back to legacy continent file
                data = cls._read_csv(cls._continent_file_path)
                if not data:
                    logger.warning(
                        f"[CarbonEmissions] Neither country-level "
                        f"({cls._country_file_path}) nor continent-level "
                        f"({cls._continent_file_path}) CSV was found. "
                        f"Carbon emissions will not be computed."
                    )
                    return {}
            cls._country_data = data
        return cls._country_data

    @classmethod
    def _load_continent_data(cls) -> dict:
        """Load legacy 2022 continent-level data (backward compatibility).

        Returns
        -------
        dict
            {continent: carbon_intensity or None}.
        """
        if cls._continent_data is None:
            data = cls._read_csv(cls._continent_file_path)
            if not data:
                data = cls._load_country_data()
            cls._continent_data = data
        return cls._continent_data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def set_external_api_enabled(cls, enabled: bool) -> None:
        """Enable/disable external API queries (Electricity Maps).

        Parameters
        ----------
        enabled : bool
            If True and API key is set, API queries are attempted.
            If False, only CSV data is used.
        """
        cls._external_api_enabled = enabled
        logger.debug(f"[CarbonEmissions] External API: {enabled}")

    @classmethod
    def get_ce_data(
        cls, use_external_api: bool | None = None
    ) -> dict:
        """Return country-level carbon intensity data.

        Queries attempt the external API first (if enabled), then fall back
        to the 2025 country-level CSV (Ember), then 2022 continent CSV.

        Parameters
        ----------
        use_external_api : bool, optional
            Override the global external API setting for this query.
            If None (default), uses the class setting.

        Returns
        -------
        dict
            {entity_name: gCO2/kWh (float or None)}.
        """
        return cls._load_country_data(use_external_api=use_external_api)

    @classmethod
    def get_country_factor(
        cls, country: str, use_external_api: bool | None = None
    ) -> Optional[float]:
        """Return the carbon intensity factor for a specific country.

        Attempts in order:
          1. Electricity Maps API (if enabled + API key available).
          2. 2025 country-level CSV (Ember).
          3. 2022 continent-level CSV (legacy).
          4. "World Average" fallback.

        Parameters
        ----------
        country : str
            Country name as used in CSV or ISO code for API
            (e.g., "Spain", "ES", "United States", "US").
        use_external_api : bool, optional
            Override global external API setting.

        Returns
        -------
        float or None
            Carbon intensity in gCO2/kWh, or None if all sources fail.
        """
        if use_external_api is None:
            use_external_api = cls._external_api_enabled

        # Attempt external API first
        if use_external_api:
            api_result = cls._query_electricity_maps_api(country)
            if api_result is not None:
                logger.debug(
                    f"[CarbonEmissions] {country}: {api_result} gCO2/kWh "
                    f"(Electricity Maps API)"
                )
                return api_result

        # Fall back to CSV data
        data = cls._load_country_data()
        factor = data.get(country)
        if factor is None and country != "World Average":
            # Try world average as ultimate fallback
            factor = data.get("World Average")

        if factor is not None:
            source = "CSV"
            if factor is not None:
                logger.debug(
                    f"[CarbonEmissions] {country}: {factor} gCO2/kWh ({source})"
                )

        return factor