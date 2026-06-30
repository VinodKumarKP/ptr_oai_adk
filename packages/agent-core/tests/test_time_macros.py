"""Tests for time_macros.py"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from oai_agent_core.macros.time_macros import (
    macro_date,
    macro_now,
    macro_timestamp,
    macro_business_days_from,
)


class TestMacroDate:
    def test_default_format(self):
        result = macro_date()
        # Should match YYYY-MM-DD format
        datetime.strptime(result, "%Y-%m-%d")

    def test_custom_format(self):
        result = macro_date("%d/%m/%Y")
        datetime.strptime(result, "%d/%m/%Y")

    def test_offset_days_positive(self):
        result = macro_date("%Y-%m-%d", "1")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert result == tomorrow

    def test_offset_days_negative(self):
        result = macro_date("%Y-%m-%d", "-1")
        yesterday = (datetime.now() + timedelta(days=-1)).strftime("%Y-%m-%d")
        assert result == yesterday

    def test_invalid_offset(self):
        result = macro_date("%Y-%m-%d", "not_a_number")
        assert "Error" in result


class TestMacroNow:
    def test_default_iso(self):
        result = macro_now()
        # Should be parseable as ISO datetime
        datetime.fromisoformat(result)

    def test_custom_format(self):
        result = macro_now("%H:%M:%S")
        datetime.strptime(result, "%H:%M:%S")


class TestMacroTimestamp:
    def test_default_unix(self):
        result = macro_timestamp()
        # Should be integer unix timestamp
        int(result)

    def test_offset_seconds(self):
        now_ts = int(datetime.now().timestamp())
        result = int(macro_timestamp("3600"))
        assert abs(result - (now_ts + 3600)) < 5

    def test_iso_format(self):
        result = macro_timestamp("0", "iso")
        datetime.fromisoformat(result)

    def test_ms_format(self):
        result = macro_timestamp("0", "ms")
        ts = int(result)
        now_ms = int(datetime.now().timestamp() * 1000)
        assert abs(ts - now_ms) < 5000

    def test_invalid_offset(self):
        result = macro_timestamp("bad_offset")
        assert "Error" in result


class TestMacroBusinessDaysFrom:
    def test_from_specific_date(self):
        result = macro_business_days_from("2024-01-15", "5")
        # 2024-01-15 is a Monday; 5 business days = Jan 22
        assert result == "2024-01-22"

    def test_from_today(self):
        result = macro_business_days_from("today", "0")
        # 0 business days from today = today
        today = datetime.now().strftime("%Y-%m-%d")
        assert result == today

    def test_negative_count(self):
        result = macro_business_days_from("2024-01-22", "-5")
        assert result == "2024-01-15"

    def test_custom_format(self):
        result = macro_business_days_from("2024-01-15", "0", "%d/%m/%Y")
        assert result == "15/01/2024"

    def test_missing_args(self):
        result = macro_business_days_from("2024-01-01")
        assert "Error" in result

    def test_invalid_count(self):
        result = macro_business_days_from("2024-01-01", "bad")
        assert "Error" in result

    def test_invalid_date(self):
        result = macro_business_days_from("not-a-date", "5")
        assert "Error" in result
