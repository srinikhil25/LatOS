"""Tests for `HiokiLoggerCsvParser`.

The fixtures are built from the example printed in Appendix 3, "Text File (CSV)
Internal Format", of the Memory HiLogger instruction manual, so what is being
tested is the documented format rather than a format guessed at from one file.

Three of these tests exist because of traps in that format specifically:

* a channel on the 100 mV range still writes its samples in volts, so a parser
  that believes the range instead of the bracketed unit is wrong by 1000x;
* a pulse count column is written `P-1[c]` while Celsius is `CH-7[C]`, which
  differ by letter case alone;
* a sample row opens with two floats, which is all the CasaXPS sniffer checks,
  so without a guard a voltage trace is ingested as an XPS spectrum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from latos.core.enums import Severity, Technique
from latos.ingestion.parsers.hioki_logger_csv import (
    HiokiLoggerCsvParser,
    is_hioki_logger_header,
)
from latos.ingestion.parsers.xps_casaxps_csv import CasaXpsCsvParser
from latos.ingestion.registry import default_registry

_ALM = ",".join(["0"] * 15)

_FULL_COLUMNS = (
    '"Time","CH-1[V]","CH-2[V]","CH-7[C]","CH-8[C]","P-1[c]","P-3[r/s]",'
    '"ALM-CH1","ALM-CH2","ALM-CH3","ALM-CH4","ALM-CH5","ALM-CH6","ALM-CH7",'
    '"ALM-CH8","ALM-CH9","ALM-CH10","ALM-PLS1","ALM-PLS2","ALM-PLS3",'
    '"ALM-PLS4","ALM-OUT","Event",'
)


def _header(*, comments: tuple[str, str, str] = ("", "", ""), scaling: str = "Off") -> list[str]:
    cell, tc_a, tc_b = comments
    return [
        '"File name","WAVE0001.CSV","V 1.00"',
        '"Title comment","x=0.50 run 3"',
        '"Trigger Time","\'26-10-02 09:14:03"',
        '"Ch","Mode","Range","Comment","Scaling","Ratio","Offset"',
        f'"CH-1","Voltage","100mV","{cell}","{scaling}","-","-"',
        '"CH-2","Voltage","1V","","Off","-","-"',
        f'"CH-7","Tc","2000 C","{tc_a}","Off","-","-"',
        f'"CH-8","Tc","2000 C","{tc_b}","Off","-","-"',
        '"P-1","Count","1000000000c","","Off","-","-"',
        '"P-3","Revolve","5000r/s","","Off","-","-"',
        '"ALM","Alarm","","",',
        _FULL_COLUMNS,
    ]


def _rows(
    n: int = 12,
    *,
    event_on_first: bool = True,
    tc7: float = 35.0,
    tc8: float = 30.0,
) -> list[str]:
    """`n` samples at 1 s. CH-1 climbs in VOLTS, CH-7 and CH-8 are thermocouples.

    The default puts the hotter reading on CH-7, the lower-numbered channel, so
    the positional fallback happens to land the right way round. `tc7`/`tc8` are
    swapped in the tests that need the physical layout to disagree with channel
    order, which is the case where the comments have to do the work.
    """
    out = []
    for i in range(n):
        volts = 0.001 * (i + 1)  # 1 mV, 2 mV, ... written as volts
        mark = "1" if (i == 0 and event_on_first) else "0"
        out.append(
            f"{float(i):.9E},{volts:.5E},-5.05000E-03,"
            f"{tc7:.5E},{tc8:.5E},"
            f"0.000000000E+00,0.000000000E+00,"
            f"{_ALM},{mark},"
        )
    return out


def _write(tmp_path: Path, lines: list[str], name: str = "WAVE0001.CSV") -> Path:
    f = tmp_path / name
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _full(tmp_path: Path, **kw) -> Path:
    return _write(tmp_path, _header(**kw) + _rows())


# ─── Class metadata ─────────────────────────────────────────────────
class TestClassMetadata:
    def test_name(self):
        assert HiokiLoggerCsvParser.name == "hioki-logger-csv"

    def test_version(self):
        assert HiokiLoggerCsvParser.version == "1.0.0"

    def test_technique(self):
        assert HiokiLoggerCsvParser.technique is Technique.THERMOELECTRIC

    def test_extensions_cover_both_names_the_instrument_uses(self):
        # The same text export is written .TXT or .CSV depending on a
        # System-screen setting, so refusing .txt would drop half of them.
        assert HiokiLoggerCsvParser.supported_extensions == (".csv", ".txt")


# ─── Recognition ────────────────────────────────────────────────────
class TestRecognition:
    def test_full_export_is_unambiguous(self, tmp_path):
        assert HiokiLoggerCsvParser().can_parse(_full(tmp_path)) == 1.0

    def test_header_suppressed_export_scores_below_certain(self, tmp_path):
        # Without the key/value block the only evidence is the CH-n[unit]
        # columns: strong, but not unique, so it must not claim certainty.
        f = _write(tmp_path, [_FULL_COLUMNS, *_rows()], name="bare.csv")
        assert HiokiLoggerCsvParser().can_parse(f) == 0.8

    def test_txt_extension_is_accepted(self, tmp_path):
        f = _write(tmp_path, _header() + _rows(), name="WAVE0001.TXT")
        assert HiokiLoggerCsvParser().can_parse(f) == 1.0

    def test_unrelated_csv_is_declined(self, tmp_path):
        f = _write(tmp_path, ["285.0,1200.0", "284.9,1250.0", "284.8,1310.0"], "x.csv")
        assert HiokiLoggerCsvParser().can_parse(f) == 0.0

    def test_wrong_extension_is_declined(self, tmp_path):
        f = _write(tmp_path, _header() + _rows(), name="WAVE0001.dat")
        assert HiokiLoggerCsvParser().can_parse(f) == 0.0

    def test_missing_file_returns_zero_rather_than_raising(self, tmp_path):
        assert HiokiLoggerCsvParser().can_parse(tmp_path / "absent.csv") == 0.0

    def test_predicate_agrees_with_can_parse(self, tmp_path):
        lines = (_header() + _rows())[:30]
        assert is_hioki_logger_header([f"{ln}\n" for ln in lines]) is True

    def test_predicate_rejects_an_xps_export(self):
        assert is_hioki_logger_header(["285.0,1200.0\n", "284.9,1250.0\n"]) is False


# ─── Dispatch ───────────────────────────────────────────────────────
class TestDispatch:
    def test_registry_routes_to_the_logger_parser(self, tmp_path):
        match = default_registry().find_parser(_full(tmp_path))
        assert match is not None
        assert match.parser.name == "hioki-logger-csv"
        assert match.confidence == 1.0

    def test_registry_routes_the_header_suppressed_form_too(self, tmp_path):
        f = _write(tmp_path, [_FULL_COLUMNS, *_rows()], name="bare.csv")
        match = default_registry().find_parser(f)
        assert match is not None
        assert match.parser.name == "hioki-logger-csv"

    def test_casaxps_declines_a_logger_trace(self, tmp_path):
        # A sample row opens with elapsed time and a voltage, which satisfies
        # the XPS numeric-pair sniff. Without the guard this scores 1.0 and
        # ties, and a voltage trace becomes a spectrum.
        assert CasaXpsCsvParser().can_parse(_full(tmp_path)) == 0.0

    def test_casaxps_declines_the_header_suppressed_form(self, tmp_path):
        f = _write(tmp_path, [_FULL_COLUMNS, *_rows()], name="bare.csv")
        assert CasaXpsCsvParser().can_parse(f) == 0.0


# ─── Arrays ─────────────────────────────────────────────────────────
class TestArrays:
    def test_voltage_is_converted_from_the_declared_unit_not_the_range(self, tmp_path):
        # CH-1 is on the 100mV range but writes volts. 0.001 V is 1 mV, and a
        # parser that trusted the range would report 0.001.
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.arrays["voltage_mv"][0] == pytest.approx(1.0)
        assert data.arrays["voltage_mv"][-1] == pytest.approx(12.0)

    def test_millivolt_columns_are_left_alone(self, tmp_path):
        header = _header()
        header[-1] = _FULL_COLUMNS.replace('"CH-1[V]"', '"CH-1[mV]"')
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, header + _rows()))
        assert data.arrays["voltage_mv"][0] == pytest.approx(0.001)

    def test_temperatures_are_both_present(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.arrays["t_hot_c"][0] == pytest.approx(35.0)
        assert data.arrays["t_cold_c"][0] == pytest.approx(30.0)

    def test_array_names_match_what_the_slope_analyzer_reads(self, tmp_path):
        # ThermovoltageSlopeAnalyzer accepts the raw-ramp form by these exact
        # names, so a parsed trace is analysable with no further wiring.
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert {"t_hot_c", "t_cold_c", "voltage_mv"} <= set(data.arrays)

    def test_time_is_elapsed_seconds(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.arrays["time_s"][0] == pytest.approx(0.0)
        assert data.arrays["time_s"][-1] == pytest.approx(11.0)

    def test_event_marks_are_kept(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.arrays["event_mark"][0] == pytest.approx(1.0)
        assert data.metadata["event_sample_indices"] == [0]
        assert data.metadata["event_times_s"] == [0.0]

    def test_no_event_array_when_nothing_was_marked(self, tmp_path):
        f = _write(tmp_path, _header() + _rows(event_on_first=False))
        assert "event_mark" not in HiokiLoggerCsvParser().parse(f).arrays


# ─── Channel roles ──────────────────────────────────────────────────
class TestChannelRoles:
    def test_comments_override_channel_order(self, tmp_path):
        # The hot electrode is wired to CH-8 and labelled there, so channel
        # order disagrees with the physical layout. Reading the labels gives
        # dT = +5 K; reading the order would give -5 K, inverting the sign of S
        # for the whole campaign with nothing downstream able to catch it.
        lines = _header(comments=("cell", "cold junction", "hot plate"))
        lines += _rows(tc7=30.0, tc8=35.0)
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, lines))
        assert data.metadata["channel_roles"] == {
            "cell": "CH-1",
            "hot": "CH-8",
            "cold": "CH-7",
        }
        assert data.arrays["t_hot_c"][0] == pytest.approx(35.0)
        assert data.arrays["t_cold_c"][0] == pytest.approx(30.0)
        delta = data.arrays["t_hot_c"] - data.arrays["t_cold_c"]
        assert delta[0] == pytest.approx(5.0)

    def test_comment_assignment_raises_no_warning(self, tmp_path):
        f = _full(tmp_path, comments=("cell", "cold", "hot"))
        data = HiokiLoggerCsvParser().parse(f)
        assert data.metadata["channel_roles_positional"] == []
        assert not [i for i in data.issues if i.field == "channel_roles"]

    def test_positional_fallback_is_reported(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.metadata["channel_roles_positional"] == ["cell", "hot", "cold"]
        warned = [i for i in data.issues if i.field == "channel_roles"]
        assert len(warned) == 1
        assert warned[0].severity is Severity.WARNING

    def test_pulse_column_is_not_mistaken_for_a_thermocouple(self, tmp_path):
        # `P-1[c]` (a pulse count) and `CH-7[C]` (Celsius) differ by case only.
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        roles = data.metadata["channel_roles"]
        assert "P-1" not in roles.values()
        assert "P-3" not in roles.values()


# ─── Refusals and warnings ──────────────────────────────────────────
class TestRefusals:
    def test_scaling_enabled_is_an_error(self, tmp_path):
        # With scaling on, the values are in some operator-chosen unit and the
        # bracketed unit no longer describes them.
        f = _full(tmp_path, comments=("cell", "hot", "cold"), scaling="On")
        errors = [i for i in HiokiLoggerCsvParser().parse(f).issues if i.field == "scaling"]
        assert len(errors) == 1
        assert errors[0].severity is Severity.ERROR

    def test_no_column_header_is_an_error(self, tmp_path):
        f = _write(tmp_path, _header()[:-1] + _rows(), name="broken.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.arrays == {}
        assert any(i.severity is Severity.ERROR for i in data.issues)

    def test_header_without_samples_is_an_error(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, _header()))
        assert data.arrays == {}
        assert any("sample" in i.message.lower() for i in data.issues)

    def test_missing_voltage_column_is_an_error_but_time_survives(self, tmp_path):
        header = _header()
        header[-1] = '"Time","CH-7[C]","CH-8[C]","Event",'
        rows = [f"{float(i):.6E},3.00000E+01,3.50000E+01,0," for i in range(12)]
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, header + rows))
        assert "voltage_mv" not in data.arrays
        assert data.arrays["time_s"].size == 12
        assert any(i.field == "voltage_mv" and i.severity is Severity.ERROR for i in data.issues)

    def test_single_thermocouple_warns_without_losing_the_voltage(self, tmp_path):
        header = _header()
        header[-1] = '"Time","CH-1[V]","CH-7[C]","Event",'
        rows = [f"{float(i):.6E},1.00000E-03,3.00000E+01,0," for i in range(12)]
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, header + rows))
        assert "voltage_mv" in data.arrays
        assert "t_hot_c" not in data.arrays
        assert any(i.field == "t_hot_c" and i.severity is Severity.WARNING for i in data.issues)

    def test_unreadable_numbers_are_counted_not_fatal(self, tmp_path):
        rows = _rows()
        rows[4] = rows[4].replace("3.50000E+01", "oops")
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, _header() + rows))
        assert data.arrays["time_s"].size == 12
        assert np.isnan(data.arrays["t_hot_c"][4])


# ─── Locale and layout variants ─────────────────────────────────────
class TestVariants:
    def test_semicolon_delimiter_with_comma_decimals(self, tmp_path):
        # The instrument's CSV saving settings choose both characters.
        lines = [
            '"Time";"CH-1[V]";"CH-7[C]";"CH-8[C]";"Event";',
            *[f"{i};0,00{i + 1};35,0;30,0;0;" for i in range(12)],
        ]
        f = _write(tmp_path, lines, name="euro.csv")
        parser = HiokiLoggerCsvParser()
        assert parser.can_parse(f) == 0.8
        data = parser.parse(f)
        assert data.metadata["delimiter"] == ";"
        assert data.arrays["voltage_mv"][0] == pytest.approx(1.0)
        assert data.arrays["t_hot_c"][0] == pytest.approx(35.0)

    def test_tab_separated_text_export_is_read(self, tmp_path):
        # Logger Utility's save dialog offers Text (Tab separated), written .TXT.
        # Before this was handled the file was simply unclaimed.
        lines = [
            "\t".join(['"Time"', '"CH-1[V]"', '"CH-7[C]"', '"CH-8[C]"', '"Event"']),
            *["\t".join([str(i), f"0.00{i + 1}", "35.0", "30.0", "0"]) for i in range(12)],
        ]
        f = _write(tmp_path, lines, name="WAVE0001.TXT")
        parser = HiokiLoggerCsvParser()
        assert parser.can_parse(f) == 0.8
        data = parser.parse(f)
        assert data.metadata["delimiter"] == "\t"
        assert data.arrays["voltage_mv"][0] == pytest.approx(1.0)
        assert data.arrays["t_hot_c"][0] == pytest.approx(35.0)

    def test_short_rows_do_not_abort_the_parse(self, tmp_path):
        rows = _rows()
        rows[3] = "3.000000000E+00,4.00000E-03"
        data = HiokiLoggerCsvParser().parse(_write(tmp_path, _header() + rows))
        assert data.arrays["time_s"].size == 12
        assert np.isnan(data.arrays["t_hot_c"][3])


# ─── Time Axis Format ───────────────────────────────────────────────
class TestTimeAxisFormat:
    """The save dialog's Time Axis Format has four settings; one is what we want.

    Second is correct. Absolute Time and Relative Time are textual and are
    converted here rather than rejected. Point writes the sample index, which is
    identical to seconds at a 1 s interval and silently wrong at any other, so it
    can only be reported as a suspicion.
    """

    @staticmethod
    def _with_time(column: list[str]) -> list[str]:
        header = ['"Time","CH-1[V]","CH-7[C]","CH-8[C]","Event",']
        return header + [f"{stamp},0.00{i + 1},35.0,30.0,0," for i, stamp in enumerate(column)]

    def test_absolute_time_is_rebased_to_elapsed_seconds(self, tmp_path):
        stamps = [f'"2026-10-02 09:14:{s:02d}"' for s in range(12)]
        f = _write(tmp_path, self._with_time(stamps), name="abs.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.metadata["time_axis_format"] == "absolute"
        assert data.arrays["time_s"][0] == pytest.approx(0.0)
        assert data.arrays["time_s"][-1] == pytest.approx(11.0)
        assert any(i.severity is Severity.INFO and i.field == "time_s" for i in data.issues)

    def test_relative_time_is_converted_to_seconds(self, tmp_path):
        stamps = [f'"0:00:{s:02d}"' for s in range(12)]
        f = _write(tmp_path, self._with_time(stamps), name="rel.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.metadata["time_axis_format"] == "relative"
        assert data.arrays["time_s"][-1] == pytest.approx(11.0)

    def test_relative_time_handles_hours_and_minutes(self, tmp_path):
        stamps = ['"0:00:00"', '"0:01:30"', '"1:00:00"'] + [f'"1:00:{s:02d}"' for s in range(1, 10)]
        f = _write(tmp_path, self._with_time(stamps), name="rel2.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.arrays["time_s"][1] == pytest.approx(90.0)
        assert data.arrays["time_s"][2] == pytest.approx(3600.0)

    def test_a_seconds_column_with_a_unit_suffix_is_read(self, tmp_path):
        # Clearing the dialog's "Save with a decimal" box leaves the unit on.
        stamps = [f"{i}s" for i in range(12)]
        f = _write(tmp_path, self._with_time(stamps), name="unit.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.metadata["time_axis_format"] == "seconds"
        assert data.arrays["time_s"][-1] == pytest.approx(11.0)

    def test_integer_step_column_is_reported_as_possibly_an_index(self, tmp_path):
        # 0,1,2,3... is either a 1 s interval or Point format. Unresolvable from
        # the file, so it is recorded rather than silently assumed.
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.metadata["time_column_could_be_sample_index"] is True
        flagged = [i for i in data.issues if i.field == "time_s"]
        assert len(flagged) == 1
        assert flagged[0].severity is Severity.INFO
        assert "Point" in flagged[0].message

    def test_a_non_unit_interval_is_unambiguous(self, tmp_path):
        # Stepping by 2 cannot be a sample index, so there is nothing to report.
        stamps = [f"{2.0 * i:.6E}" for i in range(12)]
        f = _write(tmp_path, self._with_time(stamps), name="two.csv")
        data = HiokiLoggerCsvParser().parse(f)
        assert data.metadata["time_column_could_be_sample_index"] is False
        assert not [i for i in data.issues if i.field == "time_s"]


# ─── Metadata and features ──────────────────────────────────────────
class TestMetadataAndFeatures:
    def test_trigger_time_is_read_and_flagged_as_assumed_utc(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.measured_at is not None
        assert data.measured_at.year == 2026
        assert data.measured_at.tzinfo is not None
        # The file states no zone, so the flag says the zone was assumed.
        assert data.metadata["trigger_time_zone_assumed_utc"] is True
        assert data.metadata["trigger_time_text"] == "'26-10-02 09:14:03"

    def test_instrument_and_title_are_recorded(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.instrument == "HIOKI Memory HiLogger"
        assert data.metadata["title_comment"] == "x=0.50 run 3"
        assert data.metadata["file_name"] == "WAVE0001.CSV"

    def test_channel_table_is_kept_verbatim(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        table = {row["channel"]: row for row in data.metadata["channel_table"]}
        assert table["CH-1"]["range"] == "100mV"
        assert table["CH-7"]["mode"] == "Tc"

    def test_features_describe_the_run(self, tmp_path):
        data = HiokiLoggerCsvParser().parse(_full(tmp_path))
        assert data.features["n_points"] == pytest.approx(12.0)
        assert data.features["duration_s"] == pytest.approx(11.0)
        assert data.features["sampling_interval_s"] == pytest.approx(1.0)
        assert data.features["voltage_mv_final"] == pytest.approx(12.0)
        assert data.features["delta_t_k_median"] == pytest.approx(5.0)

    def test_parser_identity_round_trips(self, tmp_path):
        parser = HiokiLoggerCsvParser()
        data = parser.parse(_full(tmp_path))
        assert data.parser_name == parser.name
        assert data.parser_version == parser.version
