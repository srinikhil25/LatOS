"""Tests for `latos.ingestion.labeling.anomalies.flag_anomalies`."""

from __future__ import annotations

from pathlib import Path

from latos.core.enums import FileRole, Technique
from latos.core.models import FileRef, Measurement, Sample, new_id, utc_now
from latos.ingestion.labeling.anomalies import flag_anomalies


def _meas(sample_id: str, filename: str) -> Measurement:
    return Measurement(
        id=new_id(),
        sample_id=sample_id,
        technique=Technique.HALL,
        instrument=None,
        measured_at=None,
        parsed_at=utc_now(),
        parser_version="1.0.0",
        files=(
            FileRef(
                path=Path(f"D:/proj/{filename}"),
                sha256="0" * 64,
                size_bytes=1,
                role=FileRole.RAW,
                scanned_at=utc_now(),
            ),
        ),
        issues=(),
        parsed_data_path=None,
    )


def _sample(name: str, *filenames: str) -> Sample:
    sid = new_id()
    measurements = tuple(_meas(sid, fn) for fn in filenames) or (
        _meas(sid, f"{name}.dat"),
    )
    return Sample(
        id=sid, project_id=new_id(), canonical_name=name, measurements=measurements
    )


class TestMixedSamples:
    def test_session_folder_flagged_with_related(self):
        samples = [
            _sample("CS-1", "CS-1.dat"),
            _sample("CS-3", "CS-3.dat"),
            _sample("CS-5", "CS-5.dat"),
            # A session bucket whose files belong to the three above.
            _sample("Divyamahalakshmi_07042025", "CS-1.xls", "CS-3.xls", "CS-5.xls"),
        ]
        out = flag_anomalies(samples)
        flagged = {a.sample_name: a for a in out}
        assert "Divyamahalakshmi_07042025" in flagged
        a = flagged["Divyamahalakshmi_07042025"]
        assert a.kind == "mixed_samples"
        assert set(a.related) == {"CS-1", "CS-3", "CS-5"}

    def test_real_samples_not_flagged(self):
        samples = [
            _sample("CS-1", "CS-1.dat"),
            _sample("CS-3", "CS-3.dat"),
            _sample("CS-5", "CS-5.dat"),
            _sample("Divyamahalakshmi_07042025", "CS-1.xls", "CS-3.xls", "CS-5.xls"),
        ]
        flagged = {a.sample_name for a in flag_anomalies(samples)}
        assert "CS-1" not in flagged
        assert "CS-3" not in flagged

    def test_single_match_is_not_enough(self):
        # One filename matching another sample is a coincidence, not a bucket.
        samples = [
            _sample("CS-1", "CS-1.dat"),
            _sample("SessionX", "CS-1.xls", "random.dat"),
        ]
        flagged = {a.sample_name for a in flag_anomalies(samples)}
        assert "SessionX" not in flagged


class TestNonSampleName:
    def test_generic_folder_name_flagged(self):
        samples = [_sample("Images", "Image_001.tif", "Image_002.tif")]
        out = flag_anomalies(samples)
        assert len(out) == 1
        assert out[0].kind == "non_sample_name"

    def test_date_name_flagged_when_not_mixed(self):
        # A date-named sample whose files do NOT match other samples is
        # still flagged on its name alone.
        samples = [_sample("20250704_batch", "scan1.dat", "scan2.dat")]
        out = flag_anomalies(samples)
        assert len(out) == 1
        assert out[0].kind == "non_sample_name"

    def test_real_sample_not_flagged(self):
        samples = [_sample("CSCBI-3", "CSCBI-3.xrdml")]
        assert flag_anomalies(samples) == []

    def test_mixed_wins_over_name(self):
        # Divyamahalakshmi_07042025 matches both detectors; only one
        # anomaly is emitted, and it's the more actionable mixed_samples.
        samples = [
            _sample("CS-1", "CS-1.dat"),
            _sample("CS-3", "CS-3.dat"),
            _sample("Divyamahalakshmi_07042025", "CS-1.xls", "CS-3.xls"),
        ]
        flagged = [a for a in flag_anomalies(samples) if a.sample_name.startswith("Divya")]
        assert len(flagged) == 1
        assert flagged[0].kind == "mixed_samples"
