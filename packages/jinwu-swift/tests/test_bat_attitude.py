import numpy as np
import pytest
import astropy.units as u

from jinwu.swift.bat.attitude import Attitude
from jinwu.swift.bat.bat_observation import BATObservation
import jinwu.swift.bat.bat_observation as bat_observation_module


@pytest.fixture
def attitude():
    return Attitude(
        time=np.array([10.0, 11.0, 12.0]) * u.s,
        ra=np.array([359.0, 0.0, 1.0]) * u.deg,
        dec=np.array([-10.0, 0.0, 10.0]) * u.deg,
        roll=np.array([359.0, 0.0, 1.0]) * u.deg,
    )


@pytest.mark.parametrize("query_time", [9.0, 13.0])
def test_pointing_at_rejects_times_outside_attitude_coverage(attitude, query_time):
    with pytest.raises(ValueError, match="outside BAT attitude coverage"):
        attitude.pointing_at(query_time)


@pytest.mark.parametrize("query_time", [9.0, 13.0])
def test_roll_at_rejects_times_outside_attitude_coverage(attitude, query_time):
    with pytest.raises(ValueError, match="outside BAT attitude coverage"):
        attitude.roll_at(query_time)


def test_attitude_queries_interpolate_inside_and_include_coverage_endpoints(attitude):
    start_ra, start_dec = attitude.pointing_at(10.0)
    mid_ra, mid_dec = attitude.pointing_at(10.5)
    end_ra, end_dec = attitude.pointing_at(12.0)

    assert start_ra.to_value(u.deg) == pytest.approx(359.0)
    assert start_dec.to_value(u.deg) == pytest.approx(-10.0)
    assert mid_ra.to_value(u.deg) == pytest.approx(359.5)
    assert mid_dec.to_value(u.deg) == pytest.approx(-5.0)
    assert end_ra.to_value(u.deg) == pytest.approx(1.0)
    assert end_dec.to_value(u.deg) == pytest.approx(10.0)
    assert attitude.roll_at(10.5).to_value(u.deg) == pytest.approx(359.5)
    assert attitude.roll_at(12.0).to_value(u.deg) == pytest.approx(1.0)


def test_parse_sao_accepts_scalar_ra_dec_columns_without_pointing_matrix():
    parsed = Attitude._parse_sao(
        {
            "TIME": np.array([10.0, 11.0]) * u.s,
            "RA": np.array([120.0, 121.0]) * u.deg,
            "DEC": np.array([-20.0, -19.0]) * u.deg,
            "ROLL": np.array([15.0, 16.0]) * u.deg,
        }
    )

    ra, dec = parsed.pointing_at(10.5)
    assert ra.to_value(u.deg) == pytest.approx(120.5)
    assert dec.to_value(u.deg) == pytest.approx(-19.5)


def test_parse_sao_accepts_pointing_rows_as_fallback():
    parsed = Attitude._parse_sao(
        {
            "TIME": np.array([10.0, 11.0]) * u.s,
            "POINTING": np.array([[120.0, -20.0, 15.0], [121.0, -19.0, 16.0]]) * u.deg,
        }
    )

    ra, dec = parsed.pointing_at(10.5)
    assert ra.to_value(u.deg) == pytest.approx(120.5)
    assert dec.to_value(u.deg) == pytest.approx(-19.5)
    assert parsed.roll_at(10.5).to_value(u.deg) == pytest.approx(15.5)


def test_parse_sao_fails_clearly_without_pointing_columns():
    with pytest.raises(ValueError, match="SAO attitude data must include"):
        Attitude._parse_sao({"TIME": np.array([10.0, 11.0]) * u.s})


def test_parse_sao_fails_clearly_without_any_roll_source():
    with pytest.raises(ValueError, match="PA_PNT, ROLL, or a POINTING third column"):
        Attitude._parse_sao(
            {
                "TIME": np.array([10.0, 11.0]) * u.s,
                "RA": np.array([120.0, 121.0]) * u.deg,
                "DEC": np.array([-20.0, -19.0]) * u.deg,
            }
        )


@pytest.mark.parametrize(
    "error_type, message",
    [
        (ValueError, "outside BAT attitude coverage"),
        (RuntimeError, "interpolation failed"),
    ],
)
def test_bat_observation_propagates_attitude_errors_without_using_midpoint(
    tmp_path, monkeypatch, error_type, message
):
    class FailingAttitude:
        time = np.array([10.0, 11.0, 12.0]) * u.s
        ra = np.array([10.0, 20.0, 30.0]) * u.deg
        dec = np.array([-10.0, 0.0, 10.0]) * u.deg

        def pointing_at(self, met_time):
            raise error_type(message)

    class FakeAttitude:
        @classmethod
        def from_file(cls, attitude_file):
            return FailingAttitude()

    monkeypatch.setattr(bat_observation_module, "Attitude", FakeAttitude)
    observation = BATObservation.__new__(BATObservation)
    observation.srctime = 13.0 * u.s

    with pytest.raises(error_type, match=message):
        observation._load_from_attitude(tmp_path / "synthetic.mkf")

    assert not hasattr(observation, "pointing")


def test_attitude_query_converts_quantity_to_sample_time_unit(attitude):
    ra, dec = attitude.pointing_at(10_500 * u.ms)
    assert ra.to_value(u.deg) == pytest.approx(359.5)
    assert dec.to_value(u.deg) == pytest.approx(-5.0)
    assert attitude.roll_at(10_500 * u.ms).to_value(u.deg) == pytest.approx(359.5)
    with pytest.raises(ValueError, match="convertible to s"):
        attitude.pointing_at(10.5 * u.m)


def test_state_flag_queries_reject_extrapolation_and_preserve_unknown_flags(attitude):
    assert attitude.in_saa_at(13.0 * u.s) is None
    assert attitude.is_settled_at(13.0 * u.s) is None

    with_flags = Attitude(
        time=np.array([10.0, 11.0, 12.0]) * u.s,
        ra=np.array([10.0, 11.0, 12.0]) * u.deg,
        dec=np.array([-1.0, 0.0, 1.0]) * u.deg,
        roll=np.array([20.0, 21.0, 22.0]) * u.deg,
        in_saa=np.array([False, True, False]),
        is_settled=np.array([True, True, False]),
    )
    assert with_flags.in_saa_at(11.0 * u.s) is True
    assert with_flags.is_settled_at(11.0 * u.s) is True
    with pytest.raises(ValueError, match="outside BAT attitude coverage"):
        with_flags.in_saa_at(13.0 * u.s)
    with pytest.raises(ValueError, match="outside BAT attitude coverage"):
        with_flags.is_settled_at(9.0 * u.s)



def test_check_gti_fails_closed_when_attitude_saa_status_is_unknown_or_fails():
    class UnknownSaa:
        def in_saa_at(self, met_time):
            return None

    observation = BATObservation.__new__(BATObservation)
    observation.srctime = 11.0 * u.s
    observation._attitude = UnknownSaa()
    assert observation.check_gti() is False

    class BrokenSaa:
        def in_saa_at(self, met_time):
            raise RuntimeError("SAA flag unavailable")

    observation._attitude = BrokenSaa()
    assert observation.check_gti() is False


def test_bat_observation_rejects_sao_frame_endpoint_substitution(monkeypatch):
    class FakeFrame:
        obstime = np.array([10.0, 12.0]) * u.s

        def at(self, met_time):
            raise AssertionError("out-of-coverage query must not select a frame")

    class FakeSao:
        def get_spacecraft_frame(self):
            return FakeFrame()

        def get_spacecraft_states(self):
            return object()

        def get_bat_pointing(self):
            raise AssertionError("out-of-coverage query must not fetch pointing")

    class FakeBatSao:
        @staticmethod
        def open(path):
            return FakeSao()

    # The optional GDT extra is absent in a core-only wheel environment.
    monkeypatch.setattr(bat_observation_module, "BatSao", FakeBatSao, raising=False)
    observation = BATObservation.__new__(BATObservation)
    observation.srctime = 13.0 * u.s

    with pytest.raises(ValueError, match="outside SAO attitude coverage"):
        observation._load_from_sao("synthetic.sao")
