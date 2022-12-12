from typing import Dict, Optional

import yaml
import julian
import datetime
from dataclasses import dataclass
import desert
import marshmallow


@dataclass
class Molecule:
    name: str
    concentration: float
    concentration_unit: str


@dataclass
class SingleChannel:
    comment: str
    antibiotic_start: str = desert.field(
        marshmallow.fields.NaiveDateTime(format="%Y-%m-%d %H-%M")
    )
    antibiotic_end: str = desert.field(
        marshmallow.fields.NaiveDateTime(format="%Y-%m-%d %H-%M")
    )
    antibiotic: Optional[Molecule]


@dataclass
class Regrowth:
    inducer: Molecule
    fluorophore: Molecule


@dataclass
class GeneralAntibioticsInfo:
    duration_hours: int


@dataclass
class GeneralInfo:
    comment: str
    antibiotics: GeneralAntibioticsInfo
    regrowth: Regrowth
    invert_stage_x: bool
    invert_stage_y: bool


@dataclass
class ExpInfo:
    general_info: GeneralInfo
    channel_infos: Dict[str, SingleChannel]


@dataclass
class TimeDiff:
    hours: int
    minutes: int


def get_exp_info(path):
    with open(path) as f:
        exp_info = yaml.safe_load(f)

    schema = desert.schema(ExpInfo)
    return schema.load(exp_info)


def antibiotic_exposure(single_channel: SingleChannel):
    td = single_channel.antibiotic_end - single_channel.antibiotic_start
    return to_time_diff(td)


def to_time_diff(td):
    hours, seconds = divmod(td.seconds, 3600)
    return TimeDiff(td.days * 24 + hours, seconds // 60)


def print_time_diff(td: TimeDiff):
    return f"{td.hours:02d}:{td.minutes:02d}"


def to_datetime(time):
    dt = datetime.timedelta(hours=2)
    return julian.from_jd(time, fmt="jd") + dt


def calc_times_(coords, single_channel, times):
    date = to_datetime(times[tuple(coords)])

    zero = datetime.timedelta(0)

    td_abx = date - single_channel.antibiotic_start
    if td_abx > zero:
        td_abx = to_time_diff(td_abx)

    td_reg = date - single_channel.antibiotic_end
    if td_reg > zero:
        td_reg = to_time_diff(td_reg)

    return td_abx, td_reg


def calc_times(coords, chip_channel_names, exp_info, times):
    channel_name = chip_channel_names[coords[1]].split("-")[0]
    return calc_times_(coords, exp_info.channel_infos[channel_name], times)
