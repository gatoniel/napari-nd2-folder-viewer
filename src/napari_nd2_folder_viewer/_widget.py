"""
This module provides a widget to open a folder of nd2 files and show them continuously.
"""
from typing import TYPE_CHECKING

import os
import pandas as pd
import numpy as np
import dask.array as da
from dask.cache import Cache

import nd2

from magicgui.widgets import FileEdit
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .exp_info import (
    get_exp_info,
    print_time_diff,
    antibiotic_exposure,
    calc_times,
    TimeDiff,
    to_datetime,
)

if TYPE_CHECKING:
    import napari

cache = Cache(2e10)  # Leverage twenty gigabytes of memory
cache.register()  # Turn cache on globally


def get_zstack_size(coord_info):
    for ci in coord_info:
        if ci[1] == "ZStackLoop":
            return ci[2]
    return 0


def get_tstack_size(coord_info):
    for ci in coord_info:
        if ci[1] == "NETimeLoop" or ci[1] == "TimeLoop":
            return ci[2]
    return 0


def get_xy_size(coord_info):
    for ci in coord_info:
        if ci[1] == "XYPosLoop":
            return ci[2]
    return 0


def get_nd2_files_in_folder(folder):
    zstack_sizes = []
    channel_names_list = []
    nd2_files = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(".nd2"):
            print(f)
            nd2_file = nd2.ND2File(os.path.join(folder, f))
            zstack_sizes.append(get_zstack_size(nd2_file._rdr._coord_info()))
            channel_names_list.append(nd2_file._rdr.channel_names())

            nd2_files.append(nd2_file)

    zlen = max(zstack_sizes)

    channel_names_len = [len(cn) for cn in channel_names_list]
    ind = np.argmax(channel_names_len)
    channel_names = channel_names_list[ind]

    xylen = nd2_files[ind].shape[-1]
    mlen = get_xy_size(nd2_files[ind]._rdr._coord_info())

    return nd2_files, xylen, mlen, zlen, channel_names


def insert_nd2_file_channels(img_, channel_names, channel_names_):
    imgs = []
    for i, cn in enumerate(channel_names):
        try:
            j = channel_names_.index(cn)
            imgs.append(img_[..., j, :, :])
        except ValueError:
            imgs.append(da.zeros_like(img_[..., 0, :, :]))
    img = da.stack(imgs, axis=-3)
    return img


def nd2_file_to_dask(nd2_file, zlen, channel_names, mlen, xylen):
    coord_info = nd2_file._rdr._coord_info()

    tlen_ = get_tstack_size(coord_info)
    if tlen_ == 0:
        tlen = 1
    else:
        tlen = tlen_

    zlen_ = get_zstack_size(coord_info)

    tmp_times = np.zeros((tlen, mlen, zlen))

    channel_names_ = nd2_file._rdr.channel_names()
    img = insert_nd2_file_channels(
        nd2_file.to_dask(), channel_names, channel_names_
    )

    if (
        tlen_ == 0
        or (tlen_ == 1 and zlen_ == 0 and img.ndim == 4)
        or (tlen_ == 1 and zlen_ != 0 and img.ndim == 5)
    ):
        img = da.expand_dims(img, axis=0)

    if zlen_ == 0:
        img = da.expand_dims(img, axis=2)

        first_img = da.zeros_like(img)
        shape = list(img.shape)
        shape[2] = zlen - 2
        last_imgs = da.zeros(
            tuple(shape),
            chunks=(1, 1, 1, 1, shape[4], shape[5]),
            dtype=np.uint16,
        )

        img = da.concatenate([first_img, img, last_imgs], axis=2)

    if tlen_ == 0 and zlen_ == 0:
        for i in range(nd2_file.metadata.contents.frameCount):
            k = nd2_file._rdr._coords_from_seq_index(i)
            tmp_times[:, k, :] = (
                nd2_file._rdr.frame_metadata(i)
                .channels[0]
                .time.absoluteJulianDayNumber
            )

    elif tlen_ == 0:
        print(nd2_file)
        print(img.shape)
        for i in range(nd2_file.metadata.contents.frameCount):
            k, l = nd2_file._rdr._coords_from_seq_index(i)
            tmp_times[:, k, l] = (
                nd2_file._rdr.frame_metadata(i)
                .channels[0]
                .time.absoluteJulianDayNumber
            )

    elif zlen_ == 0:
        for i in range(nd2_file.metadata.contents.frameCount):
            j, k = nd2_file._rdr._coords_from_seq_index(i)
            tmp_times[j, k, :] = (
                nd2_file._rdr.frame_metadata(i)
                .channels[0]
                .time.absoluteJulianDayNumber
            )

    else:
        for i in range(nd2_file.metadata.contents.frameCount):
            j, k, l = nd2_file._rdr._coords_from_seq_index(i)
            tmp_times[j, k, l] = (
                nd2_file._rdr.frame_metadata(i)
                .channels[0]
                .time.absoluteJulianDayNumber
            )

    return img, tmp_times


def color_from_name(name):
    if "GFP" in name or "epi" in name:
        return "green"
    elif "mRuby" in name:
        return "blue"
    elif "brightfield" in name or "Brightfield" in name:
        return "gray"
    else:
        return "blue"


class LoadWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.file_edit = FileEdit(label="Folder: ", mode="d")
        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.file_edit.native)
        self.layout().addWidget(btn)

        # varibales to be defined later
        self.times = None
        self.stack = None
        self.exp_info = None
        self.chip_channel_names = None

    def _on_click(self):
        root = self.file_edit.value
        nd2_files, xylen, mlen, zlen, channel_names = get_nd2_files_in_folder(
            root
        )

        imgs, times = [], []
        for nd2_file in nd2_files:
            img, tmp_times = nd2_file_to_dask(
                nd2_file, zlen, channel_names, mlen, xylen
            )
            imgs.append(img)
            times.append(tmp_times)

        self.times = np.concatenate(times, axis=0)
        self.stack = da.concatenate(imgs)

        colors = [color_from_name(cn) for cn in channel_names]
        opacities = [1,] + [
            0.6,
        ] * (len(colors) - 1)

        self.exp_info = get_exp_info(os.path.join(root, "exp-info.yaml"))
        self.chip_channel_names = pd.read_excel(
            os.path.join(root, "positions.xlsx"), header=None
        )[0]

        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 20
        self.viewer.text_overlay.color = "red"

        self.viewer.dims.events.current_step.connect(self.write_info)

        for i in range(len(channel_names)):
            self.viewer.add_image(
                self.stack[..., i, :, :],
                colormap=colors[i],
                opacity=opacities[i],
                name=channel_names[i],
            )

    def write_info(self, event):
        position = self.viewer.dims.current_step[1]

        current_step = tuple(self.viewer.dims.current_step[:3])

        pos_name = self.chip_channel_names[position]
        channel_name = pos_name.split("-")[0]
        ch = self.exp_info.channel_infos[channel_name]

        texts = [
            pos_name
            + " had "
            + print_time_diff(antibiotic_exposure(ch))
            + " hours of abx duration"
        ]

        if ch.antibiotic:
            abx = ch.antibiotic
            texts.append(
                f"{abx.name} ({abx.concentration:2.0f} {abx.concentration_unit})"
            )

        timefmt = "%Y-%m-%d %H-%M"

        durations = calc_times(
            current_step,
            self.chip_channel_names,
            self.exp_info,
            self.times,
        )

        if type(durations[0]) == TimeDiff:
            texts.append(
                f"Current abx time:           {print_time_diff(durations[0])}"
            )
        else:
            texts.append("abx not started yet")

        if type(durations[1]) == TimeDiff:
            texts.append(
                f"Current regrowth time: {print_time_diff(durations[1])}"
            )
        else:
            texts.append("regrowth not started yet")

        current_time = to_datetime(self.times[current_step])
        current_time = current_time.strftime(timefmt)
        texts.append(f"Current time: {current_time}")

        texts.append(
            f"Start time:      {ch.antibiotic_start.strftime(timefmt)}"
        )

        texts.append(f"End time:        {ch.antibiotic_end.strftime(timefmt)}")

        text = "\n".join(texts)
        self.viewer.text_overlay.text = text
