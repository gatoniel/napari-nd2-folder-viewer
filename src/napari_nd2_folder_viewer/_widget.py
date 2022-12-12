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
import napari
from sklearn.metrics import pairwise_distances
from napari_animation import Animation

from magicgui.widgets import FileEdit, Slider
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
    QVBoxLayout,
)

from .exp_info import (
    get_exp_info,
    print_time_diff,
    antibiotic_exposure,
    calc_times,
    TimeDiff,
    to_datetime,
)

cache = Cache(2e10)  # Leverage twenty gigabytes of memory
cache.register()  # Turn cache on globally


def test_nd2_timestamps(times, nd2_file):
    if np.all(np.diff(times, axis=0) == 0.0):
        print("made afjustment")
        period_time = nd2_file._rdr.experiment()[0].parameters.periodMs / 1000
        in_julian = period_time / (24 * 3600)

        add_time = np.arange(times.shape[0]) * in_julian

        return times + add_time[:, np.newaxis, np.newaxis]
    return times


def get_position_names(nd2_file):
    for exp in nd2_file.experiment:
        if exp.type == "XYPosLoop":
            return [p.name for p in exp.parameters.points]
    return []


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


def get_stage_positions(nd2_file):
    for exp in nd2_file.experiment:
        if exp.type == "XYPosLoop":
            return np.array([p.stagePositionUm for p in exp.parameters.points])
    return []


def get_position_names_and_inds(nd2_file, invert_x, invert_y):
    pos = get_stage_positions(nd2_file)
    if invert_x:
        pos[:, 0] *= -1

    dists = pairwise_distances(pos[:, 0][:, np.newaxis]).flatten()

    # beware hardcoded assumption of the distances
    nearest_neighbors_inds = np.logical_and(dists > 2000, dists < 6000)
    nearest_neighbors = dists[nearest_neighbors_inds]

    mean_diff = nearest_neighbors.mean()

    inds = []
    total_sort = []

    channel_names = np.empty(pos.shape[0], dtype=object)

    x_pos = pos[:, 0]

    min_x = x_pos.min()
    std_diff = mean_diff / 3

    nums = np.arange(pos.shape[0])

    for i in range(10):
        mid_ = min_x + i * mean_diff
        min_ = mid_ - std_diff
        max_ = mid_ + std_diff

        tmp_inds = np.logical_and(min_ < x_pos, x_pos < max_)
        inds.append(tmp_inds)

        y_pos = pos[tmp_inds, 1]
        sort_inds = np.argsort(y_pos)
        if invert_y:
            sort_inds = sort_inds[::-1]

        tmp_channel_names = np.empty(len(y_pos), dtype=object)
        tmp_channel_names[sort_inds] = [
            f"ch{i+1}-{j}" for j in range(1, len(y_pos) + 1)
        ]

        channel_names[tmp_inds] = tmp_channel_names

        total_sort.append(nums[tmp_inds][sort_inds])

    return inds, pos, channel_names, np.concatenate(total_sort)


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
        return "red"
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

        pos_btn = QPushButton("Play position!")
        pos_btn.clicked.connect(self._play_position)

        self.anim_fps_slider = Slider(value=5, min=1, max=50, step=1)
        anim_btn = QPushButton("Animate position!")
        anim_btn.clicked.connect(self._animate_position)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.file_edit.native)
        self.layout().addWidget(btn)
        self.layout().addWidget(pos_btn)
        self.layout().addWidget(self.anim_fps_slider.native)
        self.layout().addWidget(anim_btn)

        # varibales to be defined later
        self.times = None
        self.stack = None
        self.exp_info = None
        self.chip_channel_names = None
        self.channel_names = None
        self.colors = None
        self.opacities = None

    def _animate_position(self):
        animation = Animation(self.viewer)
        self.viewer.update_console({"animation": animation})

        self.viewer.reset_view()

        current_step = list(self.viewer.dims.current_step)

        current_step[0] = 0
        self.viewer.dims.current_step = tuple(current_step)
        animation.capture_keyframe()

        timelen = self.stack.shape[0] - 1
        current_step[0] = timelen

        self.viewer.dims.current_step = tuple(current_step)
        animation.capture_keyframe(steps=timelen)

        pos = self.viewer.dims.current_step[1]
        pos_name = self.chip_channel_names[pos]
        z = self.viewer.dims.current_step[2]
        fps = self.anim_fps_slider.value
        animation.animate(
            os.path.join(
                self.file_edit.value, f"{pos_name}_pos{pos}_z{z}_fps{fps}.mov"
            ),
            fps=self.anim_fps_slider.value,
            quality=9,
        )

    def _play_position(self):
        new_viewer = napari.Viewer()
        position = self.viewer.dims.current_step[1]
        z = self.viewer.dims.current_step[2]

        for i in range(len(self.channel_names)):
            image_layer = new_viewer.add_image(
                self.stack[:, position, z, i, :, :].compute(),
                colormap=self.colors[i],
                opacity=self.opacities[i],
                name=self.channel_names[i],
            )
            image_layer._keep_auto_contrast = True

        def tmp_write_info(event):
            current_step = (new_viewer.dims.current_step[0], position, z)

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

            texts.append(
                f"End time:        {ch.antibiotic_end.strftime(timefmt)}"
            )

            text = "\n".join(texts)
            new_viewer.text_overlay.text = text

        new_viewer.text_overlay.visible = True
        new_viewer.text_overlay.font_size = 20
        new_viewer.text_overlay.color = "red"

        new_viewer.dims.events.current_step.connect(tmp_write_info)

    def _on_click(self):
        root = self.file_edit.value
        (
            nd2_files,
            xylen,
            mlen,
            zlen,
            self.channel_names,
        ) = get_nd2_files_in_folder(root)
        self.exp_info = get_exp_info(os.path.join(root, "exp-info.yaml"))

        imgs, times = [], []
        channel_names = []
        for nd2_file in nd2_files:
            img_, tmp_times_ = nd2_file_to_dask(
                nd2_file, zlen, self.channel_names, mlen, xylen
            )
            _, _, tmp_channel_names, sort_inds = get_position_names_and_inds(
                nd2_file,
                self.exp_info.general_info.invert_stage_x,
                self.exp_info.general_info.invert_stage_y,
            )
            img = img_[:, sort_inds, ...]
            # print(tmp_times_.shape)
            tmp_times = tmp_times_[:, sort_inds, ...]
            tested_tmp_times = test_nd2_timestamps(tmp_times, nd2_file)
            imgs.append(img)
            times.append(tested_tmp_times)
            channel_names.append(tmp_channel_names[sort_inds])

        self.times = np.concatenate(times, axis=0)
        # print(self.times.shape)
        self.stack = da.concatenate(imgs)

        self.colors = [color_from_name(cn) for cn in self.channel_names]
        self.opacities = [1,] + [
            0.6,
        ] * (len(self.colors) - 1)

        self.chip_channel_names = channel_names[0]

        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 20
        self.viewer.text_overlay.color = "red"

        self.viewer.dims.events.current_step.connect(self.write_info)

        for i in range(len(self.channel_names)):
            image_layer = self.viewer.add_image(
                self.stack[..., i, :, :],
                colormap=self.colors[i],
                opacity=self.opacities[i],
                name=self.channel_names[i],
            )
            image_layer._keep_auto_contrast = True

    def write_info(self, event):
        position = self.viewer.dims.current_step[1]

        current_step = tuple(self.viewer.dims.current_step[:3])
        # print(current_step)

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

        # print(self.times[:, current_step[1], 1])

        durations = calc_times(
            current_step,
            self.chip_channel_names,
            self.exp_info,
            self.times,
        )
        # print(durations)

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
