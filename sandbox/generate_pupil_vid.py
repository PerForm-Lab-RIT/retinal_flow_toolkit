import json
import logging
import pathlib
import time
import uuid

import av
import click
import numpy as np


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-f", "--force", is_flag=True, help="Force overwriting of existing directory"
)
def main(input_file, force):
    input_file = pathlib.Path(input_file)
    if not force and _rec_dir_from_input_file(input_file).exists():
        raise FileExistsError(
            f"Target recording directory {_rec_dir_from_input_file(input_file)} "
            "already exists"
        )
    create_recording_directory(input_file)


def create_recording_directory(input_file):
    rec_dir = _rec_dir_from_input_file(input_file)
    target_path = rec_dir / f"world{input_file.suffix.lower()}"
    if click.confirm(
        f"{input_file.name} will be moved to {target_path}. Continue?",
        default=True,
        abort=True,
    ):
        logging.info(f"Creating recording direcotry at {rec_dir}")
        rec_dir.mkdir(exist_ok=True)
        logging.info(f"Moving {input_file.name} to {target_path}")
        input_file.rename(target_path)
        logging.info(f"Generating timestamps for {target_path.name}")
        world_timestamps = extract_and_save_timestamps_with_offset(target_path)
        logging.info("Generating info.player.json")
        create_info_player_json(rec_dir, world_timestamps)
        logging.info("Recording created successfully")


def extract_and_save_timestamps_with_offset(path_video, offset=0.0):
    pts_timestamps = sorted(extract_pts_timestamps(path_video))
    pts_timestamps = np.array(pts_timestamps)
    pts = pts_timestamps[:, 0].astype(int)
    timestamps = pts_timestamps[:, 1]
    timestamps += offset

    lookup_entry = np.dtype(
        [
            ("container_idx", "<i8"),
            ("container_frame_idx", "<i8"),
            ("timestamp", "<f8"),
            ("pts", "<i8"),
        ]
    )
    lookup = np.empty(timestamps.size, dtype=lookup_entry).view(np.recarray)
    lookup.timestamp = timestamps
    lookup.container_idx = 0
    lookup.container_frame_idx = np.arange(timestamps.size, dtype=int)
    lookup.pts = pts

    np.save(path_ts_from_video_path(path_video), timestamps)
    np.save(path_lut_from_video_path(path_video), lookup)
    return timestamps


def create_info_player_json(rec_dir, world_timestamps):
    info_player_json = rec_dir / "info.player.json"
    duration = world_timestamps[-1] - world_timestamps[0]
    with info_player_json.open("w") as f:
        json.dump(
            {
                "duration_s": duration,
                "meta_version": "2.3",
                "min_player_version": "2.0",
                "recording_name": rec_dir.name,
                "recording_software_name": "Pupil Capture",
                "recording_software_version": "3.4.0",
                "recording_uuid": str(uuid.uuid4()),
                "start_time_synced_s": 0.0,
                "start_time_system_s": time.time(),
                "system_info": __file__,
            },
            f,
        )


def extract_pts_timestamps(path_scene_video):
    container = av.open(str(path_scene_video))
    for packet in container.demux(video=0):
        if packet.pts is None:
            continue
        yield packet.pts, float(packet.pts * packet.time_base)


def path_ts_from_video_path(path_video):
    return path_video.parent / (path_video.stem + "_timestamps.npy")


def path_lut_from_video_path(path_video):
    return path_video.parent / (path_video.stem + "_lookup.npy")


def _rec_dir_from_input_file(input_file):
    return input_file.with_suffix("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("libav").setLevel(logging.ERROR)
    main()