import enum
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

# Prefer Decord for fast video decoding; fallback to OpenCV if unavailable
try:
    from decord import VideoReader as DecordVideoReader
    from decord import cpu as decord_cpu
    _HAS_DECORD = True
except Exception:  # ImportError or other runtime issues
    _HAS_DECORD = False


class VideoFormat(enum.Enum):
    """Status Enum for the CV API."""

    MP4 = "mp4"
    LIST_OF_ARRAY = "list_of_array"


class VideoInfo(BaseModel):
    """Represents information about a video file."""

    format: VideoFormat
    frame_width: int
    frame_height: int
    original_frame_count: int
    video_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    video_path: str | None = None
    processed_fps: float | None = None
    processed_frame_count: int = 0
    original_fps: float | None = None
    original_duration: float | None = None
    frame_step: int | None = None


class Video:
    """vflow's Video Object."""

    def __init__(
        self,
        read_format: VideoFormat,
        video_path: str | Path | None = None,
        sequence_of_image: list[np.ndarray] | None = None,
    ) -> None:
        """Video Frame Processor.

        Args:
            video_path (str | Path): Path to video file.
            read_format (VideoFormat): Format to read the video in.
            sequence_of_image (list[np.ndarray] | None): List of image arrays
                for processing.
        """
        self._video_path = video_path
        self._read_format = read_format
        self.video_info = None
        self._using_decord = False
        if sequence_of_image:
            self.all_frames = sequence_of_image
            if isinstance(sequence_of_image[0], list):
                self.all_frames = sequence_of_image[0]
        self.import_video(str(video_path))
        self.current_frame_index = 0
        self.current_timestamp = (0.0, 0.0)
        self.video_ended = False

    def __str__(self) -> str:
        """Return a concise string representation of the Video object."""
        return str(self.video_info)

    def __repr__(self) -> str:
        """Return a detailed string representation of the Video object."""
        return repr(self.video_info)

    def import_video(self, video_path: str | None) -> None:
        """Read video from video_path.

        Args:
            video_path (str): Path to video file.
        """
        logging.info(f"Video format: {self._read_format}")
        if self._read_format == VideoFormat.MP4:
            if _HAS_DECORD:
                try:
                    self._vr = DecordVideoReader(video_path, ctx=decord_cpu(0))
                    self._using_decord = True
                    # Probe 1st frame to get width/height
                    first_frame = self._vr[0]
                    # Decord returns (H, W, C)
                    height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
                    original_fps = float(self._vr.get_avg_fps()) if hasattr(self._vr, "get_avg_fps") else None
                    original_frame_count = int(len(self._vr))
                    self.video_info = VideoInfo(
                        video_path=str(self._video_path),
                        format=self._read_format,
                        frame_width=width,
                        frame_height=height,
                        original_fps=original_fps,
                        original_frame_count=original_frame_count,
                        original_duration=(float(original_frame_count) / float(original_fps) if original_fps else None),
                    )
                except Exception as e:
                    logging.warning(f"Decord failed to open video, falling back to OpenCV. Reason: {e}")
                    self._using_decord = False
            if not self._using_decord:
                self._cap = cv2.VideoCapture(video_path)
                ret, probe_frame = self._cap.read()
                if not ret:
                    logging.error("Video path is invalid or cannot be read.")
                frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                original_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap.get(cv2.CAP_PROP_FPS) else None
                original_frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_info = VideoInfo(
                    video_path=str(self._video_path),
                    format=self._read_format,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    original_fps=original_fps,
                    original_frame_count=original_frame_count,
                    original_duration=(float(original_frame_count) / float(original_fps) if original_fps else None),
                )
        elif self._read_format == VideoFormat.LIST_OF_ARRAY:
            self.video_info = VideoInfo(
                format=self._read_format,
                frame_width=int(self.all_frames[0].shape[1]),
                frame_height=int(self.all_frames[0].shape[0]),
                original_frame_count=len(self.all_frames),
            )

    def _resize_frame_by_scale(
        self, frame_img: np.ndarray, frame_scale: int
    ) -> np.ndarray:
        """Resize frame image.

        Args:
            frame_img (np.ndarray): Frame image.
            frame_scale (int): Scale of frame.

        Returns:
            np.ndarray: Resized frame image.
        """
        return cv2.resize(
            frame_img,
            (
                int(self.video_info.frame_width / frame_scale),
                int(self.video_info.frame_height / frame_scale),
            ),
        )

    def get_all_frames_of_video(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> list:
        """Get video frames by frame_scale and second_per_frame.

        Args:
            return_format (str, optional): Return format. Defaults to "cv2".
                Options: [cv2, ndarray]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Interval between frames in seconds.
                If provided, frames will be extracted at this interval. Defaults to None.
        """  # noqa: E501
        if self._read_format == VideoFormat.LIST_OF_ARRAY:
            resize_func = lambda img: self.process_frame_image(  # noqa: E731
                frame_img=img,
                frame_scale=frame_scale,
                return_format=return_format,
            )
            all_frames = list(map(resize_func, self.all_frames))
            self.processed_frame_count = len(all_frames)
            return all_frames

        all_frames = []

        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval_in_sec is None
        ):
            msg = (
                "Either desired_fps",
                "or desired_interval_in_sec must be provided.",
            )
            raise ValueError(msg)

        if self._read_format == VideoFormat.MP4:
            original_fps = self.video_info.original_fps or 0.0
            if desired_fps is not None:
                if desired_fps <= 0:
                    raise ValueError("desired_fps must be > 0")
                frame_step = int(round(original_fps / desired_fps)) if original_fps > 0 else 1
                processed_fps = float(desired_fps)
            else:
                if desired_interval_in_sec is None or desired_interval_in_sec <= 0:
                    raise ValueError("desired_interval_in_sec must be > 0")
                frame_step = int(round(original_fps * desired_interval_in_sec)) if original_fps > 0 else 1
                processed_fps = round(1.0 / desired_interval_in_sec, 2)

            frame_step = max(1, int(frame_step))

            for real_frame_idx in range(
                0, int(self.video_info.original_frame_count), int(frame_step)
            ):
                if getattr(self, "_using_decord", False):
                    try:
                        frame_nd = self._vr[real_frame_idx]
                        # Decord returns RGB NDArray; convert to numpy if needed
                        frame_img = (
                            frame_nd.asnumpy() if hasattr(frame_nd, "asnumpy") else np.asarray(frame_nd)
                        )
                    except Exception:
                        break
                else:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, real_frame_idx)
                    ret, frame_img = self._cap.read()
                    if not ret:
                        break
                    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

                frame_img = self.process_frame_image(
                    frame_img=frame_img,
                    frame_scale=frame_scale,
                    return_format=return_format,
                )
                all_frames.append(frame_img)
            if hasattr(self, "_cap") and self._cap is not None:
                self._cap.release()
            self.video_info.processed_frame_count = len(all_frames)
        return all_frames

    def get_next_frame(
        self,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
        desired_fps: int | None = None,
        desired_interval_in_sec: int | None = None,
    ) -> np.ndarray | None:
        """Get the next video frame based on frame step.

        Args:
            return_format (str, optional): Return format. Defaults to "ndarray".
                - [cv2, ndarray, pil]
            frame_scale (int | None, optional): Frame scale. Defaults to None.
            desired_fps (int | None, optional): Desired FPS. Defaults to None.
            desired_interval_in_sec (int | None, optional): Desired interval.
                Defaults to None.

        Returns:
            np.ndarray | None: The next frame as an ndarray, or None if no more
                frames are available or the video ended.
        """
        if (
            self._read_format == VideoFormat.MP4
            and desired_fps is None
            and desired_interval_in_sec is None
        ):
            msg = (
                "Either desired_fps or",
                "desired_interval_in_sec must be provided.",
            )
            raise ValueError(msg)

        if self.video_ended:
            logging.info("No frame available.")
            return None  # No more frames to process

        if self._read_format == VideoFormat.MP4:
            frame_step = self.get_frame_step(
                desired_fps=desired_fps,
                desired_interval_in_sec=desired_interval_in_sec,
            )
            # Skip to the next frame based on frame_step
            if getattr(self, "_using_decord", False):
                if self.current_frame_index >= int(self.video_info.original_frame_count):
                    self.video_ended = True
                    return None
                frame_nd = self._vr[self.current_frame_index]
                frame_img = (
                    frame_nd.asnumpy() if hasattr(frame_nd, "asnumpy") else np.asarray(frame_nd)
                )
            else:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                ret, frame_img = self._cap.read()
                if not ret:
                    self.video_ended = True
                    return None  # No more frames or error occurred
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        if self._read_format == VideoFormat.LIST_OF_ARRAY:
            if self.current_frame_index < len(self.all_frames):
                frame_img = self.all_frames[self.current_frame_index]
            else:
                # No more frames available.
                self.video_ended = True
                return None

        # Calculate current timestamp (real video time stamp) BEFORE updating frame index
        self.current_timestamp = self.get_current_timestamp()
        self.video_info.processed_frame_count += 1

        # Update the current frame index for the next call
        if self._read_format == VideoFormat.MP4:
            self.current_frame_index += frame_step
        elif self._read_format == VideoFormat.LIST_OF_ARRAY:
            self.current_frame_index += 1

        return self.process_frame_image(
            frame_img=frame_img,
            frame_scale=frame_scale,
            return_format=return_format,
        )

    def process_frame_image(
        self,
        frame_img: np.ndarray,
        return_format: str = "ndarray",
        frame_scale: int | None = None,
    ) -> np.ndarray:
        """Process a single frame image.

        Args:
            frame_img (np.ndarray): Input frame image.
            return_format (str, optional): Desired return format.
                Defaults to "ndarray".
            frame_scale (int | None, optional): Scale factor for resizing.
                Defaults to None.

        Returns:
            np.ndarray: Processed frame image.
        """
        if frame_scale is not None:
            frame_img = self._resize_frame_by_scale(frame_img, frame_scale)
        if return_format == "pil":
            frame_img = Image.fromarray(frame_img).convert("RGB")
        return frame_img

    def get_frame_step(
        self,
        desired_interval_in_sec: int | None = None,
        desired_fps: int | None = None,
    ) -> int:
        """Calculate the frame step based on desired interval or FPS.

        Args:
            desired_interval_in_sec (int | None): Desired interval between frames in seconds.
            desired_fps (int | None): Desired frames per second.

        Returns:
            int: Calculated frame step.
        """  # noqa: E501
        # Validate parameters
        if (desired_fps is None) == (desired_interval_in_sec is None):
            raise ValueError(
                ("Either desired_fps or desired_interval_in_sec must be provided, but not both.")
            )

        original_fps = self.video_info.original_fps or 0.0
        if desired_fps is not None:
            if desired_fps <= 0:
                raise ValueError("desired_fps must be > 0")
            frame_step = int(round(original_fps / desired_fps)) if original_fps > 0 else 1
            processed_fps = float(desired_fps)
        else:
            if desired_interval_in_sec is None or desired_interval_in_sec <= 0:
                raise ValueError("desired_interval_in_sec must be > 0")
            frame_step = int(round(original_fps * desired_interval_in_sec)) if original_fps > 0 else 1
            processed_fps = round(1.0 / desired_interval_in_sec, 2)

        frame_step = max(1, int(frame_step))
        self.video_info.processed_fps = processed_fps
        self.video_info.frame_step = frame_step
        return frame_step

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format.

        Args:
            seconds (float): Time in seconds.

        Returns:
            str: Time in HH:MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_current_timestamp(self) -> tuple[str, str]:
        """Calculate and return the current timestamp range (start, end) in HH:MM:SS format.

        Returns:
            tuple[str, str]: Start and end timestamp in HH:MM:SS format for the current frame.
        """
        if self._read_format == VideoFormat.MP4 and self.video_info.original_fps:
            # For MP4 videos, calculate timestamp using current frame index and original FPS
            frame_duration = 1.0 / self.video_info.original_fps
            start_time = self.current_frame_index / self.video_info.original_fps
            end_time = start_time + frame_duration
            return (
                self._seconds_to_timestamp(start_time),
                self._seconds_to_timestamp(end_time),
            )
        else:
            # For list of arrays or videos without FPS info, use frame index as timestamp
            start_time = float(self.current_frame_index)
            end_time = float(self.current_frame_index + 1)
            return (
                self._seconds_to_timestamp(start_time),
                self._seconds_to_timestamp(end_time),
            )

    def get_start_end_timestamp(self) -> tuple[float, float]:
        """Get the start and end timestamp of the video.

        Returns:
            tuple[float, float]: The start and end timestamp of the video.
        """
        if self.video_info.original_fps and self.video_info.original_fps > 0:
            duration = float(self.video_info.original_frame_count) / float(self.video_info.original_fps)
        else:
            # Fallback: use frame indices as seconds if FPS unknown
            duration = float(self.video_info.original_frame_count)
        return 0.0, duration

    def insert_annotation_to_current_frame(self, annotations: list[str]) -> None:
        """Insert annotations to the current frame.

        Args:
            annotations (list[str]): List of annotations.
        """

    def get_video_info(self) -> VideoInfo:
        """Return the VideoInfo object containing video information."""
        return self.video_info
