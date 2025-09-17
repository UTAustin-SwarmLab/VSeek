from pydantic import BaseModel

from vseek.data.frame import VideoFrames


class DataInput(BaseModel):
    video: VideoFrames
    question: str
    answer: str
    options: str
