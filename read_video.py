import numpy as np
import cv2
import requests
from yt_dlp import YoutubeDL
from contextlib import redirect_stdout
from pathlib import Path
import io
import imageio.v3 as iio


url = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_10MB.mp4"
vid = cv2.VideoCapture(url)
# ret, frame = vid.read()

while(True):
    # Capture frame-by-frame
    ret, frame = vid.read()
    #print cap.isOpened(), ret
    if frame is not None:
        pass
        # print(frame.shape)
    else:
        break

print(vid.isOpened(), frame is not None)

buffer = io.BytesIO(requests.get(url).content)
video = buffer.getvalue()
frames = iio.imread(video, index=None)
print(frames.shape)





youtube_id = "https://www.youtube.com/watch?v=BaW_jenozKc"

ctx = {
    "outtmpl": "-",
    'logtostderr': True
}

buffer = io.BytesIO()
with redirect_stdout(buffer), YoutubeDL(ctx) as foo:
    foo.download([youtube_id])
# Path(f"vi.mp4").write_bytes(buffer.getvalue())

video = buffer.getvalue()
print(type(video))
frames = iio.imread(video, index=None)
print(frames.shape)


import decord
file_obj = io.BytesIO(video)
container = decord.VideoReader(file_obj)
print(container[2].shape)

# print(np.frombuffer(video, dtype=np.uint8).shape)
# img_array = np.asarray(bytearray(video), dtype=np.uint8)
# im = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)



import av

file_obj = io.BytesIO(video)
container = av.open(file_obj)
container.seek(0)
frames = []
for i, frame in enumerate(container.decode(video=0)):
    if i > 10:
        break
    if i >= 0:
        frames.append(frame)
out = np.stack([x.to_ndarray(format="rgb24") for x in frames])
print(out.shape)
