import cv2
import subprocess

class Camera:
    def __init__(self, source, width=1280, height=720, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

    def open(self):
        subprocess.run([
            "v4l2-ctl",
            "-d",
            self.source,
            "--set-ctrl=exposure_dynamic_framerate=0"
        ])
        
        self.capture = cv2.VideoCapture(
            self.source,
            cv2.CAP_V4L2
        )

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        return self.capture.isOpened()
    
    def read(self):
        return self.capture.read()

    def release(self):
        self.capture.release()