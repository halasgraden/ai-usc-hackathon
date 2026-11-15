# Face-guard model (YuNet)

`capture/face_guard.py` needs OpenCV's YuNet face detector weights,
`face_detection_yunet_2023mar.onnx`, from the opencv_zoo repository
(`models/face_detection_yunet/` in https://github.com/opencv/opencv_zoo).
The file is not shipped in the `opencv-contrib-python-headless` wheel and
`*.onnx` is git-ignored, so each rig machine downloads it once:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The rig reads it from `LEFTOVER_FACE_MODEL` (default
`models/face_detection_yunet_2023mar.onnx`, relative to the working directory).

Without a loadable file the guard fails closed: every captured frame is
treated as containing a face and dropped, so the rig uploads nothing and says
why on each plate line. The model has not yet been run in this repository
(the sandbox could not download it), including whether this file loads under
the installed OpenCV 5.0; `tests/test_capture.py::test_real_yunet_finds_no_face_in_a_blank_or_plate_frame`
runs automatically once the file is in place.
