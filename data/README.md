`data/raw` is intended for original datasets such as DAiSEE exports.

`data/processed` is intended for cleaned frame crops, derived features, or train/validation/test splits.

For the comparison notebook currently in this repo:

- `data/processed/0` maps to `not_engaged`
- `data/processed/1` maps to `engaged`

`data/models/face_detection` stores the YOLO face detector used by the webcam demo.

`data/models/classification` is the handoff point for your future facial-expression model.
