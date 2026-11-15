# Measurement: from a plate photo to millilitres

Entry point: `depth.volume.measure_food_volumes(image, plate_mask, food_masks, estimator=..., focal_px=...)`.
It returns a `PlateMeasurement` (one volume in ml per food mask, plus the scale,
focal length, plate distance and depth scale it used), or `None` when the frame
must not be measured.

## Pipeline

1. **Scale from the plate** (`capture/scale.py`). Every Village plate is 8.5 in
   (21.59 cm) across. We take the convex hull of the segmented plate (food
   leaves holes and bites in the mask), fit an ellipse to it, and set
   `cm_per_px = 21.59 / major_axis_px`. With a slightly tilted camera the rim
   becomes an ellipse whose major axis still spans the full diameter, so the
   ellipse handles small tilt.
2. **Plate distance** (`depth/volume.py`). Pinhole camera: `Z = focal_px * cm_per_px`.
   The focal length comes from a one-off calibration of the rig phone (preferred:
   photograph the plate at a tape-measured distance, `f = major_px * Z / 21.59`,
   or use `detect_scale` with an ArUco marker), or else from the intrinsics that
   Depth Anything 3 predicts. An optional `rim_height_cm` moves the anchor from
   the rim down to the well floor.
3. **Depth** (`depth/estimator.py`). Depth Anything 3 Small (`depth-anything/DA3-SMALL`)
   gives depth along the optical axis up to one unknown scale factor. We fit a
   plane to bare plate pixels inside the inner 75% of the rim ellipse (skipping
   the sloped rim) and scale the map so that plane sits `Z` cm from the camera
   at the plate centre (`to_centimetres`).
4. **Volume** (`depth/plane.py` + `depth/volume.py`). Height above the plane per
   pixel, then each food mask's volume, where each pixel counts its true ground
   footprint under perspective: `(z/f)^2 + (z/f)(r_x dz/du + r_y dz/dv)`. Using
   the plate's scale for every pixel would over-count tall food by about
   `(Z_plate / z_food)^2`.

## Why relative depth anchored on the plate, not metric depth or a calibration block

The rig cannot rely on a 2 cm calibration block in every frame, so the old
block-based `to_centimetres` is gone. DA3 has a metric checkpoint
(`DA3METRIC-LARGE`), but its own conversion to metres
(`focal * output / 300`) needs the focal length too, so it removes no
dependency, and it is a Large model, slower on the rig. Any global scale error
of a monocular metric model multiplies straight into every food height;
anchoring on the plate replaces that error with one we can reason about (the
plate's measured size and the focal length). So the default is DA3-Small with
the plate anchor. A metric checkpoint still works through the same code, and
then `PlateMeasurement.depth_scale` should be close to 1: its distance from 1
is a free consistency check between the model and the plate.

Error propagation (algebra, not measurement): volume is proportional to
`focal_px * cm_per_px^3`. A focal length error passes through one-for-one; a
scale error is cubed, so the plate's ellipse fit matters most.

## Assumptions and failure modes

- **Plate partly out of frame**: the hull is cut by the border. Any plate pixel
  within 2 px of the edge returns `None`.
- **Implausible outline**: minor/major axis below 0.8 (tilt past about 37
  degrees, or not a round dish), or hull area / ellipse area outside
  0.95 to 1.05 (a tray or square dish) returns `None`.
- **Non-standard dish**: a round bowl or side plate passes the checks and gives
  a wrong scale. It has to be caught by the segmentation class (`bowl` vs `plate`).
- **Food hiding a whole arc of the rim**: the hull loses that arc and the major
  axis can come out short. Not detected beyond the fill-ratio check.
- **No bare plate visible**: nothing to fit the plane to, returns `None`.
- **DA3 output must be proportional to distance** (scale only, no offset). If
  the model adds an offset, the plate anchor does not remove it. Untested.
- **Flat plate**: `rim_height_cm` defaults to 0; the Village plate's rim height
  has not been measured. Plane fitting in depth (not inverse depth) is exact
  only for a plate facing the camera squarely.
- **Principal point** is assumed at the image centre.
- The slope term uses depth gradients, so it will pick up DA3 noise; it is
  linear in the gradient, so noise partly averages out over a food region. Untested.

## What is tested and what is not

Tested (`tests/test_depth.py`, `tests/test_capture.py`), on synthetic data only:
ray-cast pinhole scenes of a 21.59 cm plate 40 cm below a 600 px focal camera
with a block and a spherical-cap mound. With noise-free depth known only up to
an arbitrary scale, the recovered volumes match the analytic volumes within the
tolerances stated in the tests (2%, from pixel quantisation). Plate scale from
discs, bitten discs, tilted ellipses; rejection of clipped, tiny, elongated and
square masks; the ArUco path on a generated marker; the estimator wrapper with
an injected fake model (resize to frame, focal rescaling, BGR to RGB, bad output).

Not tested at all: the real DA3 loader (`load_da3`). It is written against the
DA3 README and `depth_anything_3/api.py` but has never loaded weights or run
(no network to Hugging Face or GitHub where it was written). There is no
accuracy number for real plates. The first real check should be the
measuring-cup test in `eval/`: known volumes of rice or mashed potato on a
Village plate under the rig camera.

## Installing DA3

From the official repository (`ByteDance-Seed/Depth-Anything-3`): install
`torch`/`torchvision`, then `pip install -e .` in a clone. The `depth-anything-3`
package on PyPI is not published by the DA3 authors; do not use it.

## First run on real photos (15 Nov 2026)

`notebooks/colab_pipeline_real.ipynb`, Colab T4: `village-v2.onnx`, DA3-Small and
SigLIP 2 on six Village photos from 14 Nov, matched against that evening's Dinner menu.
This checks that the stages run and give physically sensible output. **No grams are
validated**: no plate has been weighed.

What it showed:

* All three models loaded and ran end to end.
* The empty plate (`empty_p1_clean_01`) had no food regions.
* Camera height recovered from the plate's known 21.59 cm diameter: 21-46 cm across the
  photos, consistent with a hand-held phone.
* Every real food was matched to a dish on the menu: Assorted French Fries, Roasted Lemon
  Zest Asparagus, Rice Pilaf, Garlic and Herb Roasted Chicken.
* The focal length DA3 predicts (1339-1551 px) and the iPhone main camera's nominal value
  (1156 px at 1600 px wide) disagree, and volume moves one-for-one with focal length:
  the two estimates of the same food differed by roughly 15-35%. A one-off focal
  calibration of the rig camera removes this uncertainty.
* Height maps: the bare plate was flat apart from a band of false height right beside the
  food (DA3 blurs depth across food edges) and at the raised rim.

Changes made in response:

* `api/pipeline.py`: food regions under 2 ml are dropped before matching. Small false
  detections measured 0.0 ml and were still being given a dish name.
* `depth/volume.py`: bare plate within 3% of the plate diameter of any food is left out of
  the plane fit. On a synthetic plate with a 1 cm halo round a 60 ml block this removes a
  3% under-count (`tests/test_depth.py`). On the real rice photo it changed the volume by
  under 1% (118.7 to 119.5 ml).
* A trimmed refit of the plane (`depth.plane.fit_plane_robust`) was also tried. On the same
  rice photo it halved the volume (119.5 to 51.6 ml). A step that sensitive cannot be
  trusted without weighed plates, so it is **off by default** (`robust=False`).

The same photo measured 137 ml on Colab (GPU, two detected regions) and 119 ml on a Mac
(CPU, one region). That gap is not explained yet.

Known gap: a pile of fries was segmented on its top layer only, so its volume is likely
low. Weighing a handful of plates is the next step for any accuracy claim.
