# Leftover Lens

A camera over the dish return at USC Village Dining Hall that measures what
comes back uneaten. Each plate becomes anonymous numbers (dish, grams, percent
left) for a staff dashboard and a weekly report USG can publish. It stores
plates, never people: no student app, no IDs, no faces.

USC forecasts what to cook and composts what is left, but nothing measures
plate waste per dish. This fills that gap.

## How it works

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Capture | `capture/` | One sharp frame per plate from a phone on a stand; the 8.5 in plate sets the scale; frames with a face are dropped |
| 2. Segment | `labeling/`, `vision/` | Grounding DINO + SAM 2.1 auto-label training masks offline; a YOLO26n-seg model we train runs live through ONNX Runtime |
| 3. Match | `menu/` | Imports today's Village menu; a region can only be one of today's dishes or "unknown" |
| 4. Measure | `depth/`, `portion/` | Depth Anything 3 depth, scaled by the plate's known size; plate-plane fit, volume in ml; grams = volume x bulk density |
| 5. Report | `analytics/`, `api/`, `dashboard/` | Per-dish waste, flags, staff dashboard, USG report |

## Status

| Module | State |
| --- | --- |
| `menu/` importer, cleaner, categorizer | Working, tested against a real Village menu |
| `portion/` grams and percent left | Working, tested |
| `depth/plane.py` plane fit, height map, volume | Working, tested on synthetic geometry |
| `capture/scale.py` plate-ellipse scale; `depth/estimator.py`, `depth/volume.py` DA3 depth to ml | Working on synthetic geometry (see `docs/MEASUREMENT.md`); real DA3 not yet run |
| `menu/matcher.py` SigLIP match to today's menu, `eval/matching.py` | Tested with a fake embedder; SigLIP not yet run, thresholds untuned |
| `labeling/` Voxel51 converter, class map, mask geometry | Working; Voxel51 conversion run on the real dataset |
| `labeling/autolabel.py` Grounding DINO + SAM 2.1 | Filtering, prompts, split and label export tested; model calls not yet run |
| `augment/composite.py` paste and lighting augmentation | Working, tested on synthetic plates |
| `augment/dataset.py` composites into the training split only | Working, tested |
| `eval/segmentation.py` held-out per-class mask AP, written to `eval/results/` | Working, tested; first real run pending |
| `vision/infer.py` ONNX decode, NMS, masks | Working, tested; checked on real photos with the Voxel51 baseline |
| `vision/train.py`, `vision/export_onnx.py` | `village-v1` trained on our Village photos (`notebooks/colab_train_village.ipynb`); 1024 px `village-v2` training |
| `capture/` source, plate trigger, YuNet face guard, rig loop `python -m capture` | Tested on synthetic frames and against the API with a fake pipeline; trigger thresholds untuned; face guard fails closed and the real YuNet model has not been run yet |
| `api/` | All endpoints in `docs/CONTRACTS.md` work on SQLite; `POST /plates/image` runs segment, merge per class, depth to ml, grams, menu match (tested with fake models; needs the exported ONNX model, DA3 and SigLIP installed to run for real) |
| `analytics/` per-dish aggregates, trends, high-waste flags; `python -m api.seed` demo data (labelled `seeded`) | Working, tested |
| `dashboard/`, grams eval (needs weighed plates) | Not started; API and dashboard contracts are in `docs/CONTRACTS.md` |

## Results

Every number here is copied from a file that an `eval/` script wrote under `eval/results/`;
nothing is typed by hand. Models, limits and licences: [MODEL_CARD.md](MODEL_CARD.md).
The photos, labelling and splits: [DATA_CARD.md](DATA_CARD.md).

#### Segmentation: village-v1 (`eval/results/village-v1.json`)

Held-out plates, mask mAP50 0.755, mask mAP50-95 0.545.

| Class | Val images | Val instances | Precision | Recall | Mask AP50 | Mask AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| plate | 19 | 19 | 0.965 | 1.000 | 0.995 | 0.950 |
| bowl * | 2 | 2 | 1.000 | 0.000 | 0.745 | 0.124 |
| poultry * | 3 | 3 | 1.000 | 0.906 | 0.995 | 0.995 |
| cooked_vegetables | 3 | 12 | 0.537 | 0.500 | 0.379 | 0.185 |
| potatoes_fried_roasted | 12 | 26 | 0.498 | 0.692 | 0.664 | 0.470 |

\* fewer than 5 validation instances: too few to trust.
Classes with no validation instances are not listed; they have no held-out score.

How to read it:
* The validation set is 19 real photos from three whole plates held out by plate
  (`asparagus+chicken_p1`, `empty_p1`, `fries_p4`); no photo of those plates, and no
  composite made from them, was trained on.
* There is no separate test set: the same 19 photos also chose the best epoch, so these
  numbers are likely somewhat optimistic.
* Rice, burger, short ribs and crepe were photographed at only one sitting each, so they have
  no held-out score at all.
* Not measured yet: grams (no weighed plates), menu matching, and the end-to-end pipeline.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Python 3.11
pip install -e ".[dev]"
cp .env.example .env                                  # then set LEFTOVER_DEVICE_TOKEN

python -m menu                                        # today's plate candidates
python -m menu --file tests/fixtures/village_2026-11-14.json
uvicorn api.main:app --reload                         # http://localhost:8000/docs

ruff check . && ruff format --check . && mypy . && pytest
```

Or `docker compose up --build` for the API on port 8000.

The rig (phone over the dish return, posting each plate to the API):

```bash
pip install -e ".[dev,vision]"
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
set -a; source .env; set +a                           # nothing loads .env automatically
python -m api.seed                                    # optional: 4 weeks of labelled demo data
python -m capture --meal Dinner                       # start with the slot empty
```

Extras: `.[vision]` for the live rig (OpenCV, ONNX Runtime), `.[train]` for
Ultralytics training, `.[match]` for SigLIP. Grounding DINO and SAM 2.1 run in
`notebooks/colab_autolabel.ipynb` through Hugging Face `transformers`; Depth Anything 3 is
installed from its own repository on the GPU machine.

## The menu stage

`python -m menu` prints what the camera is allowed to recognize for each meal.
The pipeline is `menu/importer.py` (API, then cache, then a hand-entered
file), `menu/cleaner.py` (roles, merging, implied dishes) and
`menu/categorizer.py` (dish to coarse category). The rules are data, in
`menu/data/*.yaml`; a dish the rules get wrong is fixed with one line in
`menu/data/overrides.yaml`.

## Repository layout

```
core/        shared types: FoodCategory, FoodRegion, PortionEstimate
capture/     camera source, plate trigger, plate scale, face guard, rig loop
models/      model files (git-ignored); see models/FACE_MODEL.md
labeling/    Grounding DINO + SAM 2.1 auto-labeling, Voxel51 converter, pilot dish list
augment/     copy-paste, geometric and photometric augmentation
vision/      YOLO26n-seg training config, ONNX export, inference
depth/       depth model wrapper, plate plane, height map, volume
portion/     volume to grams, densities.yaml
menu/        Village menu importer, cleaner, categorizer
analytics/   per-dish aggregates and flags
api/         FastAPI app
eval/        evaluation scripts and committed results
tests/       unit tests per module, real menu fixture
```
