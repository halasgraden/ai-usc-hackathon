# Leftover Lens

A camera over the dish return at USC Village Dining Hall that measures what
comes back uneaten. Each plate becomes anonymous numbers (dish, grams, percent
left) for a staff dashboard and a weekly report USG can publish. It stores
plates, never people: no student app, no IDs, no faces.

USC forecasts what to cook and composts what is left, but nothing measures
plate waste per dish. This fills that gap.

## Run the dining dashboard

The administrator demo is now a working **two-tab dashboard** backed by the API:
all 37 seeded menu dishes, uncertainty ranges, simulated serving comparisons,
inline dish inspection, trends, CSV export, and weekly PDF/email previews.

```bash
# Terminal 1, from the repo root (Python environment installed as below)
python -m api.seed
LEFTOVER_STAFF_TOKEN=local-demo-staff LEFTOVER_DEVICE_TOKEN=local-demo-device python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
cd dashboard
npm ci
cp .env.example .env.local
npm run dev
```

Open **http://127.0.0.1:3000**. This is a local demo; generated plate and serving data
is visibly labeled. No CV weights are needed to explore it. No real CBORD connection
or email sending is claimed.

[Dashboard setup, metric definitions, workflow research and judging walkthrough](dashboard/README.md)
· [Verification evidence](docs/DASHBOARD_VERIFICATION.md)

![Full-menu dining dashboard](dashboard/public/evidence/desktop.png)

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
| `vision/train.py`, `vision/export_onnx.py` | `village-v2` (1024 px) trained on our Village photos and shipped; `village-v1` (640 px) kept for comparison (`notebooks/colab_train_village.ipynb`) |
| `capture/` source, plate trigger, YuNet face guard, rig loop `python -m capture` | Tested on synthetic frames and against the API with a fake pipeline; trigger thresholds untuned; face guard fails closed and the real YuNet model has not been run yet |
| `api/` | All endpoints in `docs/CONTRACTS.md` work on SQLite; `POST /plates/image` runs segment, merge per class, depth to ml, grams, menu match (tested with fake models; needs the exported ONNX model, DA3 and SigLIP installed to run for real) |
| `analytics/` per-dish aggregates, trends, high-waste flags; `python -m api.seed` demo data (labelled `seeded`) | Working, tested |
| `dashboard/` | Working Next.js two-tab module using FastAPI; full menu, trends, serving comparisons, uncertainty, CSV and report export. See `dashboard/README.md`. |
| Grams eval (needs weighed plates) | Not measured yet. |

## Results

Every number here is copied from a file that an `eval/` script wrote under `eval/results/`;
nothing is typed by hand. Models, limits and licences: [MODEL_CARD.md](MODEL_CARD.md).
The photos, labelling and splits: [DATA_CARD.md](DATA_CARD.md).

#### Segmentation: village-v2, the shipped model (`eval/results/village-v2.json`)

Held-out plates at 1024 px, mask mAP50 0.756, mask mAP50-95 0.615.

| Class | Val images | Val instances | Precision | Recall | Mask AP50 | Mask AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| plate | 19 | 19 | 0.954 | 1.000 | 0.995 | 0.954 |
| bowl * | 2 | 2 | 1.000 | 0.000 | 0.495 | 0.198 |
| poultry * | 3 | 3 | 1.000 | 0.900 | 0.995 | 0.995 |
| cooked_vegetables | 3 | 12 | 0.658 | 0.583 | 0.623 | 0.371 |
| potatoes_fried_roasted | 12 | 26 | 0.559 | 0.654 | 0.669 | 0.556 |

\* fewer than 5 validation instances: too few to trust.
Classes with no validation instances are not listed; they have no held-out score.

#### v1 (640 px) vs v2 (1024 px), same held-out photos (`python -m eval.compare village-v1 village-v2`)

| | village-v1 | village-v2 (1024 px) |
|---|---:|---:|
| Mask mAP50, all classes | 0.755 | 0.756 |
| Mask mAP50-95, all classes | 0.545 | 0.615 |
| plate AP50 | 0.995 | 0.995 |
| plate AP50-95 | 0.950 | 0.954 |
| bowl * AP50 | 0.745 | 0.495 |
| bowl * AP50-95 | 0.124 | 0.198 |
| poultry * AP50 | 0.995 | 0.995 |
| poultry * AP50-95 | 0.995 | 0.995 |
| cooked_vegetables AP50 | 0.379 | 0.623 |
| cooked_vegetables AP50-95 | 0.185 | 0.371 |
| potatoes_fried_roasted AP50 | 0.664 | 0.669 |
| potatoes_fried_roasted AP50-95 | 0.470 | 0.556 |

\* fewer than 5 validation instances in at least one run: too few to trust.

How to read it:
* Training at 1024 px instead of 640 px is the one change between v1 and v2. It mostly helped
  thin foods: asparagus spears (`cooked_vegetables`) and tighter fries masks (AP50-95).
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
