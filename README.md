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
| 1. Capture | `capture/` | One sharp frame per plate from a phone on a stand; ArUco marker sets the scale; frames with a face are dropped |
| 2. Segment | `labeling/`, `vision/` | Grounding DINO + SAM 2.1 auto-label training masks offline; a YOLO26n-seg model we train runs live through ONNX Runtime |
| 3. Match | `menu/` | Imports today's Village menu; a region can only be one of today's dishes or "unknown" |
| 4. Measure | `depth/`, `portion/` | Depth map, plate-plane fit, volume in ml; grams = volume x bulk density |
| 5. Report | `analytics/`, `api/`, `dashboard/` | Per-dish waste, flags, staff dashboard, USG report |

## Status

| Module | State |
| --- | --- |
| `menu/` importer, cleaner, categorizer | Working, tested against a real Village menu |
| `portion/` grams and percent left | Working, tested |
| `depth/plane.py` plane fit, height map, volume | Working, tested on synthetic geometry |
| `labeling/` Voxel51 converter, class map, mask geometry | Working; Voxel51 conversion run on the real dataset |
| `labeling/autolabel.py` Grounding DINO + SAM 2.1 | Filtering, prompts, split and label export tested; model calls not yet run |
| `augment/composite.py` paste and lighting augmentation | Working, tested on synthetic plates |
| `vision/infer.py` ONNX decode, NMS, masks | Working, tested; checked on real photos with the Voxel51 baseline |
| `vision/train.py`, `vision/export_onnx.py` | Written; need a GPU run (`notebooks/colab_train.ipynb`) |
| `capture/`, `depth/estimator.py`, `analytics/` | Typed interfaces; tests marked `xfail` until implemented |
| `api/` | `/health` and `/menu/{date}` work; `POST /plates` is authenticated but returns 501 |
| `dashboard/`, `eval/` | Not started; API and dashboard contracts are in `docs/CONTRACTS.md` |

## Results

None yet. Every number in this section will be written by `eval/` from a
frozen test set and committed under `eval/results/`; nothing here is typed by hand.

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
capture/     camera source, plate trigger, ArUco scale, face guard
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
