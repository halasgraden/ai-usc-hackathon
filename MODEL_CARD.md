# Model card: Leftover Lens

Leftover Lens chains several models. One of them, the plate and food segmentation
model, was trained by us. The others are off-the-shelf. This card says what each
one does, what has actually been measured, and what has not.

Every model-quality number below is copied from `eval/results/village-v1.json`
(rendered as `eval/results/village-v1.md` by `eval/segmentation.py`). No other
quality number exists yet. When this card says "not measured", nothing was measured.

## Models in the pipeline

| Role | Model | Where it runs | Trained by us? | Run on real Village photos? |
| --- | --- | --- | --- | --- |
| Segmentation (student) | YOLO26n-seg, run `village-v1` | Live, ONNX Runtime (`vision/infer.py`) | Fine-tuned | Yes. Evaluated on 19 held-out photos (below) |
| Auto-label teacher: boxes | Grounding DINO base (`IDEA-Research/grounding-dino-base`) | Offline, Colab GPU (`labeling/autolabel.py`) | No | Yes. Made the training labels |
| Auto-label teacher: masks | SAM 2.1 Hiera small (`facebook/sam2.1-hiera-small`) | Offline, Colab GPU | No | Yes. Made the training labels |
| Depth | Depth Anything 3 Small (`depth-anything/DA3-SMALL`); `DA3METRIC-LARGE` optional | Live (`depth/estimator.py`) | No | **No. It has never loaded weights or run** |
| Menu matcher | SigLIP 2 ViT-B/16 224 px (`google/siglip2-base-patch16-224`) | Live (`menu/matcher.py`) | No | **No. Tested only with a fake embedder** |
| Grams | Volume x bulk density (`portion/grams.py`, `portion/densities.yaml`) | Live | Lookup table, not a learned model | Never checked against a scale |

### Segmentation student: `village-v1`

- **Architecture**: Ultralytics YOLO26n-seg (nano instance segmentation).
- **Classes**: 23, `core.types.SEG_CLASSES`. These are `plate`, `bowl` and the 21 coarse
  food categories of `core.types.FoodCategory`. The model predicts a category such as
  `poultry`. It does not predict a dish name; the menu matcher supplies that.
- **Starting point**: a checkpoint (`voxel51-v0`) first trained on the public Voxel51
  food-waste dataset, converted to the same 23-class list by `labeling/voxel51.py`
  (`notebooks/colab_train.ipynb`). That checkpoint started from Ultralytics'
  pretrained `yolo26n-seg.pt`. No held-out number for `voxel51-v0` is committed: not measured.
- **Fine-tuning** (`notebooks/colab_train_village.ipynb`, `vision/configs/yolo26n_seg.yaml`):
  640 px input, `epochs=100`, `patience=25`, `batch=8`, seed 0. Rotation 180 degrees, up/down
  and left/right flips, mosaic, mixup 0.1, Ultralytics copy-paste 0.5, brightness (`hsv_v`) 0.4.
  The notebook's `patience` and `batch` override the config file's values (20 and 16).
  Training stopped early at epoch 61. The best epoch was 36. It took about 16 minutes on a Colab T4.
- **Training data**: 66 real Village photos plus 200 copy-paste composites. See [DATA_CARD.md](DATA_CARD.md).
- **Deployment**: exported to ONNX at 640 px (`vision/export_onnx.py`). On the rig it runs
  with ONNX Runtime only, without PyTorch. Defaults are confidence 0.25, NMS IoU 0.7 and mask
  threshold 0.5 (`vision/infer.py`). `api/pipeline.py` merges all masks of one class on a plate
  into one region before measuring.
- **In progress**: `village-v2` is training now at 1024 px. It has no results yet.

### Offline teachers: Grounding DINO + SAM 2.1

These run only in `notebooks/colab_autolabel.ipynb` to draft training masks. They never
run in the live path. Each photo's file name gives its dishes. The labeler runs one
single-phrase Grounding DINO pass per class (`plate`, `bowl`, and each named food's
phrase from `labeling/pilot_items.yaml`). The box threshold is 0.30 for plate/bowl and 0.25
for food, and the text threshold is 0.25. SAM 2.1 turns each box into a mask.
`filter_labels` then:

- keeps one plate, chosen by score x area;
- drops food masks that are mostly off the plate's convex hull;
- drops a "bowl" that sits mostly on the plate, since that is usually a food pile;
- drops food masks the size of the whole plate;
- drops duplicate masks of the same class.

The teachers' own accuracy on Village photos was not measured. Their output was reviewed
by eye, and photos with bad labels were removed (DATA_CARD.md).

### Depth: Depth Anything 3

`depth/estimator.py` and `depth/volume.py` turn a relative depth map into millilitres.
They anchor the depth scale on the 21.59 cm plate (method in `docs/MEASUREMENT.md`). The
geometry is tested on synthetic ray-cast scenes only. The real DA3 loader (`load_da3`)
has never run. Its volume accuracy on real plates is not measured.

### Menu matcher: SigLIP 2

The matcher restricts candidates to today's dishes that have the predicted category. If
there is one candidate, no model is used. If there are several, SigLIP 2 compares the
masked crop with the dish names, and the matcher returns "unknown" below a probability of 0.5
or a margin of 0.15. Those thresholds are untuned starting values. `eval/matching.py` exists,
but no matching results have been committed: not measured.

## Intended use

- Estimating anonymous, per-dish plate waste at the USC Village Dining Hall dish return.
  The input is a top-down photo of one white 8.5 in Village plate. The outputs feed a staff
  dashboard and a weekly aggregate report.
- Numbers are meant to be read in aggregate over many plates and days. A single plate's
  number should not be read on its own.

## Out of scope

- Identifying, tracking or judging individual diners. The system stores plates, not people.
- Nutrition, calorie or allergen estimates.
- Other dining halls, other plate sizes or shapes, trays, bowls as the main dish, or
  angled or side-on views. The scale step assumes the Village plate, and none of these were
  in the data.
- Any decision about a person: staff discipline, student charges, or grading.
- Reporting grams, cost or CO2e as measured facts. Grams are unvalidated, and the cost and
  CO2e factors in `analytics/assumptions.py` are placeholders.

## Evaluation

**Data**: the validation split of the Village dataset. It has 19 real photos in three whole
held-out plate groups (`asparagus+chicken_p1`, `empty_p1`, `fries_p4`). No composites are
included. Metrics are Ultralytics mask precision, recall, AP50 and AP50-95 at 640 px. They
were computed against the auto-labels, which were reviewed by eye but not drawn by hand.

**Results** (quoted from `eval/results/village-v1.json` / `eval/results/village-v1.md`):

Overall mask mAP50 **0.755**, mask mAP50-95 **0.545**.

| Class | Val images | Val instances | Precision | Recall | Mask AP50 | Mask AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| plate | 19 | 19 | 0.965 | 1.000 | 0.995 | 0.950 |
| bowl * | 2 | 2 | 1.000 | 0.000 | 0.745 | 0.124 |
| poultry * | 3 | 3 | 1.000 | 0.906 | 0.995 | 0.995 |
| cooked_vegetables | 3 | 12 | 0.537 | 0.500 | 0.379 | 0.185 |
| potatoes_fried_roasted | 12 | 26 | 0.498 | 0.692 | 0.664 | 0.470 |

\* fewer than 5 validation instances: too few to trust (`MIN_INSTANCES = 5` in
`eval/segmentation.py`).

How to read this:

- Plate segmentation is the only strong result with a meaningful sample (19 instances). The
  rest of the pipeline relies on it, because the plate sets the scale.
- The food classes that have enough instances to read are weak: cooked vegetables reach mask
  AP50 0.379 and fries reach 0.664. Each comes from a single held-out plate (asparagus and
  fries, respectively).
- The overall mAP averages these five classes, so the untrustworthy `bowl` and `poultry` rows
  pull it in both directions. Do not quote the overall number without the table.
- The validation split is also the set Ultralytics used to pick the best epoch and to stop
  early. There is no separate untouched test set, so these numbers are likely optimistic.

**Not evaluated**:

| What | Status |
| --- | --- |
| 18 of the 23 classes, including `rice_grains`, `pizza_burger_sandwich` (burger, hot dog), `red_meat` (short ribs), `baked_dessert` (crepe) | No held-out photos. These dishes had only one plate group, which is never held out |
| Carrots as a dish | Carrot photos are not in the validation groups; `cooked_vegetables` is scored on asparagus only |
| Teacher (Grounding DINO + SAM 2.1) label accuracy | Not measured; visual review only |
| Volume (ml) from DA3 on real plates | Not measured; DA3 never run |
| Grams | Not measured; no plate has been weighed |
| Menu matching accuracy | Not measured; SigLIP never run |
| End-to-end (photo to grams per dish) | Not measured |
| Other days, meals, lighting, the rig camera | Not measured |
| `voxel51-v0` starting checkpoint | No held-out number committed |
| `village-v2` (1024 px) | Training; no results |

## Known limitations and failure modes

- **Piles are segmented on top only.** On a held-out fries pile, the predicted mask covered
  roughly the top layer only. We saw this in one validation image and did not measure it. The
  2D mask feeds the volume step, so under-covered piles probably under-count volume.
- **Weak food masks**: see the cooked vegetables and fries rows above.
- **Labels come from teachers.** Training and validation masks were drawn by Grounding DINO +
  SAM 2.1, so the student learns the teachers' mistakes and is scored against them. Photos
  where a named food got no mask were dropped (mostly chicken next to asparagus). The kept data
  is therefore biased toward foods the teachers find easily.
- **Tiny, narrow dataset**: 85 photos from one evening in one hall on one plate type
  (DATA_CARD.md). Lighting, cameras, tables, menus and plates outside that are untested.
- **Merging by class**: two different dishes of the same class on one plate are measured and
  matched as one region (`api/pipeline.py`).
- **Scale depends on the plate.** A bowl or side plate passes the shape checks and gives a
  wrong scale unless segmentation calls it `bowl` (`docs/MEASUREMENT.md`). `bowl` has 2
  validation instances, too few to know how well that works.
- **Grams carry assumed uncertainty.** Densities are partly estimates. The ranges are
  +/-15% (published) and +/-30% (estimate), and those percentages are placeholders
  (`portion/densities.yaml`).

## Ethical and privacy considerations

- No student data is collected or stored. There is no app, card ID, or face or person field
  (`docs/CONTRACTS.md`, rule 1). The camera points down at plates on the dish return.
- Training photos show plates only.
- Frames containing a face are to be dropped unstored by `capture/face_guard.py`. That guard
  is being implemented now; until it ships, the face backstop does not exist.
- Waste numbers about dishes could be misread as judgements of the cooks or of diners. They
  are estimates with unvalidated grams. Present them as trends, not exact amounts.
- Cost and CO2e numbers are placeholder factors, not USC figures, and must be labelled that way.

## Licenses

| Component | License | Note |
| --- | --- | --- |
| Ultralytics YOLO26 code and pretrained `yolo26n-seg.pt` | AGPL-3.0 | See below |
| `village-v1` weights (derived from the above) | Treat as AGPL-3.0 | Ultralytics' stated position is that models trained from its weights fall under AGPL-3.0 unless an Enterprise license is bought |
| Grounding DINO base | Apache-2.0 (Hugging Face model page) | Offline only |
| SAM 2.1 Hiera small | Apache-2.0 (Hugging Face model page) | Offline only |
| Depth Anything 3 Small; DA3METRIC-LARGE | Apache-2.0 (Hugging Face model pages) | Other DA3 checkpoints may differ; check before switching |
| SigLIP 2 base patch16-224 | Apache-2.0 (Hugging Face model page) | |
| Voxel51 food-waste dataset | MIT (Hugging Face dataset page) | Used for the starting checkpoint |
| This repository | No license file yet | |

**AGPL-3.0 implication**: the live rig and API run `village-v1` through ONNX Runtime without the
`ultralytics` package. The weights still derive from Ultralytics' AGPL-3.0 model. If Leftover
Lens is deployed as a network service, for example the USC dashboard and API, AGPL-3.0
requires offering the complete corresponding source to its users. The options are to release
this repository under AGPL-3.0-compatible terms, buy an Ultralytics Enterprise license, or
switch to a segmentation model under a permissive license. This needs a decision before any
deployment beyond the hackathon. It is not legal advice.
