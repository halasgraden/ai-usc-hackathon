# Data card: Village plates v1

This is the dataset used to fine-tune and evaluate the segmentation model `village-v1`
(see [MODEL_CARD.md](MODEL_CARD.md)). The photos and the exported dataset
(`village-dataset.zip`) live in the team's Google Drive. They are not committed to this
repository.

## Collection

| | |
| --- | --- |
| Where | USC Village Dining Hall |
| When | 14 Nov 2026, one session |
| Who | The project team |
| Camera | iPhone, hand-held, top-down |
| Format | HEIC, converted to upright JPEG with the longest side 1600 px (`notebooks/colab_autolabel.ipynb`) |
| Plates | The hall's white 8.5 in (21.59 cm) plates only |
| Backgrounds | Dark tables and steel dish-return counters |
| Raw files | 121, including 53 tighter "-crop" re-framings (37 + 16) of other photos |

Each photo's file name encodes its content as `item_plate#_state_##.jpg`, for example
`fries_p2_scraps_03.jpg`. Mixed plates join items with `+`, as in
`asparagus+chicken_p1_half_04.jpg`. The first two tokens form the **plate group**
(`fries_p2`), which means one physical plate. A crop keeps the same group as its original.

## Composition

Dishes photographed, with their segmentation class (`labeling/pilot_items.yaml`):

| Dish | Slug | Class |
| --- | --- | --- |
| Fries | `fries` | `potatoes_fried_roasted` |
| Roasted chicken | `chicken` | `poultry` |
| Rice pilaf | `rice-pilaf` | `rice_grains` |
| Asparagus | `asparagus` | `cooked_vegetables` |
| Carrots | `carrots` | `cooked_vegetables` |
| Burger | `burger` | `pizza_burger_sandwich` |
| Hot dog | `hotdog` | `pizza_burger_sandwich` |
| Crepe | `crepe` | `baked_dessert` |
| Braised short ribs | `short-ribs` | `red_meat` |
| Empty plate | `empty` | (plate only) |
| Mixed plates | `a+b` | one class per item |

`pasta` and `grapes` are in the pilot list but were not photographed.

**Per-dish photo counts are unknown.** The file list is not in the repository, so they
cannot be computed here. The figures that are known:

| Stage | Photos |
| --- | ---: |
| Raw | 121 |
| Auto-excluded (a named food got no mask) | 29 |
| Excluded by hand after visual review | 7 |
| **Final** | **85** |
| Train (real) | 66 |
| Validation (real) | 19 |
| Train composites (synthetic) | 200 |

The validation split by class (from `eval/results/village-v1.json`) is plate 19 instances in
19 images, bowl 2 in 2, poultry 3 in 3, cooked_vegetables 12 in 3, and
potatoes_fried_roasted 26 in 12. The excluded photos were all removed before the split.
Every kept photo has a mask for each food it names. From both facts, the validation split is
12 photos of `fries_p4`, 3 of `asparagus+chicken_p1` and 4 of `empty_p1`. We derived this
breakdown and did not count it directly. Instance counts for the training split are not
recorded in the repository.

## Labeling process and QA

1. **Auto-labels** (`labeling/autolabel.py`, run in `notebooks/colab_autolabel.ipynb`). The
   labeler runs one single-phrase Grounding DINO base pass per class (`plate`, `bowl`, each
   named food), and SAM 2.1 small turns each box into a mask. Filters: one plate (score x
   area), food must be mostly on the plate's convex hull, a "bowl" sitting on the plate is
   dropped, whole-plate food masks are dropped, duplicate masks are dropped, and slivers under
   64 px are dropped. Masks are exported as YOLO polygons. No mask was drawn or edited by hand.
2. **Automatic exclusion** (`missing_items`): a photo is dropped if any food named in its file
   name got no mask. Training on it would teach that the food is background. This removed 29
   photos, mostly the chicken on asparagus+chicken plates.
3. **Visual review**: every photo's overlay was inspected in a grid, and 7 more were dropped:

   | Reason | Photos |
   | --- | ---: |
   | Cutlery labelled as food | 2 |
   | Burger wrapper labelled as burger | 2 |
   | Fries only partly masked | 2 |
   | Wrong plate chosen | 1 |

QA limits: the review only removed photos and never corrected masks. One reviewer pass found
these 7, so remaining label errors are possible. There is no hand-drawn ground truth, so label
quality is not measured.

## Splits and leakage controls

- **Split unit is the plate group, not the photo.** `split_groups` (seed 0) ranks each dish's
  groups by `sha256("0:<group>")` and holds out about 20% of them, at least one per dish. The
  split covers only dishes with 2 or more groups. All photos of a held-out plate, crops
  included, go to validation, so no near-duplicate of a validation plate is in training.
- **Held-out groups**: `asparagus+chicken_p1`, `empty_p1`, `fries_p4`.
- **Dishes with only one plate group are never held out**, so rice pilaf, burger, short ribs
  and crepe have no held-out evaluation. Carrots, hot dog and any chicken-only plates are not
  in the validation groups either.
- **Composites use training data only.** `augment/dataset.py` reads only `images/train`, cuts
  food only from training photos, and writes `comp_XXXX.jpg` only to the training split.
  Validation is real photos only.
- **Caveat**: the validation split was also used for early stopping and best-epoch selection.
  No separate test split exists.

## Augmentation (training only)

- **Copy-paste composites** (`augment/composite.py`, `python -m augment.dataset --n 200 --seed 0`).
  Each composite pastes 1 to 3 food cutouts from training photos onto a training plate. Each
  cutout is scaled to the target plate's size (+/-20%) and rotated at random. Pastes must lie
  at least 90% on the plate and overlap earlier pastes by at most 30%. Labels are trimmed to
  what stays visible. Edges are feathered. Composites turn single-food photos into mixed plates.
- **Lighting changes**: `photometric_jitter` varies brightness, contrast, colour temperature,
  gamma, noise and blur. It never shifts hue, because colour separates carrots, asparagus and
  fries.
- **Ultralytics training-time augmentation**: rotation of 180 degrees, flips, mosaic, mixup 0.1,
  copy-paste 0.5, brightness 0.4 (`vision/configs/yolo26n_seg.yaml`).

## Known gaps and biases

- **One hall, one evening, one plate type.** Every photo comes from Village on 14 Nov 2026 on
  the white 8.5 in plate. Other meals, days, menus, dishware and halls are absent.
- **Lighting and camera**: the photos were taken with a hand-held iPhone under the hall's
  evening lighting. The rig camera, its mounting height and angle, and daylight conditions are
  not represented.
- **Backgrounds**: only dark tables and steel counters appear. The real dish-return belt or
  trays may look different.
- **Food states**: there are few plates per dish, and the split between full, partly eaten,
  scraps and empty plates per dish is unknown.
- **Survivor bias from exclusions**: 29 of 121 photos were dropped because the teacher missed a
  food, mostly chicken. The kept set is biased toward easy-to-segment food.
- **Crops inflate the count.** 53 of the 121 raw files are re-framings, so there are fewer
  distinct views than photos. How many crops survived into the final 85 is unknown.
- **Class coverage**: 9 photographed dishes map to 7 of the 21 food classes. The other 14 food
  classes have no Village photos at all.
- **Starting checkpoint**: the earlier Voxel51 data comes from a German canteen. Its masks were
  produced automatically by YOLO-E, and it reaches only 9 of the 21 food classes
  (`labeling/voxel51.py`, `labeling/class_map.yaml`).

## Privacy

- The photos show plates of food on tables and counters. No people, faces, names, card IDs or
  student data were collected.
- The live rig is designed to drop any frame with a face before storage
  (`capture/face_guard.py`). That guard is being implemented now.

## How to regenerate

1. Put the photos in `MyDrive/leftover-photos.zip` with the file naming above, and put the
   repository in `MyDrive/leftover-lens.zip`.
2. Run `notebooks/colab_autolabel.ipynb` on a T4. It converts the photos, auto-labels them,
   auto-excludes incomplete photos and shows the review grid. Add the hand exclusions to
   `EXCLUDE`. It then splits by plate group and writes `MyDrive/village-dataset.zip`.
3. Run `notebooks/colab_train_village.ipynb`. It adds 200 composites to the training split,
   trains, runs `python -m eval.segmentation`, which writes `eval/results/<name>.json` and
   `.md`, and exports ONNX.

What stops an exact rebuild today:

- The raw photos are not in the repository.
- The names of the 7 hand-excluded photos are not recorded in the repository; the notebook ships with `EXCLUDE = set()`.
- `transformers` and `ultralytics` versions are not pinned (`pyproject.toml`).
- Grounding DINO and SAM outputs may vary slightly across library versions and GPUs.
