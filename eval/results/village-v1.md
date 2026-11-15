### Segmentation: village-v1

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
