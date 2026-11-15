### Segmentation: village-v2

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
