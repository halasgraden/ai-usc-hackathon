### Segmentation runs compared: village-v1, village-v2

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
