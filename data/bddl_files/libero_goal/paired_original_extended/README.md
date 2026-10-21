# Paired original and extended LIBERO Goal scenes

Each of the nine folders contains an original-task BDDL and a renamed
`paired_ext_*.bddl`. All 18 files use the complete canonical scenes from
`../original/` and `../extended/`: bowl, cream cheese, wine bottle, plate, and
all four fixtures (table, stove, cabinet, and wine rack). Object declarations,
regions, and initial placements have been restored from those source files.
The canonical source files are unchanged.

Both members of a pair have identical objects, fixtures, regions, initial
placements, and `:obj_of_interest`. They differ only in `:language` and `:goal`,
preserving each task's success condition. The shared `:obj_of_interest` remains
the union of both tasks' interest lists. Matching random seeds give matching
initial distributions; independent resets without a shared seed need not produce
the same pose.

| Pair folder | Original BDDL | Extended BDDL | Movable objects |
| --- | --- | --- | --- |
| `ext_02_wine_bottle_to_stove` | [put_the_bowl_on_the_stove.bddl](ext_02_wine_bottle_to_stove/put_the_bowl_on_the_stove.bddl) | [paired_ext_02_wine_bottle_to_stove.bddl](ext_02_wine_bottle_to_stove/paired_ext_02_wine_bottle_to_stove.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_03_cream_cheese_to_stove` | [put_the_bowl_on_the_stove.bddl](ext_03_cream_cheese_to_stove/put_the_bowl_on_the_stove.bddl) | [paired_ext_03_cream_cheese_to_stove.bddl](ext_03_cream_cheese_to_stove/paired_ext_03_cream_cheese_to_stove.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_05_cream_cheese_to_cabinet` | [put_the_bowl_on_top_of_the_cabinet.bddl](ext_05_cream_cheese_to_cabinet/put_the_bowl_on_top_of_the_cabinet.bddl) | [paired_ext_05_cream_cheese_to_cabinet.bddl](ext_05_cream_cheese_to_cabinet/paired_ext_05_cream_cheese_to_cabinet.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_06_wine_bottle_to_plate` | [put_the_bowl_on_the_plate.bddl](ext_06_wine_bottle_to_plate/put_the_bowl_on_the_plate.bddl) | [paired_ext_06_wine_bottle_to_plate.bddl](ext_06_wine_bottle_to_plate/paired_ext_06_wine_bottle_to_plate.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_07_cream_cheese_to_plate` | [put_the_bowl_on_the_plate.bddl](ext_07_cream_cheese_to_plate/put_the_bowl_on_the_plate.bddl) | [paired_ext_07_cream_cheese_to_plate.bddl](ext_07_cream_cheese_to_plate/paired_ext_07_cream_cheese_to_plate.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_09_cream_cheese_to_top_drawer` | [open_the_top_drawer_and_put_the_bowl_inside.bddl](ext_09_cream_cheese_to_top_drawer/open_the_top_drawer_and_put_the_bowl_inside.bddl) | [paired_ext_09_cream_cheese_to_top_drawer.bddl](ext_09_cream_cheese_to_top_drawer/paired_ext_09_cream_cheese_to_top_drawer.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_11_bottom_drawer` | [open_the_middle_drawer_of_the_cabinet.bddl](ext_11_bottom_drawer/open_the_middle_drawer_of_the_cabinet.bddl) | [paired_ext_11_bottom_drawer.bddl](ext_11_bottom_drawer/paired_ext_11_bottom_drawer.bddl) | bowl, cream cheese, wine bottle, plate |
| `ext_12_bowl_to_stove_front` | [push_the_plate_to_the_front_of_the_stove.bddl](ext_12_bowl_to_stove_front/push_the_plate_to_the_front_of_the_stove.bddl) | [paired_ext_12_bowl_to_stove_front.bddl](ext_12_bowl_to_stove_front/paired_ext_12_bowl_to_stove_front.bddl) | bowl, cream cheese, wine bottle, plate |
| `put_the_wine_bottle_in_front_of_the_stove` | [put_the_wine_bottle_on_the_rack.bddl](put_the_wine_bottle_in_front_of_the_stove/put_the_wine_bottle_on_the_rack.bddl) | [paired_ext_put_the_wine_bottle_in_front_of_the_stove.bddl](put_the_wine_bottle_in_front_of_the_stove/paired_ext_put_the_wine_bottle_in_front_of_the_stove.bddl) | bowl, cream cheese, wine bottle, plate |


`scene_manifest.json` records the full object sets, unchanged paths, and matching
semantic-variant entry keys. The [source mapping](../EXTENDED_TO_ORIGINAL_MAPPING.md)
describes original goal relationships and source initialization evidence. The
wine-bottle-to-stove-front task retains the rack original as its representative
pairing; historical parentage is unresolved.

Validation: all 18 BDDLs have balanced, parseable S-expressions. Their `:regions`,
`:fixtures`, `:objects`, and `:init` sections match their canonical source files.
Each pair still differs only in instruction and goal, and all filenames and task
instructions/goals are preserved. No simulator rollout was run for this restoration.
The previous reports and renders under
`../../../../../simulation/comparisons/libero_goal_paired_simplified/` describe
historical reduced scenes and do not represent these restored full scenes.
