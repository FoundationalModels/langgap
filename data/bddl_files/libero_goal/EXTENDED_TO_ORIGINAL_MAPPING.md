# LIBERO Goal: extended-to-original mapping

Coverage: all **9 BDDL files** currently under [`extended/`](extended/), mapped against [`original/`](original/). Paths below are relative to this directory. Full-scene copies of each mapped original and extended BDDL are stored together under [`paired_original_extended/`](paired_original_extended/), with extended copies renamed `paired_ext_*.bddl`. Within a paired folder, only instruction and goal differ; the complete canonical object sets and table placements are shared. The paired copies preserve the canonical `:regions`, `:fixtures`, `:objects`, and `:init` sections described by the source initialization evidence below. The former `extended_ambiguous/` directory, which contained three exact duplicates of extended BDDLs, has been removed.

## Matching criteria

Match the instruction and goal according to the extension dimension, while requiring the same object and fixture initialization distributions. For object changes, keep the destination fixed; for target changes, keep the manipulated object fixed. Use the actual BDDL goal to resolve differences between filenames and instruction wording.

Parsed comparisons confirm that **all 9 extended and all 10 original BDDLs have identical `:regions`, `:fixtures`, `:objects`, and `:init` sections**, including placement ranges and yaw ranges. Consequently, the initial BDDL distribution alone cannot identify a unique original. This establishes equality of the declared starting distributions, not equality of sampled simulator states.

Where available, corresponding `.pruned_init` files under `langgap/data/init_files/libero_goal/` were compared by SHA-256. Six extended files are byte-identical to exactly one original initialization file each; these matches support the mappings below. The other three have no corresponding extended `.pruned_init` file at that path.

## Mapping and instruction changes

### 1. `ext_02_wine_bottle_to_stove.bddl`

- Extended file: [`extended/dim1_change_object/ext_02_wine_bottle_to_stove.bddl`](extended/dim1_change_object/ext_02_wine_bottle_to_stove.bddl)
- Paired folder: [`paired_original_extended/ext_02_wine_bottle_to_stove/`](paired_original_extended/ext_02_wine_bottle_to_stove/)
- Original file: [`original/put_the_bowl_on_the_stove.bddl`](original/put_the_bowl_on_the_stove.bddl)
- Original instruction: “Put the bowl on the stove”
- Extended instruction: “put the wine bottle on the stove”
- Change: Manipulated object: bowl → wine bottle. Stove destination stays fixed.
- Original goal: `(And (On akita_black_bowl_1 flat_stove_1_cook_region))`
- Extended goal: `(And (On wine_bottle_1 flat_stove_1_cook_region))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 2. `ext_03_cream_cheese_to_stove.bddl`

- Extended file: [`extended/dim1_change_object/ext_03_cream_cheese_to_stove.bddl`](extended/dim1_change_object/ext_03_cream_cheese_to_stove.bddl)
- Paired folder: [`paired_original_extended/ext_03_cream_cheese_to_stove/`](paired_original_extended/ext_03_cream_cheese_to_stove/)
- Original file: [`original/put_the_bowl_on_the_stove.bddl`](original/put_the_bowl_on_the_stove.bddl)
- Original instruction: “Put the bowl on the stove”
- Extended instruction: “put the cream cheese on the stove”
- Change: Manipulated object: bowl → cream cheese. Stove destination stays fixed.
- Original goal: `(And (On akita_black_bowl_1 flat_stove_1_cook_region))`
- Extended goal: `(And (On cream_cheese_1 flat_stove_1_cook_region))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 3. `ext_05_cream_cheese_to_cabinet.bddl`

- Extended file: [`extended/dim1_change_object/ext_05_cream_cheese_to_cabinet.bddl`](extended/dim1_change_object/ext_05_cream_cheese_to_cabinet.bddl)
- Paired folder: [`paired_original_extended/ext_05_cream_cheese_to_cabinet/`](paired_original_extended/ext_05_cream_cheese_to_cabinet/)
- Original file: [`original/put_the_bowl_on_top_of_the_cabinet.bddl`](original/put_the_bowl_on_top_of_the_cabinet.bddl)
- Original instruction: “Put the bowl on the top of the drawer”
- Extended instruction: “put the cream cheese on top of the cabinet”
- Change: Manipulated object: bowl → cream cheese. Cabinet-top destination stays fixed. The instruction also calls the fixture “cabinet” instead of “drawer”.
- Original goal: `(And (On akita_black_bowl_1 wooden_cabinet_1_top_side))`
- Extended goal: `(And (On cream_cheese_1 wooden_cabinet_1_top_side))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 4. `ext_06_wine_bottle_to_plate.bddl`

- Extended file: [`extended/dim1_change_object/ext_06_wine_bottle_to_plate.bddl`](extended/dim1_change_object/ext_06_wine_bottle_to_plate.bddl)
- Paired folder: [`paired_original_extended/ext_06_wine_bottle_to_plate/`](paired_original_extended/ext_06_wine_bottle_to_plate/)
- Original file: [`original/put_the_bowl_on_the_plate.bddl`](original/put_the_bowl_on_the_plate.bddl)
- Original instruction: “Put the bowl on the plate”
- Extended instruction: “put the wine bottle on the plate”
- Change: Manipulated object: bowl → wine bottle. Plate destination stays fixed.
- Original goal: `(And (On akita_black_bowl_1 plate_1))`
- Extended goal: `(And (On wine_bottle_1 plate_1))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 5. `ext_07_cream_cheese_to_plate.bddl`

- Extended file: [`extended/dim1_change_object/ext_07_cream_cheese_to_plate.bddl`](extended/dim1_change_object/ext_07_cream_cheese_to_plate.bddl)
- Paired folder: [`paired_original_extended/ext_07_cream_cheese_to_plate/`](paired_original_extended/ext_07_cream_cheese_to_plate/)
- Original file: [`original/put_the_bowl_on_the_plate.bddl`](original/put_the_bowl_on_the_plate.bddl)
- Original instruction: “Put the bowl on the plate”
- Extended instruction: “put the cream cheese on the plate”
- Change: Manipulated object: bowl → cream cheese. Plate destination stays fixed.
- Original goal: `(And (On akita_black_bowl_1 plate_1))`
- Extended goal: `(And (On cream_cheese_1 plate_1))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 6. `ext_09_cream_cheese_to_top_drawer.bddl`

- Extended file: [`extended/dim1_change_object/ext_09_cream_cheese_to_top_drawer.bddl`](extended/dim1_change_object/ext_09_cream_cheese_to_top_drawer.bddl)
- Paired folder: [`paired_original_extended/ext_09_cream_cheese_to_top_drawer/`](paired_original_extended/ext_09_cream_cheese_to_top_drawer/)
- Original file: [`original/open_the_top_drawer_and_put_the_bowl_inside.bddl`](original/open_the_top_drawer_and_put_the_bowl_inside.bddl)
- Original instruction: “Open the top layer of the drawer and put the bowl inside”
- Extended instruction: “open the top drawer and put the cream cheese inside”
- Change: Manipulated object: bowl → cream cheese. Top-drawer destination stays fixed.
- Original goal: `(And (In akita_black_bowl_1 wooden_cabinet_1_top_region))`
- Extended goal: `(And (In cream_cheese_1 wooden_cabinet_1_top_region))`
- Initialization evidence: Same BDDL initialization distribution; corresponding extended `.pruned_init` is absent. Mapping inferred from the extension dimension and goal.

### 7. `ext_11_bottom_drawer.bddl`

- Extended file: [`extended/dim1_change_object/ext_11_bottom_drawer.bddl`](extended/dim1_change_object/ext_11_bottom_drawer.bddl)
- Paired folder: [`paired_original_extended/ext_11_bottom_drawer/`](paired_original_extended/ext_11_bottom_drawer/)
- Original file: [`original/open_the_middle_drawer_of_the_cabinet.bddl`](original/open_the_middle_drawer_of_the_cabinet.bddl)
- Original instruction: “Open the middle layer of the drawer”
- Extended instruction: “open the bottom drawer of the cabinet”
- Change: Drawer acted on: middle → bottom. This is a drawer-target change despite being stored under dim1_change_object.
- Original goal: `(And (Open wooden_cabinet_1_middle_region))`
- Extended goal: `(And (Open wooden_cabinet_1_bottom_region))`
- Initialization evidence: Same BDDL initialization distribution; saved `.pruned_init` is byte-identical to this original.

### 8. `ext_12_bowl_to_stove_front.bddl`

- Extended file: [`extended/dim1_change_object/ext_12_bowl_to_stove_front.bddl`](extended/dim1_change_object/ext_12_bowl_to_stove_front.bddl)
- Paired folder: [`paired_original_extended/ext_12_bowl_to_stove_front/`](paired_original_extended/ext_12_bowl_to_stove_front/)
- Original file: [`original/push_the_plate_to_the_front_of_the_stove.bddl`](original/push_the_plate_to_the_front_of_the_stove.bddl)
- Original instruction: “Push the plate to the front of the stove”
- Extended instruction: “put the bowl in front of the stove”
- Change: Manipulated object: plate → bowl. Stove-front destination stays fixed. Wording also changes “Push … to the front” → “put … in front”.
- Original goal: `(And (On plate_1 main_table_stove_front_region))`
- Extended goal: `(And (On akita_black_bowl_1 main_table_stove_front_region))`
- Initialization evidence: Same BDDL initialization distribution; corresponding extended `.pruned_init` is absent. Mapping inferred from the extension dimension and goal.

### 9. `put_the_wine_bottle_in_front_of_the_stove.bddl`

- Extended file: [`extended/dim2_change_target/put_the_wine_bottle_in_front_of_the_stove.bddl`](extended/dim2_change_target/put_the_wine_bottle_in_front_of_the_stove.bddl)
- Paired folder: [`paired_original_extended/put_the_wine_bottle_in_front_of_the_stove/`](paired_original_extended/put_the_wine_bottle_in_front_of_the_stove/)
- Original file: [`original/put_the_wine_bottle_on_the_rack.bddl`](original/put_the_wine_bottle_on_the_rack.bddl)
- Original instruction: “Put the wine bottle on the rack”
- Extended instruction: “put the wine bottle in front of the stove”
- Change: Destination: rack → table region in front of the stove; wine bottle stays fixed. This is a representative semantic mapping; the cabinet-top original is equally compatible (see ambiguity below).
- Original goal: `(And (On wine_bottle_1 wine_rack_1_top_region))`
- Extended goal: `(And (On wine_bottle_1 main_table_stove_front_region))`
- Initialization evidence: Same BDDL initialization distribution; corresponding extended `.pruned_init` is absent. Mapping inferred from the extension dimension and goal.

## Ambiguity and source notes

- **Wine bottle → stove front (dim2):** both [`put_the_wine_bottle_on_the_rack.bddl`](original/put_the_wine_bottle_on_the_rack.bddl) and [`put_the_wine_bottle_on_top_of_the_cabinet.bddl`](original/put_the_wine_bottle_on_top_of_the_cabinet.bddl) preserve the wine bottle and change only the destination to `main_table_stove_front_region`. Both share the exact BDDL initialization distribution. The cabinet original’s instruction is “Put the wine bottle on the top of the drawer” and its goal is `(And (On wine_bottle_1 wooden_cabinet_1_top_side))`. Neither the current BDDL nor an available matching extended initialization file identifies a unique historical parent. The rack original in entry 9 is a representative choice, not verified provenance.
- **Cream cheese → cabinet:** the wine-bottle-to-cabinet original is also a possible object-change match semantically. The bowl-to-cabinet mapping is supported by the exact saved initialization-file match and the generator configuration.
- The existing [`LIBERO_GOAL_DIM1` configuration](../../../tools/generate_extended_bddl.py) explicitly identifies the parents for entries 1–5 and 7. It copies each parent’s saved initialization file. It does not describe the current top-drawer cream-cheese task, bowl-to-stove-front task, or wine-bottle-to-stove-front task. In particular, its historical `ext_09` refers to a plate-to-rack task, so extension numbers alone must not be used as parent identifiers.
- Original instructions above are transcribed from `:language`, including “layer of the drawer” and “top of the drawer” wording. Cabinet top (`wooden_cabinet_1_top_side`) and top drawer (`wooden_cabinet_1_top_region`) are distinct destinations. The top-drawer tasks encode only `In` in the final goal; the instruction’s “open” is not a separate final `Open` predicate.
