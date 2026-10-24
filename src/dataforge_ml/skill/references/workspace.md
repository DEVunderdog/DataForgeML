# workspace

These commands read or export what the Workspace holds. None of them changes a Handle.

## list-handles

`list-handles` finds Handles made earlier: after a restart, as a sub-agent sharing the Workspace, or when you have lost a Handle string. Each Handle comes with its Lineage: the command that made it, the arguments it was passed and the Handles it read. Read the Lineage to tell two Handles of the same kind apart. `--kind` narrows to one kind: `frame`, `profile`, `routing`, `recipe`, `unit`, `imputer` or `c2st`. Handles never expire and are never deleted.

## query-frame

`query-frame` reads actual rows from any frame Handle: to check a sentinel is where you think, to see what a fill wrote, or to eyeball a fold.

- `--filter` is an OR of AND-lists: `[[{"column":"amount","op":"eq","value":-999}]]` keeps every row where `amount` is `-999`.
- `--mode` picks rows mode (the first rows, a seeded random sample, or one column's extremes) or aggregate mode (counts and statistics per group of up to 3 columns). `dataforge schema query-frame mode` lists the fields.
- Cells come back exactly as stored. A declared sentinel is a value like any other, so `-999` shows as `-999`. `missingness` gives the profile's view of what is missing.
- Under Literal Policy `deny`, cells, group keys, `min`, `max` and `median` come back `null`, but `matched_rows` and the counts stay. See [harness.md](harness.md).

## export-frame

`export-frame` writes a frame Handle as a CSV in the Workspace's `exports/` folder and returns its path. Use it to hand a filled frame or a fold to another tool. A CSV keeps no dtypes, so a tool reading it back may infer types again.
