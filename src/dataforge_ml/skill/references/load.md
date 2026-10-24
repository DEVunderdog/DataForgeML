# load

`load` turns one CSV into a frame Handle. Load each dataset once; every later command reads it through that Handle.

- Pass the path exactly as you were given it. A relative path resolves from the working directory.
- Loading the same path twice makes two Handles over identical data. Before loading again, run `dataforge list-handles --kind frame` and reuse the Handle you already have.
- With Declared Inputs set, `load` reads only the declared paths. Any other path fails with `input_not_declared` (exit 2), and so does an edited copy of a declared file. Ask the Operator to declare the file.
- `load` infers each column's dtype from the CSV. It decides nothing about meaning: semantic types, sentinels and targets are declared on `profile`.
- A file that cannot be read as CSV fails with `unexpected` (exit 1). Stop and report it.
