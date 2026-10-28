# load

The terms the load commands use. Look one up with `dataforge glossary <term>`.

## Declared Inputs

Aliases:
Related: Operator, Workspace Settings
Commands: load

The CSV paths the Operator lists in the `declared_inputs` Workspace Setting. With none listed, `load` reads any path. Once any are listed, `load` reads only those paths and refuses every other one, an edited copy of a declared file included, with `input_not_declared`. Ask the Operator to declare a new file.
