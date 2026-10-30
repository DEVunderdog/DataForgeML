# profile

The terms the profile commands use. Look one up with `dataforge glossary <term>`.

## Dataset Digest

Aliases: Digest
Related: Dial Pointer, Handle
Commands: profile

What `profile` returns: the dataset at low resolution, a summary card plus one row per column, flagged columns first, with no cell values. A flag is named `<group>.<flag>` after the `thresholds` group that decides it, and the Digest lists each flag once with its Dial Pointers. The detail commands, such as `describe-columns` and `value-summary`, read the rest off the profile Handle.
