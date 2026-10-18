# Coding Standards

Judgement calls that no linter or test can check. The code-review step holds every
diff against each rule here. Anything mechanical belongs in a linter, a type check,
or a contract test instead, so this file never lists it.

Each rule states what the code does and why.

## Reuse the shared helpers

A new Operation reuses what its sibling Operations already share. When a helper
exists, call it. When a second Operation needs logic that so far lives in one
place, extract it into a shared module and have both call it.

The shared helpers today:

- `_boundary/_detail.py`: `page` cuts a result page and builds its `omitted` entry;
  `unknown_columns` builds the one `unknown_column` error.
- `_boundary/_workspace.py`: the Handle readers (`read_record`, `read_frame`,
  `read_profile`, `read_routing`) and `write_handle`.

**Why:** a pasted copy drifts. The paging rule, the `unknown_column` error shape,
and the Handle checks are part of the Agent-facing contract. When one copy changes
and another does not, two Operations answer the same mistake in two ways. On #598 an
Operation pasted all three helpers, and only review caught it.

**In review:** flag a function in the diff that reproduces the shape of a shared
helper, even under a new name. The fix is to call the helper, or to extract the
shared shape.
