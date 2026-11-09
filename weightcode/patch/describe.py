"""Editor-facing description of the patch format (embedded in the editor prompt, HARNESS_SPEC section 6).

Every rule stated here is enforced by ``apply_patch``, and the tests apply every JSON example in the text.
"""

from __future__ import annotations

from weightcode.patch.files import MAX_PATCH_JSON_BYTES, NPY_NAME_PATTERN
from weightcode.patch.limits import Limits
from weightcode.patch.ops import SCHEMA


def _bytes(count: int) -> str:
    if count and count % 2**20 == 0:
        return f"{count // 2**20} MiB ({count} bytes)"
    return f"{count} bytes"


def _size_rule(limits: Limits) -> str:
    lines = ["- Size limits:"]
    if limits.max_edit_frac is None:
        lines.append("  there is no limit on the edit size;")
    else:
        lines.append(f"  the edit size, the count of numbers a patch specifies, is at most {limits.max_edit_frac!r} times the")
        lines.append("  number of parameter entries. Each op adds: scale_rows and scale_cols, the number of listed")
        lines.append("  indices; set and add, the number of index tuples (index form), or 1 with `value` and the block")
        lines.append("  size with `values_file` (slice form); project_out, the length of the direction; low_rank_add,")
        lines.append("  r * (rows + cols) of the 2-D view; add_dense, the tensor size; tie, the length of the group")
        lines.append("  map. A sparse change written as")
        lines.append("  add_dense counts the whole tensor, so write it in index form;")
    if limits.max_changed_frac is None:
        lines.append("  the number of changed entries is not limited;")
    else:
        lines.append(f"  at most a fraction {limits.max_changed_frac!r} of all parameter entries may change, counted on the")
        lines.append("  float32 result (project_out, low_rank_add and tie change almost every entry of their tensor);")
    if limits.max_rel_frob is None:
        lines.append("  there is no limit on the relative size of a tensor's change.")
    else:
        floor = limits.rel_frob_min_ndim
        lines.append(f"  every tensor of rank >= {floor} needs ||child - parent||_F / ||parent||_F <= {limits.max_rel_frob!r} on")
        if floor == 1:
            lines.append("  the float32 result.")
        elif floor == 2:
            lines.append("  the float32 result; 1-D tensors (biases, normalisation gains) are not limited by this rule.")
        else:
            lines.append(f"  the float32 result; tensors of rank below {floor} are not limited by this rule.")
        if limits.rel_frob_exempt:
            names = ", ".join(f"`{name}`" for name in limits.rel_frob_exempt)
            lines.append(f"  This rule does not apply to {names}.")
    return "\n".join(lines)


def describe_ops(limits: Limits | None = None) -> str:
    """Markdown description of the ``weightcode.patch.v0`` format with one JSON example per op.

    Numbers come from ``limits`` (``Limits()`` when None). The text is ASCII and short, for an editor prompt.
    """
    limits = Limits() if limits is None else limits
    return f"""# Patch format {SCHEMA}

A patch is one UTF-8 JSON object of at most {_bytes(MAX_PATCH_JSON_BYTES)} with exactly these keys:

```json
{{
  "schema": "{SCHEMA}",
  "parent_sha256": "<params_sha256 of the model you were shown>",
  "ops": [
    {{"op": "scale_rows", "tensor": "fc2.weight", "rows": [3, 17], "factor": 0.5}}
  ],
  "hypothesis": "why this edit should make the model right on more examples",
  "predicted_effect": "testable statement on accuracy and on the examples it already gets right"
}}
```

- `hypothesis` and `predicted_effect` are required non-empty strings. Unknown keys are errors, here and in ops.
- `ops` run in list order, each on the result of the ones before it; at most {limits.max_ops} ops.
- Arithmetic is float64, starting from the float32 parent; the result is cast to float32 once, at the end.
  An entry counts as changed only if its final float32 value differs from the parent's.
- Tensor names and layouts are those of model_spec.json (PyTorch layout: linear [out, in], conv
  [out, in, kh, kw]). W2 is the 2-D view T.reshape(shape[0], -1): one row per output unit or channel,
  columns over the flattened (in, kh, kw) in C order.
- Indices are 0-based, non-negative JSON integers (no negative indexing). Every number must be finite.
- A file field holds the bare name of a .npy file written with numpy.save in the same directory as the patch
  JSON: the name matches {NPY_NAME_PATTERN}, the file is regular (no symbolic link, no subdirectory),
  its dtype is integer or floating and its values are finite. The distinct files of one patch may total at
  most {_bytes(limits.max_file_bytes)}.
{_size_rule(limits)}
- Validation is all-or-nothing: nothing is applied unless the whole patch is valid, and every problem found
  is reported together.

## Ops

### scale_rows
`T[rows, ...] *= factor` on axis 0 (output units or output channels; the entries of a 1-D tensor). `rows` is
a non-empty list of distinct indices; `factor` is any finite number.
```json
{{"op": "scale_rows", "tensor": "fc2.weight", "rows": [3, 17], "factor": 0.5}}
```

### scale_cols
`T[:, cols, ...] *= factor` on axis 1 (input units, or input channels of a conv weight); the tensor must have
rank >= 2. `cols` is a non-empty list of distinct indices.
```json
{{"op": "scale_cols", "tensor": "fc3.weight", "cols": [40], "factor": 0.0}}
```

### set and add
`set` assigns and `add` increments, in one of two forms.
- Index form: `index` is a list of full index tuples (one integer per axis) and `values` has one number per
  tuple. A repeated tuple is an error for `set` and accumulates for `add`.
- Slice form: `slice` has one entry per axis, either `[start, stop]` with 0 <= start < stop <= axis length
  (stop exclusive) or `null` for the whole axis. Give exactly one of `value` (one number for the whole block)
  or `values_file` (a file whose shape equals the block shape).
```json
{{"op": "set", "tensor": "fc1.weight", "slice": [[10, 14], null], "value": 0.0}}
```
```json
{{"op": "add", "tensor": "fc3.weight", "index": [[4, 12], [4, 40]], "values": [0.1, -0.2]}}
```

### project_out
Removes a direction d from W2 with `strength` s in (0, 1]. `direction` is a file name or an inline list; it is
normalised to unit length and must not be zero.
- `side: "in"`: W2 <- W2 - s (W2 d) d^T, with d of length prod(shape[1:]).
- `side: "out"`: W2 <- W2 - s d (d^T W2), with d of length shape[0]. Optional `bias` (side "out" only): the
  name of the layer's 1-D bias, updated as b <- b - s d (d^T b).
```json
{{"op": "project_out", "tensor": "fc2.weight", "side": "in", "direction": "dir01.npy", "strength": 1.0}}
```

### low_rank_add
W2 <- W2 + u v^T, where `u` is a file of shape [shape[0], r], `v` a file of shape [prod(shape[1:]), r] and
1 <= r <= {limits.max_rank}.
```json
{{"op": "low_rank_add", "tensor": "fc2.weight", "u": "u.npy", "v": "v.npy"}}
```

### add_dense
T <- T + delta, where `delta` is a file of exactly the tensor's shape.
```json
{{"op": "add_dense", "tensor": "fc3.weight", "delta": "d_fc3.npy"}}
```

### tie
Pulls the rows (`axis: 0`) or the columns (`axis: 1`) of W2 toward the mean of their group:
W2 <- W2 + s (G - W2), where G replaces each row or column by the mean of the rows or columns in its group.
`groups` holds one integer group id in [0, L) per row or column (L = shape[0] for axis 0, prod(shape[1:]) for
axis 1), inline or as a file; a row or column alone in its group is unchanged. `strength` s is in (0, 1]; s = 1
replaces every member by its group mean. Axis 1 needs rank >= 2.
```json
{{"op": "tie", "tensor": "fc2.weight", "axis": 1, "groups": "groups.npy", "strength": 0.5}}
```
"""
