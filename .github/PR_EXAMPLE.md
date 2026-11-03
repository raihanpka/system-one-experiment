# PR example: a finished tracker description

Illustrative. Keep only the lines you finished, delete unchecked boxes, and paste real output.

Title: `fix(validate): allow blank cells in a field and check only the filled ones`

## Summary

A field with blank cells was rejected whole. Blank cells are now allowed, and every filled value must still pass the template type.

## Type

- [x] fix (bug fix)

## Changes

- Allow blank cells in `field_valid` and require at least one filled value.
- Add a test for a field with one blank cell among valid dates.

## Verification

- [x] `uv build` passes
- [x] `make lint` passes
- [x] `make test` passes

```
154 passed in 40.32s
```
