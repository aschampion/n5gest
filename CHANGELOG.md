# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [0.3.9] - 2020-11-10
### Added
- New command: `slice-img` exports an image from a 2D slice. Unlike `export`
  this supports N-dimensional volumes and arbitrary slicing dimensions.
- Commands which take slabbing options now accept ranges of slab coordinates
  via `--slab-min` and `--slab-max`.

## [0.3.8] - 2020-11-04
### Added
- `ls` now takes options for sorting the listing table and sorts by default by
  a `numeric-suffix` variation of lexicographic that attempts to order the first
  different path component between two datasets by the order of its numeric
  suffix if it exists. This properly sorts scale level datasets like `s9` and
  `s10`.
- `ls` now optionally lists nested datasets with the `nested` option.
- `stat` now lists ratio of block and dataset byte sizes versus max and
  the percentage of occupied blocks.

## [0.3.7] - 2020-10-26
### Added
- `ls` now takes a second optional argument for a group path to restrict listing
  to.

### Fixed
- Fixed performance regression in upstream n5.

## [0.3.6] - 2020-10-21
### Changed
- `ls` now bolds only the part of the dataset path that has changed from the
  previous entry, which helps to visualize large, complex hierarchies.

### Fixed
- Incorrect handling of absolute paths.

## [0.3.5] - 2020-10-20
### Changed
- Dataset paths can now be absolute (with a leading `/`)

### Fixed
- `ls` now traverses symlinks.

## [0.3.4] - 2020-07-20
### Changed
- `cast` is now available on stable Rust without the `nightly` feature.

### Removed
- The `nightly` feature is removed. Technically this is a breaking change but
  should be trivial to fix.

## [0.3.3] - 2020-06-28
### Fixed
- Error reporting update for `cast`. Fixes `nightly` feature builds.

## [0.3.2] - 2020-06-12
### Fixed
- `import` supports 16-bit images.
- Correct max block coordinates in `stat`.
- Several upstream bugs corrupting some PNGs and TIFFs during import have been
  fixed.
- Error reporting improved.

### Added
- New command: `import-tiff`. Imports single file tiff stacks.
- Many commands now accept options that bound their operation to slabs along
  particular axes. See help for the `--slab-axis` and `--slab-coord` options.

### Changed
- `ls` sorts printed paths lexicographically.
- Compression options can be passed via simple strings like `gzip`, which use
  default parameters.

## [0.3.1] - 2020-01-18
### Fixed
- Fixed some cases of corruption when PNGs are exported. However, some errors
  in the underlying deflate implementation are not yet resolved, so output
  should be inspected.
- `crop` now writes to the correct dataset in the output N5 rather than the
  input dataset name.
- `stat` now works on most platforms.

### Added
- New command: `delete-uniform-blocks`
- New command: `map`. Maps datasets elementwise to new datasets with an
  expression.
- `import`: option to elide uniform blocks.
- `ls`: include grid extent and max block count.
- `stat`: report average statistics in addition to min and max.
- `stat`: also report block coordinate statistics.
