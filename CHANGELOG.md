# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


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