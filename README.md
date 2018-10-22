# n5gest

CLI utilities for working with [N5](https://github.com/saalfeldlab/n5) files.

Written in Rust using the [Rust N5 crate](https://crates.io/crates/n5).

```console
$ cargo build --release
$ target/release/n5gest -h
n5gest 0.1.0
Andrew Champion <andrew.champion@gmail.com>
Utilities for N5 files.

USAGE:
    n5gest [OPTIONS] <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -t, --threads <threads>    Number of threads for parallel processing. By default, the number of CPU
 cores is used.

SUBCOMMANDS:
    bench-read         Benchmark reading an entire dataset.
    crop-blocks        Crop wrongly sized blocks to match dataset dimensions at the end of a given axis.
    help               Prints this message or the help of the given subcommand(s)
    import             Import a sequence of image files as a series of z-sections into a 3D N5 dataset.
    ls                 List all datasets under an N5 root.
    map-fold           Run simple math expressions as folds over blocks. For example, to find the
                       maximum value in a positive dataset: `map-fold example.n5 dataset 0 "max(acc, x)"`
    recompress         Recompress an existing dataset into a new dataset with a given compression.
    stat               Retrieve metadata about the number of blocks that exists and their timestamps.
    validate-blocks    Report malformed blocks.
```

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
