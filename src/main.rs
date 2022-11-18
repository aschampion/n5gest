#![recursion_limit = "256"]
#![forbid(unsafe_code)]

use indicatif::{
    ProgressBar,
    ProgressDrawTarget,
    ProgressStyle,
};
use num_derive::{
    FromPrimitive,
    ToPrimitive,
};
use num_traits::{
    FromPrimitive,
    ToPrimitive,
};
use structopt::StructOpt;

mod command;
mod common {
    pub use std::io::Result;
    pub use std::sync::Arc;
    pub use std::time::Instant;

    pub use anyhow::Context;
    pub use indicatif::{
        HumanBytes,
        HumanDuration,
    };
    pub use n5::prelude::*;
    pub use prettytable::{
        cell,
        row,
    };
    pub use structopt::StructOpt;

    pub(crate) use crate::{
        command::CommandType,
        processing::{
            BlockReaderMapReduce,
            BlockTypeMap,
            DataTypeBounds,
        },
        GridBoundsOption,
        Options,
    };

    pub(crate) fn from_plain_or_json_str<'a, T: serde::Deserialize<'a> + std::str::FromStr>(
        serial: &'a str,
    ) -> serde_json::Result<T> {
        serial
            .parse::<T>()
            .or_else(|_| serde_json::from_str(serial))
    }
}
mod iterator;
mod pool;
mod processing;

/// Utilities for N5 files.
#[derive(StructOpt, Debug)]
#[structopt(
    name = "n5gest",
    author,
    global_settings(&[
        structopt::clap::AppSettings::ColoredHelp,
        structopt::clap::AppSettings::VersionlessSubcommands,
    ])
)]
struct Options {
    /// Number of threads for parallel processing.
    /// By default, the number of CPU cores is used.
    #[structopt(short = "t", long = "threads")]
    threads: Option<usize>,
    #[structopt(subcommand)]
    command: command::Command,
}

#[derive(FromPrimitive, ToPrimitive)]
enum MetricPrefix {
    None = 0,
    Kilo,
    Mega,
    Giga,
    Tera,
    Peta,
    Exa,
    Zetta,
    Yotta,
}

impl MetricPrefix {
    fn reduce(mut number: usize) -> (usize, MetricPrefix) {
        let mut order = MetricPrefix::None.to_usize().unwrap();
        let max_order = MetricPrefix::Yotta.to_usize().unwrap();

        while number > 10_000 && order <= max_order {
            number /= 1_000;
            order += 1;
        }

        (number, MetricPrefix::from_usize(order).unwrap())
    }
}

impl std::fmt::Display for MetricPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MetricPrefix::None => " ",
                MetricPrefix::Kilo => "K",
                MetricPrefix::Mega => "M",
                MetricPrefix::Giga => "G",
                MetricPrefix::Tera => "T",
                MetricPrefix::Peta => "P",
                MetricPrefix::Exa => "E",
                MetricPrefix::Zetta => "Z",
                MetricPrefix::Yotta => "Y",
            }
        )
    }
}

fn main() -> anyhow::Result<()> {
    let opt = Options::from_args();

    command::dispatch(&opt)
}

fn default_progress_bar(size: u64) -> ProgressBar {
    let pbar = ProgressBar::new(size);
    pbar.set_draw_target(ProgressDrawTarget::stderr());
    pbar.set_style(ProgressStyle::default_bar().template(
        "[{elapsed_precise}] [{wide_bar:.cyan/blue}] \
         {bytes}/{total_bytes} ({percent}%) [{eta_precise}]",
    ));

    pbar
}

/// Common structopt options for commands that support grid coordinate bounds
/// for their coordinate iterators.
#[derive(StructOpt, Debug)]
struct GridBoundsOption {
    /// Axis along which to bound the grid coordinate slab.
    #[structopt(long = "slab-axis")]
    axis: Option<usize>,
    /// Grid coordinate of the slab to bound to along the axis given by `slab-axis`.
    /// Equivalent to `--slab-min N --slab-max N + 1`.
    #[structopt(
        long = "slab-coord",
        requires("axis"),
        conflicts_with("slab_min"),
        conflicts_with("slab_max")
    )]
    slab_coord: Option<u64>,
    /// Min grid coordinate of the slab to bound to along the axis given by `slab-axis`.
    #[structopt(long = "slab-min", requires("axis"))]
    slab_min: Option<u64>,
    /// Max grid coordinate (exclusive) of the slab to bound to along the axis given by `slab-axis`.
    #[structopt(long = "slab-max", requires("axis"))]
    slab_max: Option<u64>,
}

impl GridBoundsOption {
    fn to_factory(&self) -> anyhow::Result<Box<dyn iterator::CoordIteratorFactory>> {
        match (self.axis, self.slab_coord) {
            (Some(axis), Some(slab_coord)) => Ok(Box::new(iterator::GridSlabCoordIter {
                axis,
                slab: (Some(slab_coord), Some(slab_coord + 1)),
            })),
            (Some(axis), None) => {
                if let (Some(a), Some(b)) = (self.slab_min, self.slab_max) {
                    if b <= a {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Slab max coord must be greater than slab min coord",
                        )
                        .into());
                    }
                }
                Ok(Box::new(iterator::GridSlabCoordIter {
                    axis,
                    slab: (self.slab_min, self.slab_max),
                }))
            }
            _ => Ok(Box::new(iterator::DefaultCoordIter {})),
        }
    }
}

/// Container with dataset and attributes, for argument brevity.
struct ContainedDataset<N5> {
    container: N5,
    name: String,
    attrs: n5::DatasetAttributes,
}
