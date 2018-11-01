extern crate chrono;
extern crate futures;
extern crate futures_cpupool;
extern crate image;
extern crate indicatif;
extern crate itertools;
extern crate meval;
extern crate n5;
#[macro_use]
extern crate num_derive;
extern crate num_traits;
#[macro_use]
extern crate prettytable;
extern crate regex;
extern crate serde_json;
extern crate strfmt;
// This `macro_use` is linted in beta and nightly, but is necessary for stable.
#[macro_use]
extern crate structopt;


use std::io::Result;
use std::path::PathBuf;
use std::sync::{
    Arc,
    RwLock,
};
use std::time::Instant;

use futures::Future;
use futures_cpupool::{
    CpuFuture,
    CpuPool,
};
use indicatif::{
    HumanBytes,
    HumanDuration,
    ProgressBar,
    ProgressDrawTarget,
    ProgressStyle,
};
use itertools::Itertools;
use n5::prelude::*;
use n5::DataBlockCreator;
use num_traits::{
    FromPrimitive,
    ToPrimitive,
};
use prettytable::Table;
use structopt::StructOpt;


mod bench_read;
mod crop_blocks;
mod export;
mod import;
mod list;
mod map_fold;
mod recompress;
mod stat;
mod validate_blocks;


/// Utilities for N5 files.
#[derive(StructOpt, Debug)]
#[structopt(name = "n5gest")]
struct Options {
    /// Number of threads for parallel processing.
    /// By default, the number of CPU cores is used.
    #[structopt(short = "t", long = "threads")]
    threads: Option<usize>,
    #[structopt(subcommand)]
    command: Command,
}

#[derive(StructOpt, Debug)]
enum Command {
    /// List all datasets under an N5 root.
    #[structopt(name = "ls")]
    List(list::ListOptions),
    /// Retrieve metadata about the number of blocks that exists and their
    /// timestamps.
    #[structopt(name = "stat")]
    Stat(stat::StatOptions),
    /// Benchmark reading an entire dataset.
    #[structopt(name = "bench-read")]
    BenchRead(bench_read::BenchReadOptions),
    /// Crop wrongly sized blocks to match dataset dimensions at the end of a
    /// given axis.
    #[structopt(name = "crop-blocks")]
    CropBlocks(crop_blocks::CropBlocksOptions),
    /// Export a sequence of image files from a series of z-sections.
    #[structopt(name = "export")]
    Export(export::ExportOptions),
    /// Import a sequence of image files as a series of z-sections into a 3D
    /// N5 dataset.
    #[structopt(name = "import")]
    Import(import::ImportOptions),
    /// Run simple math expressions as folds over blocks.
    /// For example, to find the maximum value in a positive dataset:
    /// `map-fold example.n5 dataset 0 "max(acc, x)"`
    #[structopt(name = "map-fold")]
    MapFold(map_fold::MapFoldOptions),
    /// Recompress an existing dataset into a new dataset with a given
    /// compression.
    #[structopt(name = "recompress")]
    Recompress(recompress::RecompressOptions),
    /// Report malformed blocks.
    #[structopt(name = "validate-blocks")]
    ValidateBlocks(validate_blocks::ValidateBlocksOptions),
}

trait CommandType {
    type Options: StructOpt;

    fn run(opt: &Options, com_opt: &Self::Options) -> Result<()>;
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
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", match self {
            MetricPrefix::None => " ",
            MetricPrefix::Kilo => "K",
            MetricPrefix::Mega => "M",
            MetricPrefix::Giga => "G",
            MetricPrefix::Tera => "T",
            MetricPrefix::Peta => "P",
            MetricPrefix::Exa => "E",
            MetricPrefix::Zetta => "Z",
            MetricPrefix::Yotta => "Y",
        })
    }
}

fn main() {
    let opt = Options::from_args();

    match opt.command {
        Command::List(ref ls_opt) =>
            list::ListCommand::run(&opt, ls_opt).unwrap(),
        Command::Stat(ref st_opt) =>
            stat::StatCommand::run(&opt, st_opt).unwrap(),
        Command::BenchRead(ref br_opt) =>
            bench_read::BenchReadCommand::run(&opt, br_opt).unwrap(),
        Command::CropBlocks(ref crop_opt) =>
            crop_blocks::CropBlocksCommand::run(&opt, crop_opt).unwrap(),
        Command::Export(ref exp_opt) =>
            export::ExportCommand::run(&opt, exp_opt).unwrap(),
        Command::Import(ref imp_opt) =>
            import::ImportCommand::run(&opt, imp_opt).unwrap(),
        Command::MapFold(ref mf_opt) =>
            map_fold::MapFoldCommand::run(&opt, mf_opt).unwrap(),
        Command::Recompress(ref com_opt) =>
            recompress::RecompressCommand::run(&opt, com_opt).unwrap(),
        Command::ValidateBlocks(ref vb_opt) =>
            validate_blocks::ValidateBlocksCommand::run(&opt, vb_opt).unwrap(),
    }
}


fn default_progress_bar(size: u64) -> ProgressBar {
    let pbar = ProgressBar::new(size);
    pbar.set_draw_target(ProgressDrawTarget::stderr());
    pbar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{wide_bar:.cyan/blue}] \
            {bytes}/{total_bytes} ({percent}%) [{eta_precise}]"));

    pbar
}


fn slab_coord_iter(
    data_attrs: &DatasetAttributes,
    axis: usize,
    slab_coord: i64,
) -> (impl Iterator<Item = Vec<i64>>, usize) {

    let mut coord_ceil = data_attrs.get_dimensions().iter()
        .zip(data_attrs.get_block_size().iter())
        .map(|(&d, &s)| (d + i64::from(s) - 1) / i64::from(s))
        .collect::<Vec<_>>();
    coord_ceil.remove(axis as usize);
    let total_coords = coord_ceil.iter().product::<i64>() as usize;

    let iter = coord_ceil.into_iter()
        .map(|c| 0..c)
        .multi_cartesian_product()
        .map(move |mut c| {
            c.insert(axis as usize, slab_coord);
            c
        });

    (iter, total_coords)
}


trait BlockReaderMapReduce {
    type BlockResult: Send + 'static;
    type BlockArgument: Send + Sync + 'static;
    type ReduceResult;

    fn setup<N5> (
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _arg: &mut Self::BlockArgument,
    ) -> Result<()>
        where N5: N5Reader + Sync + Send + Clone + 'static {

        Ok(())
    }

    fn coord_iter(
        data_attrs: &DatasetAttributes,
        _arg: &Self::BlockArgument,
    ) -> (Box<Iterator<Item = Vec<i64>>>, usize) {

        let coord_iter = data_attrs.coord_iter();
        let total_coords = coord_iter.len();

        (Box::new(coord_iter), total_coords)
    }

    fn map<N5, T>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: Vec<i64>,
        block: Result<Option<VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send
              + num_traits::Zero + num_traits::ToPrimitive,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T>;

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        arg: &Self::BlockArgument,
    ) -> Self::ReduceResult;

    fn map_type_dispatch<N5>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: Vec<i64>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where N5: N5Reader + Sync + Send + Clone + 'static {

        match *data_attrs.get_data_type() {
            DataType::UINT8 => {
                let block = n.read_block::<u8>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::UINT16 => {
                let block = n.read_block::<u16>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::UINT32 => {
                let block = n.read_block::<u32>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::UINT64 => {
                let block = n.read_block::<u64>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::INT8 => {
                let block = n.read_block::<i8>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::INT16 => {
                let block = n.read_block::<i16>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::INT32 => {
                let block = n.read_block::<i32>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::INT64 => {
                let block = n.read_block::<i64>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::FLOAT32 => {
                let block = n.read_block::<f32>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
            DataType::FLOAT64 => {
                let block = n.read_block::<f64>(dataset, data_attrs, coord.clone());
                Self::map(n, dataset, data_attrs, coord, block, arg)
            },
        }
    }

    fn run<N5>(
        n: &N5,
        dataset: &str,
        pool_size: Option<usize>,
        mut arg: Self::BlockArgument,
    ) -> Result<Self::ReduceResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static {

        let data_attrs = n.get_dataset_attributes(dataset)?;

        Self::setup(n, dataset, &data_attrs, &mut arg)?;


        let (coord_iter, total_coords) = Self::coord_iter(&data_attrs, &arg);
        let pbar = RwLock::new(default_progress_bar(total_coords as u64));

        let mut all_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(total_coords);
        let pool = match pool_size {
            Some(threads) => CpuPool::new(threads),
            None => CpuPool::new_num_cpus(),
        };

        struct Scoped<N5, BlockArgument> {
            pbar: RwLock<ProgressBar>,
            n: N5,
            dataset: String,
            data_attrs: DatasetAttributes,
            arg: BlockArgument,
        }
        let scoped = Arc::new(Scoped {
            pbar,
            n: n.clone(),
            dataset: dataset.to_owned(),
            data_attrs,
            arg,
        });

        for coord in coord_iter {

            let local = scoped.clone();

            all_jobs.push(pool.spawn_fn(move || {
                let block_result = Self::map_type_dispatch(
                    &local.n, &local.dataset, &local.data_attrs, coord, &local.arg)?;
                local.pbar.write().unwrap().inc(1);
                Ok(block_result)
            }));
        }

        let block_results = futures::future::join_all(all_jobs).wait()?;

        scoped.pbar.write().unwrap().finish();
        Ok(Self::reduce(&scoped.data_attrs, block_results, &scoped.arg))
    }
}
