#![recursion_limit = "256"]
#![cfg_attr(feature = "nightly", feature(specialization))]
#![forbid(unsafe_code)]

use chrono;
use futures;
use image;
use meval;
use n5;
#[macro_use]
extern crate num_derive;
use num_traits;
#[macro_use]
extern crate prettytable;
use regex;
use serde_json;
#[cfg(feature = "nightly")]
extern crate serde_plain;
use strfmt;
use structopt;

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
use n5::{
    data_type_match,
    data_type_rstype_replace,
};
use num_traits::{
    FromPrimitive,
    ToPrimitive,
};
use prettytable::Table;
use structopt::StructOpt;

mod bench_read;
#[cfg(feature = "nightly")]
mod cast;
mod crop_blocks;
mod delete_uniform_blocks;
mod export;
mod import;
mod list;
mod map;
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
    /// Cast an existing dataset into a new dataset with a given
    /// data type.
    #[cfg(feature = "nightly")]
    #[structopt(name = "cast")]
    Cast(cast::CastOptions),
    /// Crop wrongly sized blocks to match dataset dimensions at the end of a
    /// given axis.
    #[structopt(name = "crop-blocks")]
    CropBlocks(crop_blocks::CropBlocksOptions),
    /// Delete blocks uniformly filled with a given value, such as empty blocks.
    #[structopt(name = "delete-uniform-blocks")]
    DeleteUniformBlocks(delete_uniform_blocks::DeleteUniformBlocksOptions),
    /// Export a sequence of image files from a series of z-sections.
    #[structopt(name = "export")]
    Export(export::ExportOptions),
    /// Import a sequence of image files as a series of z-sections into a 3D
    /// N5 dataset.
    #[structopt(name = "import")]
    Import(import::ImportOptions),
    /// Run simple math expressions mapping values to new datasets.
    /// For example, to clip values in a dataset:
    /// `map example.n5 dataset_in example.n5 dataset_out "min(128, x)"`
    /// Note that this converts back and forth to `f64` for the calculation.
    #[structopt(name = "map")]
    Map(map::MapOptions),
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

fn main() -> Result<()> {
    let opt = Options::from_args();

    #[rustfmt::skip]
    match opt.command {
        Command::List(ref ls_opt) =>
            list::ListCommand::run(&opt, ls_opt)?,
        Command::Stat(ref st_opt) =>
            stat::StatCommand::run(&opt, st_opt)?,
        Command::BenchRead(ref br_opt) =>
            bench_read::BenchReadCommand::run(&opt, br_opt)?,
        #[cfg(feature = "nightly")]
        Command::Cast(ref cast_opt) =>
            cast::CastCommand::run(&opt, cast_opt).unwrap(),
        Command::CropBlocks(ref crop_opt) =>
            crop_blocks::CropBlocksCommand::run(&opt, crop_opt)?,
        Command::DeleteUniformBlocks(ref dub_opt) =>
            delete_uniform_blocks::DeleteUniformBlocksCommand::run(&opt, dub_opt)?,
        Command::Export(ref exp_opt) =>
            export::ExportCommand::run(&opt, exp_opt)?,
        Command::Import(ref imp_opt) =>
            import::ImportCommand::run(&opt, imp_opt)?,
        Command::Map(ref m_opt) =>
            map::MapCommand::run(&opt, m_opt)?,
        Command::MapFold(ref mf_opt) =>
            map_fold::MapFoldCommand::run(&opt, mf_opt)?,
        Command::Recompress(ref com_opt) =>
            recompress::RecompressCommand::run(&opt, com_opt)?,
        Command::ValidateBlocks(ref vb_opt) =>
            validate_blocks::ValidateBlocksCommand::run(&opt, vb_opt)?,
    };

    Ok(())
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

fn bounded_slab_coord_iter(
    data_attrs: &DatasetAttributes,
    axis: usize,
    slab_coord: u64,
    min: &[u64],
    max: &[u64],
) -> (impl Iterator<Item = Vec<u64>>, usize) {
    let mut coord_ceil = max
        .iter()
        .zip(data_attrs.get_block_size().iter())
        .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
        .collect::<Vec<_>>();
    coord_ceil.remove(axis as usize);
    let mut coord_floor = min
        .iter()
        .zip(data_attrs.get_block_size().iter())
        .map(|(&d, &s)| d / u64::from(s))
        .collect::<Vec<_>>();
    coord_floor.remove(axis as usize);
    let total_coords = coord_floor
        .iter()
        .zip(coord_ceil.iter())
        .map(|(&min, &max)| max - min)
        .product::<u64>() as usize;

    let iter = coord_ceil
        .into_iter()
        .zip(coord_floor.into_iter())
        .map(|(c, f)| f..c)
        .multi_cartesian_product()
        .map(move |mut c| {
            c.insert(axis as usize, slab_coord);
            c
        });

    (iter, total_coords)
}

fn slab_coord_iter(
    data_attrs: &DatasetAttributes,
    axis: usize,
    slab_coord: u64,
) -> (impl Iterator<Item = Vec<u64>>, usize) {
    bounded_slab_coord_iter(
        data_attrs,
        axis,
        slab_coord,
        &vec![0; data_attrs.get_ndim()],
        data_attrs.get_dimensions(),
    )
}

/// Convience trait combined all the required traits on data types until trait
/// aliases are stabilized.
trait DataTypeBounds:
    'static
    + ReflectedType
    + Sync
    + Send
    + std::fmt::Debug
    + PartialEq
    + num_traits::NumCast
    + num_traits::Zero
    + num_traits::ToPrimitive
{
}
impl<T> DataTypeBounds for T
where
    T: 'static
        + ReflectedType
        + Sync
        + Send
        + std::fmt::Debug
        + PartialEq
        + num_traits::NumCast
        + num_traits::Zero
        + num_traits::ToPrimitive,
    VecDataBlock<T>: n5::DataBlock<T>,
{
}

/// Trait for mapping individual blocks within `BlockReaderMapReduce`.
/// Factored as a trait rather than a generic method in order to allow
/// specialization, and potentially reuse.
trait BlockTypeMap<T>
where
    T: DataTypeBounds,
{
    type BlockResult: Send + 'static;
    type BlockArgument: Send + Sync + 'static;

    fn map<N5>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: GridCoord,
        block: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static;
}

trait BlockMap<Res: Send + 'static, Arg: Send + Sync + 'static>:
    BlockTypeMap<u8, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<u16, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<u32, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<u64, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<i8, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<i16, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<i32, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<i64, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<f32, BlockResult = Res, BlockArgument = Arg>
    + BlockTypeMap<f64, BlockResult = Res, BlockArgument = Arg>
{
}

impl<Res: Send + 'static, Arg: Send + Sync + 'static, T> BlockMap<Res, Arg> for T where
    T: BlockTypeMap<u8, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<u16, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<u32, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<u64, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<i8, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<i16, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<i32, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<i64, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<f32, BlockResult = Res, BlockArgument = Arg>
        + BlockTypeMap<f64, BlockResult = Res, BlockArgument = Arg>
{
}

trait BlockReaderMapReduce {
    type BlockResult: Send + 'static;
    type BlockArgument: Send + Sync + 'static;
    type ReduceResult;
    // TODO: When associated type default stabilize, `Map` should default to
    // `Self`.
    type Map: BlockMap<Self::BlockResult, Self::BlockArgument>;

    fn setup<N5>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _arg: &mut Self::BlockArgument,
    ) -> Result<()>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        Ok(())
    }

    fn coord_iter(
        data_attrs: &DatasetAttributes,
        _arg: &Self::BlockArgument,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize) {
        let coord_iter = data_attrs.coord_iter();
        let total_coords = coord_iter.len();

        (Box::new(coord_iter), total_coords)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        arg: &Self::BlockArgument,
    ) -> Self::ReduceResult;

    fn map_type_dispatch<N5>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: GridCoord,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        use std::cell::RefCell;
        data_type_match! {
            *data_attrs.get_data_type(),
            {
                thread_local! {
                    pub static BUFFER: RefCell<Option<VecDataBlock<RsType>>> = RefCell::new(None)
                };
                BUFFER.with(|maybe_block| {
                    match *maybe_block.borrow_mut() {
                        ref mut m @ None => {
                            let res_block = n.read_block(dataset, data_attrs, coord.clone());
                            let pass_block = res_block.map(|maybe_block| {
                                *m = maybe_block;
                                m.as_ref()
                            });

                            Self::Map::map(n, dataset, data_attrs, coord, pass_block, arg)
                        },
                        Some(ref mut old_block) => {
                            let res_present = n.read_block_into(
                                dataset, data_attrs, coord.clone(), old_block);
                            let pass_block = res_present.map(|present| present.map(|_| &*old_block));

                            Self::Map::map(n, dataset, data_attrs, coord, pass_block, arg)
                        }
                    }
                })
            }
        }
    }

    fn run<N5>(
        n: &N5,
        dataset: &str,
        pool_size: Option<usize>,
        mut arg: Self::BlockArgument,
    ) -> Result<Self::ReduceResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
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
                    &local.n,
                    &local.dataset,
                    &local.data_attrs,
                    coord.into(),
                    &local.arg,
                )?;
                local.pbar.write().unwrap().inc(1);
                Ok(block_result)
            }));
        }

        let block_results = futures::future::join_all(all_jobs).wait()?;

        scoped.pbar.write().unwrap().finish();
        Ok(Self::reduce(&scoped.data_attrs, block_results, &scoped.arg))
    }
}
