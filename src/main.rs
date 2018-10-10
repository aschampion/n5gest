extern crate futures;
extern crate futures_cpupool;
extern crate indicatif;
extern crate itertools;
extern crate n5;
#[macro_use]
extern crate num_derive;
extern crate num_traits;
#[macro_use]
extern crate prettytable;
extern crate serde_json;
// This `macro_use` is linted in beta and nightly, but is necessary for stable.
#[macro_use]
extern crate structopt;


use std::io::Result;
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
    List {
        /// N5 root path
        #[structopt(name = "N5")]
        n5_path: String,
    },
    /// Benchmark reading an entire dataset.
    #[structopt(name = "bench-read")]
    BenchRead {
        /// Input N5 root path
        #[structopt(name = "N5")]
        n5_path: String,
        /// Input N5 dataset
        #[structopt(name = "DATASET")]
        dataset: String,
    },
    /// Crop wrongly sized blocks to match dataset dimensions at the end of a
    /// given axis.
    #[structopt(name = "crop-blocks")]
    CropBlocks(CropBlocksOptions),
    /// Recompress an existing dataset into a new dataset with a given
    /// compression.
    #[structopt(name = "recompress")]
    Recompress(RecompressOptions),
    /// Report malformed blocks.
    #[structopt(name = "validate-blocks")]
    ValidateBlocks {
        /// Input N5 root path
        #[structopt(name = "N5")]
        n5_path: String,
        /// Input N5 dataset
        #[structopt(name = "DATASET")]
        dataset: String,
    },
}

#[derive(StructOpt, Debug)]
struct CropBlocksOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// Axis to check
    #[structopt(name = "AXIS")]
    axis: i32,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
}

#[derive(StructOpt, Debug)]
struct RecompressOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// New N5 compression (JSON)
    #[structopt(name = "COMPRESSION")]
    compression: String,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
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
        Command::List {n5_path} => {
            let n = N5Filesystem::open(&n5_path).unwrap();
            let mut group_stack = vec![("".to_owned(), n.list("").unwrap().into_iter())];

            let mut datasets = vec![];

            while let Some((mut g_path, mut g_iter)) = group_stack.pop() {
                if let Some(next_item) = g_iter.next() {
                    let path: String = if g_path.is_empty() {
                        next_item
                    } else {
                        g_path.clone() + "/" + &next_item
                    };
                    group_stack.push((g_path, g_iter));
                    if let Ok(ds_attr) = n.get_dataset_attributes(&path) {
                        datasets.push((path, ds_attr));
                    } else {
                        let next_g_iter = n.list(&path).unwrap().into_iter();
                        group_stack.push((path, next_g_iter));
                    }
                }
            }

            let mut table = Table::new();
            table.set_format(*prettytable::format::consts::FORMAT_CLEAN);
            table.set_titles(row![
                "Path",
                r -> "Dims",
                r -> "Max vox",
                r -> "Block",
                "Dtype",
                "Compression",
            ]);

            for (path, attr) in datasets {
                let numel = attr.get_dimensions().iter().map(|&n| n as usize).product();
                let (numel, prefix) = MetricPrefix::reduce(numel);
                table.add_row(row![
                    b -> path,
                    r -> format!("{:?}", attr.get_dimensions()),
                    r -> format!("{} {}", numel, prefix),
                    r -> format!("{:?}", attr.get_block_size()),
                    format!("{:?}", attr.get_data_type()),
                    attr.get_compression(),
                ]);
            }

            table.printstd();
        },
        Command::BenchRead {n5_path, dataset} => {
            let n = N5Filesystem::open(&n5_path).unwrap();
            let started = Instant::now();
            let num_bytes = BenchRead::run(
                &n,
                &dataset,
                opt.threads,
                ()).unwrap();
            let elapsed = started.elapsed();
            println!("Read {} (uncompressed) in {}",
                HumanBytes(num_bytes as u64),
                HumanDuration(elapsed));
            let throughput = 1e9 * (num_bytes as f64) /
                (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
            println!("({} / s)", HumanBytes(throughput as u64));
        },
        Command::CropBlocks(ref crop_opt) => {
            let n5_in = N5Filesystem::open_or_create(&crop_opt.input_n5_path).unwrap();
            let n5_out = N5Filesystem::open_or_create(&crop_opt.output_n5_path).unwrap();
            println!("Cropping along {}", crop_opt.axis);

            let started = Instant::now();
            let (num_blocks, num_bytes) = CropBlocks::run(
                &n5_in,
                &crop_opt.input_dataset,
                opt.threads,
                CropBlocksArguments {
                    n5_out,
                    dataset_out: crop_opt.output_dataset.to_owned(),
                    axis: crop_opt.axis,
                }).unwrap();
            println!("Converted {} blocks with {} (uncompressed) in {}",
                num_blocks,
                HumanBytes(num_bytes as u64),
                HumanDuration(started.elapsed()));
        },
        Command::Recompress(ref com_opt) => {
            let n5_in = N5Filesystem::open_or_create(&com_opt.input_n5_path).unwrap();
            let n5_out = N5Filesystem::open_or_create(&com_opt.output_n5_path).unwrap();
            let compression: CompressionType = serde_json::from_str(&com_opt.compression).unwrap();
            println!("Recompressing with {}", compression);

            let started = Instant::now();
            let num_bytes = Recompress::run(
                &n5_in,
                &com_opt.input_dataset,
                opt.threads,
                RecompressArguments {
                    n5_out,
                    dataset_out: com_opt.output_dataset.to_owned(),
                    data_attrs_out: None, // TODO: this is a hack.
                    compression,
                }).unwrap();
            let elapsed = started.elapsed();
            println!("Converted {} (uncompressed) in {}",
                HumanBytes(num_bytes as u64),
                HumanDuration(elapsed));
            let throughput = 1e9 * (num_bytes as f64) /
                (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
            println!("({} / s)", HumanBytes(throughput as u64));
        },
        Command::ValidateBlocks {n5_path, dataset} => {
            let n = N5Filesystem::open(&n5_path).unwrap();
            let started = Instant::now();
            let invalid = ValidateBlocks::run(
                &n,
                &dataset,
                opt.threads,
                ()).unwrap();
            if !invalid.errored.is_empty() {
                eprintln!("Found {} errored block(s)", invalid.errored.len());
                for block_idx in invalid.errored.iter() {
                    println!("{}", n.get_block_uri(&dataset, block_idx).unwrap());
                }
            }
            if !invalid.wrongly_sized.is_empty() {
                eprintln!("Found {} wrongly sized block(s)", invalid.wrongly_sized.len());
                for block_idx in invalid.wrongly_sized.iter() {
                    println!("{}", n.get_block_uri(&dataset, block_idx).unwrap());
                }
            }
            eprintln!("Found {} invalid block(s) in {}",
                invalid.errored.len() + invalid.wrongly_sized.len(),
                HumanDuration(started.elapsed()));
        },
    }
}


fn default_progress_bar(size: u64) -> ProgressBar {
    let pbar = ProgressBar::new(size);
    pbar.set_draw_target(ProgressDrawTarget::stderr());
    pbar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({percent}%) [{eta_precise}]"));

    pbar
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
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send + num_traits::Zero,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T>;

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
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
        Ok(Self::reduce(&scoped.data_attrs, block_results))
    }
}

struct BenchRead;

impl BlockReaderMapReduce for BenchRead {
    type BlockResult = usize;
    type BlockArgument = ();
    type ReduceResult = usize;

    fn map<N5, T>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _coord: Vec<i64>,
        block_in: Result<Option<VecDataBlock<T>>>,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T> {

        let num_vox = match block_in? {
            Some(block) => {
                block.get_num_elements() as usize
            },
            None => 0,
        };

        Ok(num_vox)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
    ) -> Self::ReduceResult {

        let num_vox: usize = results.iter().sum();

        num_vox * data_attrs.get_data_type().size_of()
    }
}


struct CropBlocks<N5O> {
    _phantom: std::marker::PhantomData<N5O>,
}

#[derive(Clone)]
struct CropBlocksArguments<N5O: N5Writer + Sync + Send + Clone + 'static> {
    n5_out: N5O,
    dataset_out: String,
    axis: i32,
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static> BlockReaderMapReduce for CropBlocks<N5O> {
    type BlockResult = Option<usize>;
    type BlockArgument = CropBlocksArguments<N5O>;
    type ReduceResult = (usize, usize);

    fn setup<N5> (
        _n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        arg: &mut Self::BlockArgument,
    ) -> Result<()>
        where N5: N5Reader + Sync + Send + Clone + 'static {

        arg.n5_out.create_dataset(dataset, data_attrs)
    }

    fn coord_iter(
        data_attrs: &DatasetAttributes,
        arg: &Self::BlockArgument,
    ) -> (Box<Iterator<Item = Vec<i64>>>, usize) {

        let axis = arg.axis;
        let mut coord_ceil = data_attrs.get_dimensions().iter()
            .zip(data_attrs.get_block_size().iter())
            .map(|(&d, &s)| (d + i64::from(s) - 1) / i64::from(s))
            .collect::<Vec<_>>();
        let axis_ceil = coord_ceil.remove(axis as usize);
        let total_coords = coord_ceil.iter().product::<i64>() as usize;
        let coord_iter = coord_ceil.into_iter()
            .map(|c| 0..c)
            .multi_cartesian_product()
            .map(move |mut c| {
                c.insert(axis as usize, axis_ceil - 1);
                c
            });

        (Box::new(coord_iter), total_coords)
    }

    fn map<N5, T>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: Vec<i64>,
        block_opt: Result<Option<VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send + num_traits::Zero,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T> {

        let num_vox = match block_opt? {
            Some(_) => {
                // TODO: only reading block because it is the only way currently
                // to test block existence. To be more efficient could either
                // use another means, or crop from this read block directly rather
                // than re-reading using the ndarray convenience method.

                let (offset, size): (Vec<i64>, Vec<i64>) = data_attrs.get_dimensions().iter()
                            .zip(data_attrs.get_block_size().iter().cloned().map(i64::from))
                            .zip(coord.iter())
                            .map(|((&d, s), &c)| {
                                let offset = c * s;
                                let size = std::cmp::min((c + 1) * s, d) - offset;
                                (offset, size)
                            })
                            .unzip();

                let bbox = BoundingBox::new(offset, size.clone());

                let cropped = n.read_ndarray::<T>(
                    dataset,
                    data_attrs,
                    &bbox)?;
                assert!(!cropped.is_standard_layout(),
                    "Array should still be in f-order");
                let cropped_block = VecDataBlock::<T>::new(
                    size.into_iter().map(|n| n as i32).collect(),
                    coord,
                    cropped.as_slice_memory_order().unwrap().to_owned());
                arg.n5_out.write_block(dataset, data_attrs, &cropped_block)?;
                Some(cropped_block.get_num_elements() as usize)
            },
            None => None,
        };

        Ok(num_vox)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
    ) -> Self::ReduceResult {

        let (num_blocks, num_vox): (usize, usize) = results.iter()
            .fold((0, 0), |(blocks, total), vox| if let Some(count) = vox {
                    (blocks + 1, total + count)
                } else {
                    (blocks, total)
                }
            );

        (num_blocks, num_vox * data_attrs.get_data_type().size_of())
    }
}


struct Recompress<N5O> {
    _phantom: std::marker::PhantomData<N5O>,
}

#[derive(Clone)]
struct RecompressArguments<N5O: N5Writer + Sync + Send + Clone + 'static> {
    n5_out: N5O,
    data_attrs_out: Option<DatasetAttributes>,
    dataset_out: String,
    compression: CompressionType,
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static> BlockReaderMapReduce for Recompress<N5O> {
    type BlockResult = usize;
    type BlockArgument = RecompressArguments<N5O>;
    type ReduceResult = usize;

    fn setup<N5> (
        _n: &N5,
        _dataset: &str,
        data_attrs: &DatasetAttributes,
        arg: &mut Self::BlockArgument,
    ) -> Result<()>
        where N5: N5Reader + Sync + Send + Clone + 'static {

        arg.data_attrs_out = Some(DatasetAttributes::new(
            data_attrs.get_dimensions().to_vec(),
            data_attrs.get_block_size().to_vec(),
            *data_attrs.get_data_type(),
            arg.compression.clone()));
        arg.n5_out.create_dataset(&arg.dataset_out, arg.data_attrs_out.as_ref().unwrap())
    }

    fn map<N5, T>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _coord: Vec<i64>,
        block_opt: Result<Option<VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T> {

        let num_vox = match block_opt? {
            Some(block) => {
                arg.n5_out.write_block(&arg.dataset_out, &arg.data_attrs_out.as_ref().unwrap(), &block)?;
                block.get_num_elements() as usize
            },
            None => 0,
        };

        Ok(num_vox)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
    ) -> Self::ReduceResult {

        let num_vox: usize = results.iter().sum();

        num_vox * data_attrs.get_data_type().size_of()
    }
}


struct InvalidBlocks {
    errored: Vec<Vec<i64>>,
    wrongly_sized: Vec<Vec<i64>>,
}

impl Default for InvalidBlocks {
    fn default() -> Self {
        Self {
            errored: vec![],
            wrongly_sized: vec![],
        }
    }
}

enum ValidationResult {
    Ok,
    Error(Vec<i64>),
    WrongSize(Vec<i64>),
}

struct ValidateBlocks;

impl BlockReaderMapReduce for ValidateBlocks {
    type BlockResult = ValidationResult;
    type BlockArgument = ();
    type ReduceResult = InvalidBlocks;

    fn map<N5, T>(
        _n: &N5,
        _dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: Vec<i64>,
        block_opt: Result<Option<VecDataBlock<T>>>,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
            DataType: TypeReflection<T> + DataBlockCreator<T>,
            VecDataBlock<T>: n5::DataBlock<T> {

        Ok(match block_opt {
            Ok(Some(block)) => {

                let expected_size: Vec<i32> = data_attrs.get_dimensions().iter()
                    .zip(data_attrs.get_block_size().iter().cloned().map(i64::from))
                    .zip(coord.iter())
                    .map(|((&d, s), &c)| (std::cmp::min((c + 1) * s, d) - c * s) as i32)
                    .collect();

                if expected_size == block.get_size() {
                    ValidationResult::Ok
                } else {
                    ValidationResult::WrongSize(coord)
                }
            },
            Ok(None) => ValidationResult::Ok,
            Err(_) => ValidationResult::Error(coord),
        })
    }

    fn reduce(
        _data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
    ) -> Self::ReduceResult {

        let mut invalid = InvalidBlocks::default();

        for result in results.into_iter() {
            match result {
                ValidationResult::Ok => {},
                ValidationResult::Error(v) => invalid.errored.push(v),
                ValidationResult::WrongSize(v) => invalid.wrongly_sized.push(v),
            }
        }

        invalid
    }
}
