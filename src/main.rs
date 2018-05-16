extern crate futures;
extern crate futures_cpupool;
extern crate indicatif;
extern crate n5;
#[macro_use]
extern crate num_derive;
extern crate num_traits;
#[macro_use]
extern crate prettytable;
extern crate serde_json;
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
};
use n5::prelude::*;
use n5::DataBlockCreator;
use num_traits::{
    FromPrimitive,
    ToPrimitive,
};
use prettytable::Table;
use structopt::StructOpt;
use std::path::Path;

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
            let num_bytes = bench_read(
                &n,
                &dataset,
                opt.threads).unwrap();
            let elapsed = started.elapsed();
            println!("Read {} (uncompressed) in {}",
                HumanBytes(num_bytes as u64),
                HumanDuration(elapsed));
            let throughput = 1e9 * (num_bytes as f64) /
                (1e9 * (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64));
            println!("({} / s)", HumanBytes(throughput as u64));
        },
        Command::Recompress(ref com_opt) => {
            let n5_in = N5Filesystem::open_or_create(&com_opt.input_n5_path).unwrap();
            let n5_out = N5Filesystem::open_or_create(&com_opt.output_n5_path).unwrap();
            let compression: CompressionType = serde_json::from_str(&com_opt.compression).unwrap();
            println!("Recompressing with {}", compression);

            let started = Instant::now();
            let num_bytes = recompress(
                &n5_in,
                &com_opt.input_dataset,
                &n5_out,
                &com_opt.output_dataset,
                compression,
                opt.threads).unwrap();
            println!("Converted {} (uncompressed) in {}",
                HumanBytes(num_bytes as u64),
                HumanDuration(started.elapsed()));
        },
        Command::ValidateBlocks {n5_path, dataset} => {
            let n = N5Filesystem::open(&n5_path).unwrap();
            let started = Instant::now();
            let invalid_blocks = get_invalid_blocks(
                &n,
                &dataset,
                opt.threads).unwrap();
            for block_idx in invalid_blocks.iter() {
                // todo: would prefer to use get_data_block_path, but it's private
                // println!("{}", n.get_data_block_path(&dataset, &block_idx)?.display());

                // todo: handle edge cases like get_data_block_path?
                let mut block_path = Path::new(&n5_path).join(&dataset);
                for val in block_idx.iter() {
                    block_path.push(val.to_string());
                }
                println!("{}", block_path.display());
            }
            eprintln!("Found {} invalid block(s) in {}",
                invalid_blocks.len(), HumanDuration(started.elapsed()));
        },
    }
}


fn bench_read<N5>(
    n: &N5,
    dataset: &str,
    pool_size: Option<usize>,
) -> Result<usize>
    where
        N5: N5Reader + Sync + Send + Clone + 'static, {

    let data_attrs = n.get_dataset_attributes(dataset)?;

    let mut all_jobs: Vec<CpuFuture<usize, std::io::Error>> =
        Vec::new();
    let pool = match pool_size {
        Some(threads) => CpuPool::new(threads),
        None => CpuPool::new_num_cpus(),
    };

    let coord_iter = data_attrs.coord_iter();
    let total_coords = coord_iter.len();
    let bar = Arc::new(RwLock::new(ProgressBar::new(total_coords as u64)));
    bar.write().unwrap().set_draw_target(ProgressDrawTarget::stderr());

    for coord in coord_iter {

        let n_c = n.clone();
        let dataset_c = dataset.to_owned();
        let data_attrs_c = data_attrs.clone();
        let bar_c = bar.clone();
        all_jobs.push(pool.spawn_fn(move || {
            // TODO: Have to work around annoying reflection issue.
            let num_vox = match *data_attrs_c.get_data_type() {
                DataType::UINT8 => bench_read_block::<u8, _>(
                    &n_c,
                    &dataset_c,
                    &data_attrs_c,
                    coord)?,
                _ => unimplemented!(),
                // DataType::UINT16 => std::mem::size_of::<u16>(),
                // DataType::UINT32 => std::mem::size_of::<u32>(),
                // DataType::UINT64 => std::mem::size_of::<u64>(),
                // DataType::INT8 => std::mem::size_of::<i8>(),
                // DataType::INT16 => std::mem::size_of::<i16>(),
                // DataType::INT32 => std::mem::size_of::<i32>(),
                // DataType::INT64 => std::mem::size_of::<i64>(),
                // DataType::FLOAT32 => std::mem::size_of::<f32>(),
                // DataType::FLOAT64 => std::mem::size_of::<f64>(),
            };
            bar_c.write().unwrap().inc(1);
            Ok(num_vox)
        }));
    }

    let num_vox: usize = futures::future::join_all(all_jobs).wait()?.iter().sum();

    bar.write().unwrap().finish();
    Ok(num_vox * data_attrs.get_data_type().size_of())
}

fn bench_read_block<T, N5>(
    n: &N5,
    dataset: &str,
    data_attrs: &DatasetAttributes,
    coord: Vec<i64>,
) -> Result<usize>
    where T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
        N5: N5Reader + Sync + Send + Clone + 'static,
        DataType: TypeReflection<T> + DataBlockCreator<T>,
        VecDataBlock<T>: n5::DataBlock<T> {

    let block_in = n.read_block::<T>(
        dataset,
        data_attrs,
        coord)?;
    let num_vox = match block_in {
        Some(block) => {
            block.get_num_elements() as usize
        },
        None => 0,
    };

    Ok(num_vox)
}


fn recompress<N5I, N5O>(
    n5_in: &N5I,
    dataset_in: &str,
    n5_out: &N5O,
    dataset_out: &str,
    compression: CompressionType,
    pool_size: Option<usize>,
) -> Result<usize>
    where
        N5I: N5Reader + Sync + Send + Clone + 'static,
        N5O: N5Writer + Sync + Send + Clone + 'static, {

    let data_attrs_in = n5_in.get_dataset_attributes(dataset_in)?;
    let data_attrs_out = DatasetAttributes::new(
        data_attrs_in.get_dimensions().to_vec(),
        data_attrs_in.get_block_size().to_vec(),
        *data_attrs_in.get_data_type(),
        compression);

    n5_out.create_dataset(dataset_out, &data_attrs_out)?;

    let mut all_jobs: Vec<CpuFuture<usize, std::io::Error>> =
        Vec::new();
    let pool = match pool_size {
        Some(threads) => CpuPool::new(threads),
        None => CpuPool::new_num_cpus(),
    };

    let coord_iter = data_attrs_in.coord_iter();
    let total_coords = coord_iter.len();
    let bar = Arc::new(RwLock::new(ProgressBar::new(total_coords as u64)));
    bar.write().unwrap().set_draw_target(ProgressDrawTarget::stderr());

    for coord in coord_iter {

        let n5_in_c = n5_in.clone();
        let n5_out_c = n5_out.clone();
        let dataset_in_c = dataset_in.to_owned();
        let dataset_out_c = dataset_out.to_owned();
        let data_attrs_in_c = data_attrs_in.clone();
        let data_attrs_out_c = data_attrs_out.clone();
        let bar_c = bar.clone();
        all_jobs.push(pool.spawn_fn(move || {
            // TODO: Have to work around annoying reflection issue.
            let num_vox = match *data_attrs_in_c.get_data_type() {
                DataType::UINT8 => recompress_block::<u8, _, _>(
                    &n5_in_c,
                    &dataset_in_c,
                    &n5_out_c,
                    &dataset_out_c,
                    &data_attrs_in_c,
                    &data_attrs_out_c,
                    coord)?,
                _ => unimplemented!(),
                // DataType::UINT16 => std::mem::size_of::<u16>(),
                // DataType::UINT32 => std::mem::size_of::<u32>(),
                // DataType::UINT64 => std::mem::size_of::<u64>(),
                // DataType::INT8 => std::mem::size_of::<i8>(),
                // DataType::INT16 => std::mem::size_of::<i16>(),
                // DataType::INT32 => std::mem::size_of::<i32>(),
                // DataType::INT64 => std::mem::size_of::<i64>(),
                // DataType::FLOAT32 => std::mem::size_of::<f32>(),
                // DataType::FLOAT64 => std::mem::size_of::<f64>(),
            };
            bar_c.write().unwrap().inc(1);
            Ok(num_vox)
        }));
    }

    let num_vox: usize = futures::future::join_all(all_jobs).wait()?.iter().sum();

    bar.write().unwrap().finish();
    Ok(num_vox * data_attrs_in.get_data_type().size_of())
}

fn recompress_block<T, N5I, N5O>(
    n5_in: &N5I,
    dataset_in: &str,
    n5_out: &N5O,
    dataset_out: &str,
    data_attrs_in: &DatasetAttributes,
    data_attrs_out: &DatasetAttributes,
    coord: Vec<i64>,
) -> Result<usize>
    where T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
        N5I: N5Reader + Sync + Send + Clone + 'static,
        N5O: N5Writer + Sync + Send + Clone + 'static,
        DataType: TypeReflection<T> + DataBlockCreator<T>,
        VecDataBlock<T>: DataBlock<T> {

    let block_in = n5_in.read_block::<T>(
        dataset_in,
        data_attrs_in,
        coord)?;
    let num_vox = match block_in {
        Some(block) => {
            n5_out.write_block(dataset_out, data_attrs_out, &block)?;
            block.get_num_elements() as usize
        },
        None => 0,
    };

    Ok(num_vox)
}

fn get_invalid_blocks<N5>(
    n: &N5,
    dataset: &str,
    pool_size: Option<usize>,
) -> Result<Vec<Vec<i64>>>
    where N5: N5Reader + Sync + Send + Clone + 'static {


    // todo: copied from bench_read; refactor
    let data_attrs = n.get_dataset_attributes(dataset)?;

    let mut all_jobs: Vec<CpuFuture<Option<Vec<i64>>, std::io::Error>> =
        Vec::new();
    let pool = match pool_size {
        Some(threads) => CpuPool::new(threads),
        None => CpuPool::new_num_cpus(),
    };

    let coord_ceil = data_attrs.get_dimensions().iter()
        .zip(data_attrs.get_block_size().iter())
        .map(|(&d, &s)| (d + i64::from(s) - 1) / i64::from(s))
        .collect::<Vec<_>>();
    let total_coords: i64 = coord_ceil.iter().product();
    let coord_iter = coord_ceil.into_iter()
        .map(|c| 0..c)
        .multi_cartesian_product();

    let bar = Arc::new(RwLock::new(ProgressBar::new(total_coords as u64)));

    for coord in coord_iter {
        let n_c = n.clone();
        let dataset_c = dataset.to_owned();
        let data_attrs_c = data_attrs.clone();
        let bar_c = bar.clone();
        all_jobs.push(pool.spawn_fn(move || {
            // TODO: Have to work around annoying reflection issue.
            let results = match *data_attrs_c.get_data_type() {
                DataType::UINT8 => validate_block::<u8, _>(
                    &n_c,
                    &dataset_c,
                    &data_attrs_c,
                    coord)?,
                _ => unimplemented!(),
                // DataType::UINT16 => std::mem::size_of::<u16>(),
                // DataType::UINT32 => std::mem::size_of::<u32>(),
                // DataType::UINT64 => std::mem::size_of::<u64>(),
                // DataType::INT8 => std::mem::size_of::<i8>(),
                // DataType::INT16 => std::mem::size_of::<i16>(),
                // DataType::INT32 => std::mem::size_of::<i32>(),
                // DataType::INT64 => std::mem::size_of::<i64>(),
                // DataType::FLOAT32 => std::mem::size_of::<f32>(),
                // DataType::FLOAT64 => std::mem::size_of::<f64>(),
            };
            bar_c.write().unwrap().inc(1);
            Ok(results)
        }));
    }

    let mut block_idxs : Vec<Vec<i64>> = Vec::new();

    for result in futures::future::join_all(all_jobs).wait()?.iter() {
        match result {
            Some(v) => block_idxs.push(v.to_vec()),
            None => {},
        }
    }

    Ok(block_idxs)
}

fn validate_block<T, N5>(
    n5: &N5,
    dataset: &str,
    data_attrs: &DatasetAttributes,
    coord: Vec<i64>,
) -> Result<Option<Vec<i64>>>
    where T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send,
          N5: N5Reader + Sync + Send + Clone + 'static,
          DataType: TypeReflection<T> + DataBlockCreator<T>,
          VecDataBlock<T>: n5::ReadableDataBlock + n5::WriteableDataBlock {

    let block_opt = n5.read_block::<T>(
        dataset,
        data_attrs,
        coord.to_vec());

    let expected_numel : i32 = data_attrs.get_block_size().to_vec().iter().product();

    match block_opt {
        Ok(o) => match o {
            Some(b) =>
                if b.get_num_elements() == expected_numel {
                    Ok(None)
                } else {
                    Ok(Some(coord))
                },
            None => Ok(None),
        },
        // todo: different handling for different types of error
        Err(_e) => Ok(Some(coord))
    }
}
