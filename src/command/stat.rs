use crate::common::*;

use std::cmp::{
    max,
    min,
};
use std::convert::TryInto;
use std::time::{
    Duration,
    SystemTime,
};

use chrono::prelude::*;
use prettytable::Table;

#[derive(StructOpt, Debug)]
pub struct StatOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    #[structopt(flatten)]
    bounds: GridBoundsOption,
}

pub struct StatCommand;

impl CommandType for StatCommand {
    type Options = StatOptions;

    fn run(opt: &Options, st_opt: &Self::Options) -> anyhow::Result<()> {
        let n = N5Filesystem::open(&st_opt.n5_path)?;
        let result = Stat::run(
            &n,
            &st_opt.dataset,
            &*st_opt.bounds.to_factory(),
            opt.threads,
            (),
        )?;
        let data_attrs = n.get_dataset_attributes(&st_opt.dataset)?;

        if let Some(agg) = result {
            let prep_date = |date: Option<SystemTime>| {
                date.map(DateTime::<Local>::from)
                    .map(|date| ToString::to_string(&date))
                    .unwrap_or_else(String::new)
            };
            let prep_size = |size: Option<u64>| {
                size.map(HumanBytes)
                    .map(|size| ToString::to_string(&size))
                    .unwrap_or_else(String::new)
            };
            let size_ratio = |size: Option<u64>, data_attrs: &DatasetAttributes| {
                size.unwrap_or(0) as f64
                    / (data_attrs.get_block_num_elements() * data_attrs.get_data_type().size_of())
                        as f64
            };

            let mut md_table = Table::new();
            md_table.set_format(*prettytable::format::consts::FORMAT_CLEAN);
            md_table.set_titles(row![
                "",
                r -> "Size",
                r -> "Ratio",
                r -> "GridCoord",
                "Created",
                "Accessed",
                "Modified",
            ]);
            md_table.add_row(row![
                "min",
                r -> prep_size(agg.min_metadata.size),
                r -> format!("{:.3}", size_ratio(agg.min_metadata.size, &data_attrs)),
                r -> format!("{:?}", agg.min_block_coord),
                prep_date(agg.min_metadata.created),
                prep_date(agg.min_metadata.accessed),
                prep_date(agg.min_metadata.modified),
            ]);
            md_table.add_row(row![
                "max",
                r -> prep_size(agg.max_metadata.size),
                r -> format!("{:.3}", size_ratio(agg.max_metadata.size, &data_attrs)),
                r -> format!("{:?}", agg.max_block_coord),
                prep_date(agg.max_metadata.created),
                prep_date(agg.max_metadata.accessed),
                prep_date(agg.max_metadata.modified),
            ]);
            let (average, avg_coord) = agg.sum_metadata.average(agg.occupied);
            md_table.add_row(row![
                "average",
                r -> prep_size(average.size),
                r -> format!("{:.3}", size_ratio(average.size, &data_attrs)),
                r -> format!("{:?}", avg_coord),
                prep_date(average.created),
                prep_date(average.accessed),
                prep_date(average.modified),
            ]);
            let ratio_of_max_fully_occupied_size = agg.sum_metadata.size.unwrap_or(0) as f64
                / (data_attrs.get_num_elements() * data_attrs.get_data_type().size_of()) as f64;
            md_table.add_row(row![
                b -> "total",
                rb -> prep_size(agg.sum_metadata.size),
                r -> format!("{:.3}", ratio_of_max_fully_occupied_size),
                rb -> format!("{}/{}", agg.occupied, agg.total),
                rb -> format!("{:.2}%", agg.occupied_percentage()),
                "",
                "",
            ]);

            md_table.printstd();
        } else {
            println!("No occupied blocks found");
        }

        Ok(())
    }
}

#[derive(Debug)]
struct AggregateStats {
    max_metadata: DataBlockMetadata,
    min_metadata: DataBlockMetadata,
    sum_metadata: SumMetadata,
    min_block_coord: GridCoord,
    max_block_coord: GridCoord,
    occupied: u64,
    total: u64,
}

impl AggregateStats {
    fn occupied_percentage(&self) -> f64 {
        (self.occupied as f64) / (self.total as f64) * 100.0
    }
}

#[derive(Debug, Default)]
struct SumMetadata {
    created: Option<Duration>,
    accessed: Option<Duration>,
    modified: Option<Duration>,
    block_coord: GridCoord,
    size: Option<u64>,
}

impl SumMetadata {
    fn average(&self, total: u64) -> (DataBlockMetadata, GridCoord) {
        let total32: u32 = total.try_into().unwrap();
        (
            DataBlockMetadata {
                created: self.created.map(|d| SystemTime::UNIX_EPOCH + d / total32),
                accessed: self.accessed.map(|d| SystemTime::UNIX_EPOCH + d / total32),
                modified: self.modified.map(|d| SystemTime::UNIX_EPOCH + d / total32),
                size: self.size.map(|s| s / total),
            },
            self.block_coord
                .iter()
                .cloned()
                .map(|c| c / total)
                .collect(),
        )
    }
}

fn option_fold<U, F>(a: Option<U>, b: Option<U>, f: F) -> Option<U>
where
    F: FnOnce(U, U) -> U,
{
    match (a, b) {
        (Some(a), Some(b)) => Some(f(a, b)),
        (a @ Some(_), None) => a,
        (None, b @ Some(_)) => b,
        (None, None) => None,
    }
}

impl std::ops::Add<(GridCoord, &DataBlockMetadata)> for SumMetadata {
    type Output = SumMetadata;

    fn add(self, tuple: (GridCoord, &DataBlockMetadata)) -> SumMetadata {
        let (coord, other) = tuple;
        let duration_add = |sd: Option<Duration>, od: Option<SystemTime>| {
            let od = od.map(|d| d.duration_since(SystemTime::UNIX_EPOCH).unwrap());
            option_fold(sd, od, |a, b| a + b)
        };

        let block_coord = if self.block_coord.len() != coord.len() {
            // Has not yet be initialized.
            coord
        } else {
            self.block_coord
                .iter()
                .zip(coord.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };

        Self {
            created: duration_add(self.created, other.created),
            accessed: duration_add(self.accessed, other.accessed),
            modified: duration_add(self.modified, other.modified),
            block_coord,
            size: option_fold(self.size, other.size, |a, b| a + b),
        }
    }
}

struct Stat;

impl<T> BlockTypeMap<T> for Stat
where
    T: DataTypeBounds,
{
    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _coord: GridCoord,
        _block_in: Result<Option<&VecDataBlock<T>>>,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        unimplemented!()
    }
}

impl BlockReaderMapReduce for Stat {
    type BlockResult = Option<(GridCoord, DataBlockMetadata)>;
    type BlockArgument = ();
    type ReduceResult = Option<AggregateStats>;
    type Map = Self;

    fn reduce(
        _data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {
        let total = results.len().try_into().unwrap();
        let mut results = results.into_iter().filter_map(|b| b);

        if let Some((coord, first)) = results.next() {
            let stats = AggregateStats {
                max_metadata: first.clone(),
                min_metadata: first.clone(),
                sum_metadata: SumMetadata::default() + (coord.clone(), &first),
                min_block_coord: coord.clone(),
                max_block_coord: coord,
                occupied: 1,
                total,
            };

            Some(results.fold(stats, |stats, (coord, block)| {
                AggregateStats {
                    max_metadata: DataBlockMetadata {
                        created: option_fold(stats.max_metadata.created, block.created, max),
                        accessed: option_fold(stats.max_metadata.accessed, block.accessed, max),
                        modified: option_fold(stats.max_metadata.modified, block.modified, max),
                        size: option_fold(stats.max_metadata.size, block.size, max),
                    },
                    min_metadata: DataBlockMetadata {
                        created: option_fold(stats.min_metadata.created, block.created, min),
                        accessed: option_fold(stats.min_metadata.accessed, block.accessed, min),
                        modified: option_fold(stats.min_metadata.modified, block.modified, min),
                        size: option_fold(stats.min_metadata.size, block.size, min),
                    },
                    min_block_coord: stats
                        .min_block_coord
                        .iter()
                        .zip(coord.iter())
                        .map(|(&a, &b)| min(a, b))
                        .collect(),
                    max_block_coord: stats
                        .max_block_coord
                        .iter()
                        .zip(coord.iter())
                        .map(|(&a, &b)| max(a, b))
                        .collect(),
                    sum_metadata: stats.sum_metadata + (coord, &block),
                    occupied: stats.occupied + 1,
                    total,
                }
            }))
        } else {
            None
        }
    }

    fn map_type_dispatch<N5>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: GridCoord,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        n.block_metadata(dataset, data_attrs, &coord)
            .map(|maybe_meta| maybe_meta.map(|meta| (coord, meta)))
    }
}
