use super::*;

use std::cmp::{max, min};

use chrono::prelude::*;


#[derive(StructOpt, Debug)]
pub struct StatOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
}

pub struct StatCommand;

impl CommandType for StatCommand {
    type Options = StatOptions;

    fn run(opt: &Options, st_opt: &Self::Options) -> Result<()> {
        let n = N5Filesystem::open(&st_opt.n5_path)?;
        let result = Stat::run(
            &n,
            &st_opt.dataset,
            opt.threads,
            ())?;

        if let Some(agg) = result {
            println!("{}/{} blocks exist", agg.occupied, agg.total);

            let mut md_table = Table::new();
            md_table.set_format(*prettytable::format::consts::FORMAT_CLEAN);
            md_table.set_titles(row![
                "",
                "Created",
                "Accessed",
                "Modified",
            ]);
            md_table.add_row(row![
                "min",
                DateTime::<Local>::from(agg.min_metadata.created),
                DateTime::<Local>::from(agg.min_metadata.accessed),
                DateTime::<Local>::from(agg.min_metadata.modified),
            ]);
            md_table.add_row(row![
                "max",
                DateTime::<Local>::from(agg.max_metadata.created),
                DateTime::<Local>::from(agg.max_metadata.accessed),
                DateTime::<Local>::from(agg.max_metadata.modified),
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
    occupied: usize,
    total: usize,
}

struct Stat;

impl<T> BlockTypeMap<T> for Stat
        where
            T: DataTypeBounds,
            VecDataBlock<T>: n5::DataBlock<T> {

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
            N5: N5Reader + Sync + Send + Clone + 'static {

        unimplemented!()
    }
}

impl BlockReaderMapReduce for Stat {
    type BlockResult = Option<DataBlockMetadata>;
    type BlockArgument = ();
    type ReduceResult = Option<AggregateStats>;
    type Map = Self;

    fn reduce(
        _data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {

        let total = results.len();
        let mut results = results.into_iter().filter_map(|b| b);

        if let Some(first) = results.next() {
            let stats = AggregateStats {
                max_metadata: first.clone(),
                min_metadata: first,
                occupied: 1,
                total
            };

            Some(results.fold(stats, |stats, block| {
                AggregateStats {
                    max_metadata: DataBlockMetadata {
                        created: max(stats.max_metadata.created, block.created),
                        accessed: max(stats.max_metadata.accessed, block.accessed),
                        modified: max(stats.max_metadata.modified, block.modified),
                    },
                    min_metadata: DataBlockMetadata {
                        created: min(stats.min_metadata.created, block.created),
                        accessed: min(stats.min_metadata.accessed, block.accessed),
                        modified: min(stats.min_metadata.modified, block.modified),
                    },
                    occupied: stats.occupied + 1,
                    total
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
        where N5: N5Reader + Sync + Send + Clone + 'static {

        n.block_metadata(dataset, data_attrs, &coord)
    }
}
