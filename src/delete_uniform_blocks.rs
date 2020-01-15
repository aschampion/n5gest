use super::*;

use std::any::Any;

use n5::{data_type_match, data_type_rstype_replace};


#[derive(StructOpt, Debug)]
pub struct DeleteUniformBlocksOptions {
    /// N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// Uniform blocks of this value will be deleted
    #[structopt(name = "UNIFORM_VALUE")]
    uniform_value: String,
    /// Dry run, do not actually delete blocks
    #[structopt(long="dry-run")]
    dry_run: bool,
    /// Print coordinates of deleted blocks
    #[structopt(long="print-coords")]
    print_coords: bool,
}

pub struct DeleteUniformBlocksCommand;

impl CommandType for DeleteUniformBlocksCommand {
    type Options = DeleteUniformBlocksOptions;

    fn run(opt: &Options, dub_opt: &Self::Options) -> Result<()> {
        let n = N5Filesystem::open(&dub_opt.n5_path)?;
        let data_attrs = n.get_dataset_attributes(&dub_opt.dataset)?;
        let data_type = data_attrs.get_data_type();
        let uniform_value: Box<dyn Any + Sync + Send + 'static> = data_type_match! {
            data_type,
            {
                Box::new(dub_opt.uniform_value
                    .parse::<RsType>()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?)
            }
        };

        let started = Instant::now();
        let deleted_coords = DeleteUniformBlocks::run(
            &n,
            &dub_opt.dataset,
            opt.threads,
            DeleteUniformBlocksArguments {
                n5_out: n.clone(),
                uniform_value,
                dry_run: dub_opt.dry_run,
            })?;

        if dub_opt.print_coords {
            for coord in &deleted_coords {
                println!("{:?}", coord);
            }
        }

        println!("Deleted {} blocks in {}",
            deleted_coords.len(),
            HumanDuration(started.elapsed()));

        Ok(())
    }
}


struct DeleteUniformBlocks<N5O> {
    _phantom: std::marker::PhantomData<N5O>,
}


struct DeleteUniformBlocksArguments<N5O>
where N5O: N5Writer + Sync + Send + 'static {
    n5_out: N5O,
    uniform_value: Box<dyn Any + Sync + Send + 'static>,
    dry_run: bool,
}


impl<N5O, T> BlockTypeMap<T> for DeleteUniformBlocks<N5O>
        where
            N5O: N5Writer + Sync + Send + Clone + 'static,
            T: DataTypeBounds,
{

    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        _n: &N5,
        dataset: &str,
        _data_attrs: &DatasetAttributes,
        coord: GridCoord,
        block_opt: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static {

        let num_vox = match block_opt? {
            Some(block) => {
                let uniform_value_t: T = arg.uniform_value
                    .downcast_ref::<T>().unwrap().clone();

                if block.get_data().iter().all(|v| *v == uniform_value_t) {
                    if ! arg.dry_run {
                        arg.n5_out.delete_block(dataset, &coord)?;
                    }
                    Some(coord)
                } else {
                    None
                }
            },
            None => None,
        };

        Ok(num_vox)
    }
}

impl<N5O> BlockReaderMapReduce for DeleteUniformBlocks<N5O>
    where
        N5O: N5Writer + Sync + Send + Clone + 'static,
{
    type BlockResult = Option<GridCoord>;
    type BlockArgument = DeleteUniformBlocksArguments<N5O>;
    type ReduceResult = Vec<GridCoord>;
    type Map = Self;

    fn reduce(
        _data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {

        results.into_iter().filter_map(std::convert::identity).collect()
    }
}
