use std::io::Result;
use std::sync::{
    Arc,
    RwLock,
};

use anyhow::Context;
use futures::executor::ThreadPool;
use indicatif::ProgressBar;
use n5::prelude::*;
use n5::{
    data_type_match,
    data_type_rstype_replace,
};

use crate::iterator::CoordIteratorFactory;
use crate::pool::pool_execute;

/// Convience trait combined all the required traits on data types until trait
/// aliases are stabilized.
pub(crate) trait DataTypeBounds:
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
pub(crate) trait BlockTypeMap<T>
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

pub(crate) trait BlockMap<Res: Send + 'static, Arg: Send + Sync + 'static>:
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

pub(crate) trait BlockReaderMapReduce {
    type BlockResult: Send + 'static;
    type BlockArgument: Send + Sync + 'static;
    type ReduceResult;
    // TODO: When associated type default stabilize, `Map` should default to
    // `Self`. Tracking issue: https://github.com/rust-lang/rust/issues/29661
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

    fn run<N5, CI>(
        n: &N5,
        dataset: &str,
        coord_iter_factory: &CI,
        pool_size: Option<usize>,
        mut arg: Self::BlockArgument,
    ) -> anyhow::Result<Self::ReduceResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
        CI: CoordIteratorFactory + ?Sized,
    {
        let data_attrs = n
            .get_dataset_attributes(dataset)
            .context("Failed to read dataset attributes")?;

        Self::setup(n, dataset, &data_attrs, &mut arg)?;

        let coord_iter = coord_iter_factory.coord_iter(&data_attrs);
        let pbar = RwLock::new(crate::default_progress_bar(coord_iter.len() as u64));

        let pool = {
            let mut builder = ThreadPool::builder();
            if let Some(threads) = pool_size {
                builder.pool_size(threads);
            }
            builder.create()?
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

        let block_results = pool_execute::<anyhow::Error, _, _, _>(
            &pool,
            coord_iter.map(|coord| {
                let local = scoped.clone();

                async move {
                    let coord: GridCoord = coord.into();
                    let block_result = Self::map_type_dispatch(
                        &local.n,
                        &local.dataset,
                        &local.data_attrs,
                        coord.clone(),
                        &local.arg,
                    )
                    .with_context(|| format!("Command failed for block at {:?}", coord))?;
                    local.pbar.write().unwrap().inc(1);
                    Ok(block_result)
                }
            }),
        )?;

        scoped.pbar.write().unwrap().finish();
        Ok(Self::reduce(&scoped.data_attrs, block_results, &scoped.arg))
    }
}
