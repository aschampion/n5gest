use crate::common::*;

#[derive(StructOpt, Debug)]
pub struct MapOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
    /// Expression for mapping values `x`
    #[structopt(name = "MAP_EXPR")]
    map_expr: String,
    #[structopt(flatten)]
    bounds: GridBoundsOption,
}

pub struct MapCommand;

impl CommandType for MapCommand {
    type Options = MapOptions;

    fn run(opt: &Options, m_opt: &Self::Options) -> anyhow::Result<()> {
        let map_expr: meval::Expr = m_opt.map_expr.parse().unwrap();
        let n5_in = N5Filesystem::open(&m_opt.input_n5_path)?;
        let n5_out = N5Filesystem::open_or_create(&m_opt.output_n5_path)?;

        let started = Instant::now();
        let num_bytes = Map::run(
            &n5_in,
            &m_opt.input_dataset,
            &*m_opt.bounds.to_factory(),
            opt.threads,
            MapArguments {
                n5_out,
                output_dataset: m_opt.output_dataset.clone(),
                map_expr,
            },
        )?;
        println!(
            "Mapped {} (uncompressed) in {}",
            HumanBytes(num_bytes as u64),
            HumanDuration(started.elapsed()),
        );

        Ok(())
    }
}

struct Map<N5O> {
    _phantom: std::marker::PhantomData<N5O>,
}

#[derive(Clone)]
struct MapArguments<N5O: N5Writer + Sync + Send + Clone + 'static> {
    n5_out: N5O,
    output_dataset: String,
    // Have to store expressions rather than bound closures because meval
    // `Context`s' `GuardedFunc` uses non-`Sync` `Rc`.
    map_expr: meval::Expr,
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static, T> BlockTypeMap<T> for Map<N5O>
where
    T: DataTypeBounds,
    VecDataBlock<T>: n5::ReinitDataBlock<T> + n5::ReadableDataBlock + n5::WriteableDataBlock,
{
    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        _n: &N5,
        _dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: GridCoord,
        block_in: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        let num_vox = match block_in? {
            Some(block) => {
                let map_fn = arg.map_expr.clone().bind("x").unwrap();

                let mapped_data = block
                    .get_data()
                    .iter()
                    .map(|x| x.to_f64().unwrap())
                    .map(map_fn)
                    .map(|x| num_traits::cast::<f64, T>(x).unwrap())
                    .collect();
                let mapped_block = VecDataBlock::new(block.get_size().into(), coord, mapped_data);
                arg.n5_out
                    .write_block(&arg.output_dataset, data_attrs, &mapped_block)?;

                mapped_block.get_num_elements() as usize
            }
            None => 0,
        };

        Ok(num_vox)
    }
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static> BlockReaderMapReduce for Map<N5O> {
    type BlockResult = usize;
    type BlockArgument = MapArguments<N5O>;
    type ReduceResult = usize;
    type Map = Self;

    fn setup<N5>(
        _n: &N5,
        _dataset: &str,
        data_attrs: &DatasetAttributes,
        arg: &mut Self::BlockArgument,
    ) -> Result<()>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        arg.n5_out.create_dataset(&arg.output_dataset, data_attrs)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {
        let num_vox: usize = results.iter().sum();

        num_vox * data_attrs.get_data_type().size_of()
    }
}
