use crate::common::*;

use n5::{
    data_type_match,
    data_type_rstype_replace,
};

use crate::iterator::CoordIteratorFactory;

#[derive(StructOpt, Debug)]
pub struct CastOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// New N5 data type
    #[structopt(name = "DATATYPE")]
    data_type_name: String,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
    #[structopt(flatten)]
    bounds: GridBoundsOption,
}

pub struct CastCommand;

impl CommandType for CastCommand {
    type Options = CastOptions;

    fn run(opt: &Options, cast_opt: &Self::Options) -> anyhow::Result<()> {
        let n5_in = N5Filesystem::open(&cast_opt.input_n5_path)?;
        let n5_out = N5Filesystem::open_or_create(&cast_opt.output_n5_path)?;
        let data_type: DataType = serde_plain::from_str(&cast_opt.data_type_name).unwrap();

        let started = Instant::now();
        let num_bytes = dispatch_cast(
            data_type,
            &n5_in,
            &cast_opt.input_dataset,
            &*cast_opt.bounds.to_factory()?,
            opt.threads,
            CastArguments {
                n5_out,
                dataset_out: cast_opt.output_dataset.to_owned(),
                data_attrs_out: None, // TODO: this is a hack.
            },
        )?;
        let elapsed = started.elapsed();
        println!(
            "Converted {} (uncompressed) in {}",
            HumanBytes(num_bytes as u64),
            HumanDuration(elapsed)
        );
        let throughput = 1e9 * (num_bytes as f64)
            / (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
        println!("({} / s)", HumanBytes(throughput as u64));

        Ok(())
    }
}

fn dispatch_cast<N5, CI>(
    data_type: DataType,
    n5_in: &N5,
    input_dataset: &str,
    coord_iter_factory: &CI,
    pool_size: Option<usize>,
    arg: CastArguments<N5>,
) -> anyhow::Result<usize>
where
    N5: N5Writer + Sync + Send + Clone + 'static,
    CI: CoordIteratorFactory + ?Sized,
{
    data_type_match! {
        data_type,
        {
            Cast::<_, RsType>::run(
                n5_in,
                input_dataset,
                coord_iter_factory,
                pool_size,
                arg)
        }
    }
}

struct Cast<N5O, TO> {
    _phantom: std::marker::PhantomData<N5O>,
    _phantom_data_type: std::marker::PhantomData<TO>,
}

#[derive(Clone)]
struct CastArguments<N5O>
where
    N5O: N5Writer + Sync + Send + Clone + 'static,
{
    n5_out: N5O,
    data_attrs_out: Option<DatasetAttributes>,
    dataset_out: String,
}

impl<N5O, TO, T> BlockTypeMap<T> for Cast<N5O, TO>
where
    N5O: N5Writer + Sync + Send + Clone + 'static,
    T: DataTypeBounds,
    TO: DataTypeBounds,
    VecDataBlock<TO>: n5::WriteableDataBlock,
{
    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _coord: GridCoord,
        block_opt: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        let num_vox = match block_opt? {
            Some(block) => {
                let cast_data: Vec<TO> = block
                    .get_data()
                    .iter()
                    .cloned()
                    .map(TO::from)
                    .collect::<Option<Vec<TO>>>()
                    .expect("Value cannot be converted");
                let cast_block = VecDataBlock::<TO>::new(
                    block.get_size().into(),
                    block.get_grid_position().into(),
                    cast_data,
                );
                arg.n5_out.write_block(
                    &arg.dataset_out,
                    arg.data_attrs_out.as_ref().unwrap(),
                    &cast_block,
                )?;
                block.get_num_elements() as usize
            }
            None => 0,
        };

        Ok(num_vox)
    }
}

impl<N5O, TO> BlockReaderMapReduce for Cast<N5O, TO>
where
    N5O: N5Writer + Sync + Send + Clone + 'static,
    TO: DataTypeBounds,
    VecDataBlock<TO>: n5::WriteableDataBlock,
{
    type BlockResult = usize;
    type BlockArgument = CastArguments<N5O>;
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
        arg.data_attrs_out = Some(DatasetAttributes::new(
            data_attrs.get_dimensions().into(),
            data_attrs.get_block_size().into(),
            TO::VARIANT,
            data_attrs.get_compression().clone(),
        ));
        arg.n5_out
            .create_dataset(&arg.dataset_out, arg.data_attrs_out.as_ref().unwrap())
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
