use crate::common::*;

#[derive(StructOpt, Debug)]
pub struct RecompressOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// New N5 compression (JSON), e.g., '{"type": "gzip"}'
    #[structopt(name = "COMPRESSION")]
    compression: String,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
    #[structopt(flatten)]
    bounds: GridBoundsOption,
}

pub struct RecompressCommand;

impl CommandType for RecompressCommand {
    type Options = RecompressOptions;

    fn run(opt: &Options, com_opt: &Self::Options) -> Result<()> {
        let n5_in = N5Filesystem::open(&com_opt.input_n5_path)?;
        let n5_out = N5Filesystem::open_or_create(&com_opt.output_n5_path)?;
        let compression: CompressionType = serde_json::from_str(&com_opt.compression)?;
        println!("Recompressing with {}", compression);

        let started = Instant::now();
        let num_bytes = Recompress::run(
            &n5_in,
            &com_opt.input_dataset,
            &*com_opt.bounds.to_factory(),
            opt.threads,
            RecompressArguments {
                n5_out,
                dataset_out: com_opt.output_dataset.to_owned(),
                data_attrs_out: None, // TODO: this is a hack.
                compression,
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

impl<N5O: N5Writer + Sync + Send + Clone + 'static, T> BlockTypeMap<T> for Recompress<N5O>
where
    T: DataTypeBounds,
    VecDataBlock<T>: n5::WriteableDataBlock,
{
    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        _n: &N5,
        _dataset: &str,
        _data_attrs: &DatasetAttributes,
        _coord: GridCoord,
        block_in: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        let num_vox = match block_in? {
            Some(block) => {
                arg.n5_out.write_block(
                    &arg.dataset_out,
                    &arg.data_attrs_out.as_ref().unwrap(),
                    block,
                )?;
                block.get_num_elements() as usize
            }
            None => 0,
        };

        Ok(num_vox)
    }
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static> BlockReaderMapReduce for Recompress<N5O> {
    type BlockResult = usize;
    type BlockArgument = RecompressArguments<N5O>;
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
            *data_attrs.get_data_type(),
            arg.compression.clone(),
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
