use crate::common::*;

#[derive(StructOpt, Debug)]
pub struct BenchReadOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    #[structopt(flatten)]
    bounds: GridBoundsOption,
}

pub struct BenchReadCommand;

impl CommandType for BenchReadCommand {
    type Options = BenchReadOptions;

    fn run(opt: &Options, br_opt: &Self::Options) -> anyhow::Result<()> {
        let n = N5Filesystem::open(&br_opt.n5_path)?;
        let started = Instant::now();
        let num_bytes = BenchRead::run(
            &n,
            &br_opt.dataset,
            &*br_opt.bounds.to_factory()?,
            opt.threads,
            (),
        )?;
        let elapsed = started.elapsed();
        println!(
            "Read {} (uncompressed) in {}",
            HumanBytes(num_bytes as u64),
            HumanDuration(elapsed)
        );
        let throughput = 1e9 * (num_bytes as f64)
            / (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
        println!("({} / s)", HumanBytes(throughput as u64));

        Ok(())
    }
}

struct BenchRead;

impl<T> BlockTypeMap<T> for BenchRead
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
        block_in: Result<Option<&VecDataBlock<T>>>,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        let num_vox = match block_in? {
            Some(block) => block.get_num_elements() as usize,
            None => 0,
        };

        Ok(num_vox)
    }
}

impl BlockReaderMapReduce for BenchRead {
    type BlockResult = usize;
    type BlockArgument = ();
    type ReduceResult = usize;
    type Map = Self;

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {
        let num_vox: usize = results.iter().sum();

        num_vox * data_attrs.get_data_type().size_of()
    }
}
