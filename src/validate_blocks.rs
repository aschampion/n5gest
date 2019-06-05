use super::*;


#[derive(StructOpt, Debug)]
pub struct ValidateBlocksOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
}

pub struct ValidateBlocksCommand;

impl CommandType for ValidateBlocksCommand {
    type Options = ValidateBlocksOptions;

    fn run(opt: &Options, vb_opt: &Self::Options) -> Result<()> {
        let n = N5Filesystem::open(&vb_opt.n5_path).unwrap();
        let started = Instant::now();
        let invalid = ValidateBlocks::run(
            &n,
            &vb_opt.dataset,
            opt.threads,
            ()).unwrap();
        if !invalid.errored.is_empty() {
            eprintln!("Found {} errored block(s)", invalid.errored.len());
            for block_idx in invalid.errored.iter() {
                println!("{}", n.get_block_uri(&vb_opt.dataset, block_idx).unwrap());
            }
        }
        if !invalid.wrongly_sized.is_empty() {
            eprintln!("Found {} wrongly sized block(s)", invalid.wrongly_sized.len());
            for block_idx in invalid.wrongly_sized.iter() {
                println!("{}", n.get_block_uri(&vb_opt.dataset, block_idx).unwrap());
            }
        }
        eprintln!("Found {} invalid block(s) in {}",
            invalid.errored.len() + invalid.wrongly_sized.len(),
            HumanDuration(started.elapsed()));

        Ok(())
    }
}


struct InvalidBlocks {
    errored: Vec<GridCoord>,
    wrongly_sized: Vec<GridCoord>,
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
    Error(GridCoord),
    WrongSize(GridCoord),
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
        coord: GridCoord,
        block_opt: Result<Option<VecDataBlock<T>>>,
        _arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static,
            T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Sync + Send,
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
        _arg: &Self::BlockArgument,
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
