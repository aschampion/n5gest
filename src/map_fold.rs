use super::*;


#[derive(StructOpt, Debug)]
pub struct MapFoldOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// Initial value for fold accumulation
    #[structopt(name = "INITIAL_VAL")]
    initial_val: f64,
    /// Expression for folding over values `x` with an accumulator `acc`
    #[structopt(name = "FOLD_EXPR")]
    fold_expr: String,
    /// Expression for folding over block results. By default FOLD_EXPR is used.
    #[structopt(name = "BLOCK_FOLD_EXPR")]
    block_fold_expr: Option<String>,
}

pub struct MapFoldCommand;

impl CommandType for MapFoldCommand {
    type Options = MapFoldOptions;

    fn run(opt: &Options, mf_opt: &Self::Options) -> Result<()> {
        let block_fold_expr: meval::Expr = mf_opt.block_fold_expr.clone()
            .unwrap_or(mf_opt.fold_expr.clone()).parse().unwrap();
        let fold_expr: meval::Expr = mf_opt.fold_expr.parse().unwrap();
        // let fold_fn = fold_parsed.bind2("acc", "x").unwrap();
        let n = N5Filesystem::open(&mf_opt.n5_path)?;

        let result = MapFold::run(
            &n,
            &mf_opt.dataset,
            opt.threads,
            MapFoldArgument {
                initial_val: mf_opt.initial_val,
                fold_expr,
                block_fold_expr,
            })?;
        println!("{}", result);

        Ok(())
    }
}


struct MapFold;

struct MapFoldArgument {
    initial_val: f64,
    // Have to store expressions rather than bound closures because meval
    // `Context`s' `GuardedFunc` uses non-`Sync` `Rc`.
    fold_expr: meval::Expr,
    block_fold_expr: meval::Expr,
}

impl<T> BlockTypeMap<T> for MapFold
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
        block_in: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
        where
            N5: N5Reader + Sync + Send + Clone + 'static {

        Ok(block_in?.map(|block| {
            let fold_fn = arg.fold_expr.clone().bind2("acc", "x").unwrap();

            block.get_data().iter()
                .map(|x| x.to_f64().unwrap())
                .fold(arg.initial_val, fold_fn)
        }))
    }
}

impl BlockReaderMapReduce for MapFold {
    type BlockResult = Option<f64>;
    type BlockArgument = MapFoldArgument;
    type ReduceResult = f64;
    type Map = Self;

    fn reduce(
        _data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {

        let fold_fn = arg.block_fold_expr.clone().bind2("acc", "x").unwrap();

        results.into_iter()
            .filter(Option::is_some)
            .map(Option::unwrap)
            .fold(arg.initial_val, fold_fn)
    }
}
