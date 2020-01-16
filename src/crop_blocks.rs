use super::*;

use n5::ndarray::prelude::*;

#[derive(StructOpt, Debug)]
pub struct CropBlocksOptions {
    /// Input N5 root path
    #[structopt(name = "INPUT_N5")]
    input_n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "INPUT_DATASET")]
    input_dataset: String,
    /// Axis to check
    #[structopt(name = "AXIS")]
    axis: i32,
    /// Output N5 root path
    #[structopt(name = "OUTPUT_N5")]
    output_n5_path: String,
    /// Output N5 dataset
    #[structopt(name = "OUPUT_DATASET")]
    output_dataset: String,
}

pub struct CropBlocksCommand;

impl CommandType for CropBlocksCommand {
    type Options = CropBlocksOptions;

    fn run(opt: &Options, crop_opt: &Self::Options) -> Result<()> {
        let n5_in = N5Filesystem::open(&crop_opt.input_n5_path)?;
        let n5_out = N5Filesystem::open_or_create(&crop_opt.output_n5_path)?;
        println!("Cropping along {}", crop_opt.axis);

        let started = Instant::now();
        let (num_blocks, num_bytes) = CropBlocks::run(
            &n5_in,
            &crop_opt.input_dataset,
            opt.threads,
            CropBlocksArguments {
                n5_out,
                dataset_out: crop_opt.output_dataset.to_owned(),
                axis: crop_opt.axis,
            },
        )?;
        println!(
            "Converted {} blocks with {} (uncompressed) in {}",
            num_blocks,
            HumanBytes(num_bytes as u64),
            HumanDuration(started.elapsed())
        );

        Ok(())
    }
}

struct CropBlocks<N5O> {
    _phantom: std::marker::PhantomData<N5O>,
}

#[derive(Clone)]
struct CropBlocksArguments<N5O: N5Writer + Sync + Send + Clone + 'static> {
    n5_out: N5O,
    dataset_out: String,
    axis: i32,
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static, T> BlockTypeMap<T> for CropBlocks<N5O>
where
    T: DataTypeBounds,
    VecDataBlock<T>: n5::ReinitDataBlock<T> + n5::ReadableDataBlock + n5::WriteableDataBlock,
{
    type BlockArgument = <Self as BlockReaderMapReduce>::BlockArgument;
    type BlockResult = <Self as BlockReaderMapReduce>::BlockResult;

    fn map<N5>(
        n: &N5,
        dataset: &str,
        data_attrs: &DatasetAttributes,
        coord: GridCoord,
        block_opt: Result<Option<&VecDataBlock<T>>>,
        arg: &Self::BlockArgument,
    ) -> Result<Self::BlockResult>
    where
        N5: N5Reader + Sync + Send + Clone + 'static,
    {
        let num_vox = match block_opt? {
            Some(_) => {
                // TODO: only reading block because it is the only way currently
                // to test block existence. To be more efficient could either
                // use another means, or crop from this read block directly rather
                // than re-reading using the ndarray convenience method.

                let (offset, size): (GridCoord, GridCoord) = data_attrs
                    .get_dimensions()
                    .iter()
                    .zip(data_attrs.get_block_size().iter().cloned().map(u64::from))
                    .zip(coord.iter())
                    .map(|((&d, s), &c)| {
                        let offset = c * s;
                        let size = std::cmp::min((c + 1) * s, d) - offset;
                        (offset, size)
                    })
                    .unzip();

                let bbox = BoundingBox::new(offset, size.clone());

                let cropped = n.read_ndarray::<T>(dataset, data_attrs, &bbox)?;
                assert!(
                    !cropped.is_standard_layout(),
                    "Array should still be in f-order"
                );
                let cropped_block = VecDataBlock::<T>::new(
                    size.into_iter().map(|n| n as u32).collect(),
                    coord,
                    cropped.as_slice_memory_order().unwrap().to_owned(),
                );
                arg.n5_out
                    .write_block(&arg.dataset_out, data_attrs, &cropped_block)?;
                Some(cropped_block.get_num_elements() as usize)
            }
            None => None,
        };

        Ok(num_vox)
    }
}

impl<N5O: N5Writer + Sync + Send + Clone + 'static> BlockReaderMapReduce for CropBlocks<N5O> {
    type BlockResult = Option<usize>;
    type BlockArgument = CropBlocksArguments<N5O>;
    type ReduceResult = (usize, usize);
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
        arg.n5_out.create_dataset(&arg.dataset_out, data_attrs)
    }

    fn coord_iter(
        data_attrs: &DatasetAttributes,
        arg: &Self::BlockArgument,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize) {
        let axis = arg.axis;
        let mut coord_ceil = data_attrs
            .get_dimensions()
            .iter()
            .zip(data_attrs.get_block_size().iter())
            .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
            .collect::<Vec<_>>();
        let axis_ceil = coord_ceil.remove(axis as usize);
        let total_coords = coord_ceil.iter().product::<u64>() as usize;
        let coord_iter = coord_ceil
            .into_iter()
            .map(|c| 0..c)
            .multi_cartesian_product()
            .map(move |mut c| {
                c.insert(axis as usize, axis_ceil - 1);
                c
            });

        (Box::new(coord_iter), total_coords)
    }

    fn reduce(
        data_attrs: &DatasetAttributes,
        results: Vec<Self::BlockResult>,
        _arg: &Self::BlockArgument,
    ) -> Self::ReduceResult {
        let (num_blocks, num_vox): (usize, usize) =
            results.iter().fold((0, 0), |(blocks, total), vox| {
                if let Some(count) = vox {
                    (blocks + 1, total + count)
                } else {
                    (blocks, total)
                }
            });

        (num_blocks, num_vox * data_attrs.get_data_type().size_of())
    }
}
