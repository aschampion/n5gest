use crate::common::*;

use std::convert::TryFrom;

use n5::ndarray::{
    BoundingBox,
    N5NdarrayReader,
};
use n5::{
    smallvec::smallvec,
    ReadableDataBlock,
    ReinitDataBlock,
};

#[derive(StructOpt, Debug)]
pub struct SliceImgOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// Output image filename
    #[structopt(name = "FILENAME")]
    filename: String,
    /// Origin voxel coordinates, e.g., `100,200,300`
    #[structopt(long = "origin", use_delimiter = true, required = true)]
    coordinates: Vec<u64>,
    /// Dimensions to slice along, e.g., `0,1`
    #[structopt(
        long = "dims",
        number_of_values = 2,
        use_delimiter = true,
        required = true
    )]
    dims: Vec<u64>,
    /// Size in slicing dimensions, e.g., `512,512`
    #[structopt(
        long = "size",
        number_of_values = 2,
        use_delimiter = true,
        required = true
    )]
    size: Vec<u64>,
}

pub struct SliceImgCommand;

impl CommandType for SliceImgCommand {
    type Options = SliceImgOptions;

    fn run(_opt: &Options, com_opt: &Self::Options) -> anyhow::Result<()> {
        let n = N5Filesystem::open(&com_opt.n5_path)?;
        let started = Instant::now();

        let data_attrs = n
            .get_dataset_attributes(&com_opt.dataset)
            .with_context(|| {
                format!(
                    "Failed to read dataset attributes ({}): {}",
                    &com_opt.n5_path, &com_opt.dataset
                )
            })?;
        anyhow::ensure!(
            data_attrs.get_ndim() == com_opt.coordinates.len(),
            "Expected {} origin dimensions but got {}",
            data_attrs.get_ndim(),
            com_opt.coordinates.len()
        );

        read_and_encode_dispatch(&n, &data_attrs, com_opt)?;

        let num_bytes =
            com_opt.size.iter().product::<u64>() as usize * data_attrs.get_data_type().size_of();
        let elapsed = started.elapsed();
        println!(
            "Wrote {} (uncompressed) in {}",
            HumanBytes(num_bytes as u64),
            HumanDuration(elapsed)
        );
        let throughput = 1e9 * (num_bytes as f64)
            / (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
        println!("({} / s)", HumanBytes(throughput as u64));

        Ok(())
    }
}

fn read_and_encode_dispatch<N: N5Reader>(
    n: &N,
    data_attrs: &DatasetAttributes,
    opt: &SliceImgOptions,
) -> anyhow::Result<()> {
    match *data_attrs.get_data_type() {
        DataType::UINT8 => read_and_encode::<u8, _>(n, data_attrs, opt),
        DataType::UINT16 => read_and_encode::<u16, _>(n, data_attrs, opt),
        _ => unimplemented!(),
    }
}

fn read_and_encode<T, N: N5Reader>(
    n: &N,
    data_attrs: &DatasetAttributes,
    opt: &SliceImgOptions,
) -> anyhow::Result<()>
where
    n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    T: ReflectedType + num_traits::identities::Zero + image::Primitive,
    [T]: image::EncodableLayout,
{
    let image = read_image::<T, N>(n, data_attrs, opt)?;
    encode_image(image, opt)
}

fn read_image<T, N: N5Reader>(
    n: &N,
    data_attrs: &DatasetAttributes,
    opt: &SliceImgOptions,
) -> Result<ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>>
where
    n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    T: ReflectedType + num_traits::identities::Zero,
{
    // Express the tile as an N-dim bounding box.
    let mut size = smallvec![1u64; data_attrs.get_dimensions().len()];
    size[opt.dims[0] as usize] = opt.size[0];
    size[opt.dims[1] as usize] = opt.size[1];
    let bbox = BoundingBox::new(opt.coordinates.iter().cloned().collect(), size);

    // Read the N-dim slab of blocks containing the tile from N5.
    n.read_ndarray::<T>(&opt.dataset, data_attrs, &bbox)
}

fn encode_image<T>(
    image: ndarray::Array<T, ndarray::Dim<ndarray::IxDynImpl>>,
    opt: &SliceImgOptions,
) -> anyhow::Result<()>
where
    n5::VecDataBlock<T>: n5::DataBlock<T> + ReinitDataBlock<T> + ReadableDataBlock,
    T: ReflectedType + num_traits::identities::Zero + image::Primitive,
    [T]: image::EncodableLayout,
{
    use image::{
        ImageBuffer,
        Luma,
    };
    let data = if opt.dims[0] > opt.dims[1] {
        // Note, this works correctly because the slab is f-order.
        image.into_iter().cloned().collect()
    } else {
        image.into_raw_vec()
    };

    let width = u32::try_from(opt.size[0])?;
    let height = u32::try_from(opt.size[1])?;
    ImageBuffer::<Luma<T>, _>::from_vec(width, height, data)
        .unwrap()
        .save(&opt.filename)?;
    Ok(())
}
