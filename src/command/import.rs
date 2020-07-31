use crate::common::*;

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::RwLock;

use futures::executor::ThreadPool;
use image::{
    DynamicImage,
    GenericImageView,
};
use n5::smallvec::smallvec;

use crate::default_progress_bar;
use crate::iterator::{
    CoordIteratorFactory,
    GridSlabCoordIter,
};
use crate::pool::pool_execute;

pub mod tiff;

#[derive(StructOpt, Debug)]
pub struct ImportOptions {
    /// Ouput N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Ouput N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// New N5 compression (optionally JSON), e.g., 'gzip' or '{"type": "gzip", "level": 2}'
    #[structopt(name = "COMPRESSION")]
    compression: String,
    /// Block size, e.g., 128,128,13
    #[structopt(name = "BLOCK_SIZE")]
    block_size: BlockSizeParam,
    /// Files to import, ordered by increasing z
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,
    /// Uniform blocks of this value will not be written
    #[structopt(long = "elide_fill_value")]
    elide_fill_value: Option<String>,
}

#[derive(Debug)]
struct BlockSizeParam([usize; 3]);

impl FromStr for BlockSizeParam {
    type Err = std::io::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let re = regex::Regex::new(r"(?P<x>\d+),(?P<y>\d+),(?P<z>\d+)").unwrap();
        let caps = re.captures(s).unwrap();

        Ok(BlockSizeParam([
            caps["x"].parse().unwrap(),
            caps["y"].parse().unwrap(),
            caps["z"].parse().unwrap(),
        ]))
    }
}

pub struct ImportCommand;

impl CommandType for ImportCommand {
    type Options = ImportOptions;

    fn run(opt: &Options, imp_opt: &Self::Options) -> anyhow::Result<()> {
        let n = Arc::new(N5Filesystem::open_or_create(&imp_opt.n5_path)?);

        if n.exists(&imp_opt.dataset)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Dataset {} already exists", imp_opt.dataset),
            )
            .into());
        }
        let compression: CompressionType = from_plain_or_json_str(&imp_opt.compression)
            .with_context(|| {
                format!(
                    "Failed to parse new compression type: {}",
                    &imp_opt.compression
                )
            })?;
        let started = Instant::now();

        let files = imp_opt.files.as_slice();
        let z_dim = files.len();
        let ref_img = image::open(&files[0])?;
        let xy_dims = ref_img.dimensions();
        let dtype = color_to_dtype(ref_img.color());

        let data_attrs = Arc::new(DatasetAttributes::new(
            smallvec![u64::from(xy_dims.0), u64::from(xy_dims.1), z_dim as u64],
            imp_opt.block_size.0.iter().map(|&b| b as u32).collect(),
            dtype,
            compression,
        ));

        let dataset = Arc::new(imp_opt.dataset.clone());
        n.create_dataset(&dataset, &data_attrs)?;
        let slab_size = imp_opt.block_size.0[2];

        let slab_img_buff = Arc::new(RwLock::new(Vec::<Option<DynamicImage>>::with_capacity(
            slab_size,
        )));

        let pool = {
            let mut builder = ThreadPool::builder();
            if let Some(threads) = opt.threads {
                builder.pool_size(threads);
            }
            builder.create()?
        };
        let pbar = RwLock::new(default_progress_bar(files.len() as u64));
        let elide_fill_value = imp_opt.elide_fill_value.clone().map(Arc::new);

        for (slab_coord, slab_files) in files.chunks(slab_size).enumerate() {
            import_slab(
                &n,
                &dataset,
                &pool,
                &slab_img_buff,
                &slab_files,
                &data_attrs,
                slab_coord,
                elide_fill_value.clone(),
            )?;
            pbar.write().unwrap().inc(slab_files.len() as u64);
        }

        pbar.write().unwrap().finish();

        let num_bytes = data_attrs.get_num_elements() * data_attrs.get_data_type().size_of();
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

fn color_to_dtype(color: image::ColorType) -> DataType {
    match color {
        image::ColorType::L8 => DataType::UINT8,
        image::ColorType::L16 => DataType::UINT16,
        _ => unimplemented!(),
    }
}

fn import_slab<N5: N5Writer + Sync + Send + Clone + 'static>(
    n: &Arc<N5>,
    dataset: &Arc<String>,
    pool: &ThreadPool,
    slab_img_buff: &Arc<RwLock<Vec<Option<DynamicImage>>>>,
    slab_files: &[PathBuf],
    data_attrs: &Arc<DatasetAttributes>,
    slab_coord: usize,
    elide_fill_value: Option<Arc<String>>,
) -> anyhow::Result<()> {
    {
        let mut buff_vec = slab_img_buff.write().unwrap();
        buff_vec.clear();
        buff_vec.resize(slab_files.len(), None);
    }

    pool_execute::<anyhow::Error, _, _, _>(
        pool,
        slab_files.iter().enumerate().map(|(i, file)| {
            let owned_file = file.clone();
            let data_attrs = data_attrs.clone();
            let slab_img_buff = slab_img_buff.clone();
            async move {
                let image = image::open(&owned_file)?;
                assert_eq!(color_to_dtype(image.color()), *data_attrs.get_data_type());
                assert_eq!(
                    u64::from(image.dimensions().0),
                    data_attrs.get_dimensions()[0]
                );
                assert_eq!(
                    u64::from(image.dimensions().1),
                    data_attrs.get_dimensions()[1]
                );
                slab_img_buff.write().unwrap()[i] = Some(image);

                Ok(())
            }
        }),
    )?;

    let coord_iter = GridSlabCoordIter {
        axis: 2,
        slab_coord: slab_coord as u64,
    };
    let slab_coord_iter = coord_iter.coord_iter(&*data_attrs);

    pool_execute(
        pool,
        slab_coord_iter.map(|coord| {
            let n = n.clone();
            let dataset = dataset.clone();
            let slab_img_buff = slab_img_buff.clone();
            let data_attrs = data_attrs.clone();
            let elide_fill_value = elide_fill_value.clone();
            async move {
                let slab_read = slab_img_buff.read().unwrap();
                slab_block_dispatch(
                    &*n,
                    &*dataset,
                    coord.into(),
                    &slab_read,
                    &*data_attrs,
                    elide_fill_value,
                )
            }
        }),
    )?;

    Ok(())
}

fn slab_block_dispatch<N5>(
    n: &N5,
    dataset: &str,
    coord: GridCoord,
    slab_img_buff: &[Option<DynamicImage>],
    data_attrs: &DatasetAttributes,
    elide_fill_value: Option<Arc<String>>,
) -> Result<()>
where
    N5: N5Writer + Sync + Send + Clone + 'static,
{
    match *data_attrs.get_data_type() {
        DataType::UINT8 => {
            let elide_fill_value: Option<u8> = elide_fill_value
                .as_ref()
                .map(|v| v.parse())
                .transpose()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            slab_block_writer(
                n,
                dataset,
                coord,
                slab_img_buff
                    .iter()
                    .map(|di| di.as_ref().map(|di| di.as_luma8().unwrap())),
                data_attrs,
                elide_fill_value,
            )
        }
        DataType::UINT16 => {
            let elide_fill_value: Option<u16> = elide_fill_value
                .as_ref()
                .map(|v| v.parse())
                .transpose()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
            slab_block_writer(
                n,
                dataset,
                coord,
                slab_img_buff
                    .iter()
                    .map(|di| di.as_ref().map(|di| di.as_luma16().unwrap())),
                data_attrs,
                elide_fill_value,
            )
        }
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Unsupported data type for import. The image library only supports u8 and u16.",
        )),
    }
}

fn slab_block_writer<'a, N5, T>(
    n: &N5,
    dataset: &str,
    coord: GridCoord,
    slab_img_buff: impl Iterator<
        Item = Option<&'a (impl GenericImageView<Pixel = impl image::Pixel<Subpixel = T>> + 'a)>,
    >,
    data_attrs: &DatasetAttributes,
    elide_fill_value: Option<T>,
) -> Result<()>
where
    N5: N5Writer + Sync + Send + Clone + 'static,
    VecDataBlock<T>: n5::WriteableDataBlock,
    T: DataTypeBounds + image::Primitive,
{
    let block_loc = data_attrs
        .get_block_size()
        .iter()
        .cloned()
        .map(u64::from)
        .zip(coord.iter())
        .map(|(s, &i)| i * s)
        .collect::<GridCoord>();
    let crop_block_size = data_attrs
        .get_dimensions()
        .iter()
        .zip(data_attrs.get_block_size().iter().cloned().map(u64::from))
        .zip(coord.iter())
        .map(|((&d, s), &c)| {
            let offset = c * s;
            (std::cmp::min((c + 1) * s, d) - offset) as u32
        })
        .collect::<BlockCoord>();

    let mut data = Vec::with_capacity(crop_block_size.iter().product::<u32>() as usize);

    for img in slab_img_buff {
        let slice = img.as_ref().unwrap().view(
            block_loc[0] as u32,
            block_loc[1] as u32,
            crop_block_size[0] as u32,
            crop_block_size[1] as u32,
        );
        for (_, _, pixel) in slice.pixels() {
            data.push(pixel.channels()[0]);
        }
    }

    if let Some(fill_value) = elide_fill_value {
        // TODO: bad cast necessary until due to limited DynamicImage type support.
        if data.iter().all(|&v| v == fill_value) {
            return Ok(());
        }
    }

    let block = VecDataBlock::new(crop_block_size, coord, data);
    n.write_block(dataset, data_attrs, &block)?;

    Ok(())
}
