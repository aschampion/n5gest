use super::*;

use std::str::FromStr;

use image::{
    DynamicImage,
    GenericImageView,
};


#[derive(StructOpt, Debug)]
pub struct ImportOptions {
    /// Ouput N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Ouput N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// New N5 compression (JSON)
    #[structopt(name = "COMPRESSION")]
    compression: String,
    /// Block size
    #[structopt(name = "BLOCK_SIZE")]
    block_size: BlockSizeParam,
    /// Files to import, ordered by increasing z
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,
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

    fn run(opt: &Options, imp_opt: &Self::Options) -> Result<()> {
        let n = Arc::new(N5Filesystem::open_or_create(&imp_opt.n5_path)?);

        if n.exists(&imp_opt.dataset) {
            return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists,
                format!("Dataset {} already exists", imp_opt.dataset)));
        }
        let compression: CompressionType = serde_json::from_str(&imp_opt.compression).unwrap();
        let started = Instant::now();

        let files = imp_opt.files.as_slice();
        let z_dim = files.len();
        let ref_img = image::open(&files[0]).unwrap();
        let xy_dims = ref_img.dimensions();
        let dtype = color_to_dtype(ref_img.color());

        let data_attrs = Arc::new(DatasetAttributes::new(
            vec![i64::from(xy_dims.0), i64::from(xy_dims.1), z_dim as i64],
            imp_opt.block_size.0.iter().map(|&b| b as i32).collect(),
            dtype,
            compression));

        let dataset = Arc::new(imp_opt.dataset.clone());
        n.create_dataset(&dataset, &data_attrs)?;
        let slab_size = imp_opt.block_size.0[2];

        let slab_img_buff = Arc::new(RwLock::new(Vec::<Option<DynamicImage>>::with_capacity(slab_size)));

        let pool = match opt.threads {
            Some(threads) => CpuPool::new(threads),
            None => CpuPool::new_num_cpus(),
        };
        let pbar = RwLock::new(default_progress_bar(files.len() as u64));

        for (slab_coord, slab_files) in files.chunks(slab_size).enumerate() {
            import_slab(
                &n,
                &dataset,
                &pool,
                &slab_img_buff,
                &slab_files,
                &data_attrs,
                slab_coord)?;
            pbar.write().unwrap().inc(slab_files.len() as u64);
        }

        pbar.write().unwrap().finish();

        let num_bytes = data_attrs.get_num_elements() * data_attrs.get_data_type().size_of();
        let elapsed = started.elapsed();
        println!("Wrote {} (uncompressed) in {}",
            HumanBytes(num_bytes as u64),
            HumanDuration(elapsed));
        let throughput = 1e9 * (num_bytes as f64) /
            (1e9 * (elapsed.as_secs() as f64) + f64::from(elapsed.subsec_nanos()));
        println!("({} / s)", HumanBytes(throughput as u64));

        Ok(())
    }
}


fn color_to_dtype(color: image::ColorType) -> DataType {
    match color {
        image::ColorType::Gray(8) => DataType::UINT8,
        image::ColorType::Gray(16) => DataType::UINT16,
        image::ColorType::Gray(32) => DataType::UINT32,
        image::ColorType::Gray(64) => DataType::UINT64,
        _ => unimplemented!(),
    }
}


fn import_slab<N5: N5Writer + Sync + Send + Clone + 'static>(
    n: &Arc<N5>,
    dataset: &Arc<String>,
    pool: &CpuPool,
    slab_img_buff: &Arc<RwLock<Vec<Option<DynamicImage>>>>,
    slab_files: &[PathBuf],
    data_attrs: &Arc<DatasetAttributes>,
    slab_coord: usize,
) -> Result<()> {

    let mut slab_load_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(slab_files.len());
    slab_img_buff.write().unwrap().clear();
    for _ in 0..slab_files.len() {
        slab_img_buff.write().unwrap().push(None);
    };

    for (i, file) in slab_files.iter().enumerate() {
        let owned_file = file.clone();
        let data_attrs = data_attrs.clone();
        let slab_img_buff = slab_img_buff.clone();
        slab_load_jobs.push(pool.spawn_fn(move || {
            let image = image::open(&owned_file).unwrap();
            assert_eq!(color_to_dtype(image.color()), *data_attrs.get_data_type());
            assert_eq!(i64::from(image.dimensions().0), data_attrs.get_dimensions()[0]);
            assert_eq!(i64::from(image.dimensions().1), data_attrs.get_dimensions()[1]);
            slab_img_buff.write().unwrap()[i] = Some(image);

            Ok(())
        }));
    }

    futures::future::join_all(slab_load_jobs).wait()?;

    let (slab_coord_iter, total_coords) = slab_coord_iter(&*data_attrs, 2, slab_coord as i64);
    let mut slab_coord_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(total_coords);

    for coord in slab_coord_iter {
        let n = n.clone();
        let dataset = dataset.clone();
        let slab_img_buff = slab_img_buff.clone();
        let data_attrs = data_attrs.clone();
        slab_coord_jobs.push(pool.spawn_fn(move || {
            let slab_read = slab_img_buff.read().unwrap();
            slab_block_dispatch(
                &*n,
                &*dataset,
                coord,
                &slab_read,
                &*data_attrs)
        }));
    }

    futures::future::join_all(slab_coord_jobs).wait()?;

    Ok(())
}

fn slab_block_dispatch<N5>(
    n: &N5,
    dataset: &str,
    coord: Vec<i64>,
    slab_img_buff: &[Option<DynamicImage>],
    data_attrs: &DatasetAttributes,
) -> Result<()>
where
    N5: N5Writer + Sync + Send + Clone + 'static {

    match *data_attrs.get_data_type() {
        DataType::UINT8 => {
            slab_block_writer::<_, u8>(
                n,
                dataset,
                coord,
                slab_img_buff,
                data_attrs)
        },
        DataType::UINT16 => {
            slab_block_writer::<_, u16>(
                n,
                dataset,
                coord,
                slab_img_buff,
                data_attrs)
        },
        DataType::UINT32 => {
            slab_block_writer::<_, u32>(
                n,
                dataset,
                coord,
                slab_img_buff,
                data_attrs)
        },
        DataType::UINT64 => {
            slab_block_writer::<_, u64>(
                n,
                dataset,
                coord,
                slab_img_buff,
                data_attrs)
        },
        _ => unimplemented!(),
    }
}

fn slab_block_writer<N5, T>(
    n: &N5,
    dataset: &str,
    coord: Vec<i64>,
    slab_img_buff: &[Option<DynamicImage>],
    data_attrs: &DatasetAttributes,
) -> Result<()>
where
    N5: N5Writer + Sync + Send + Clone + 'static,
    T: 'static + std::fmt::Debug + Clone + PartialEq + Sync + Send + num_traits::Zero,
    DataType: TypeReflection<T> + DataBlockCreator<T>,
    VecDataBlock<T>: n5::DataBlock<T> {

    let block_loc = data_attrs.get_block_size().iter().cloned().map(i64::from)
        .zip(coord.iter())
        .map(|(s, &i)| i*s)
        .collect::<Vec<_>>();
    let crop_block_size = data_attrs.get_dimensions().iter()
        .zip(data_attrs.get_block_size().iter().cloned().map(i64::from))
        .zip(coord.iter())
        .map(|((&d, s), &c)| {
            let offset = c * s;
            (std::cmp::min((c + 1) * s, d) - offset) as i32
        })
        .collect::<Vec<_>>();

    let mut data = Vec::with_capacity(crop_block_size.iter().product::<i32>() as usize);

    for img in slab_img_buff {
        let slice = img.as_ref().unwrap().view(
            block_loc[0] as u32,
            block_loc[1] as u32,
            crop_block_size[0] as u32,
            crop_block_size[1] as u32);
        for (_, _, pixel) in slice.pixels() {
            data.push(pixel[0]);
        }
    }

    let block = VecDataBlock::new(
        crop_block_size,
        coord,
        data);
    n.write_block(dataset, data_attrs, &block)?;

    Ok(())
}
