use super::*;

use std::cmp::min;

use n5::WriteableDataBlock;
use strfmt::Format;


#[derive(StructOpt, Debug)]
pub struct ExportOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// Minimum Z for export (inclusive)
    #[structopt(name = "Z_MIN")]
    z_min: usize,
    /// Maximum Z for export (exclusive)
    #[structopt(name = "Z_MAX")]
    z_max: usize,
    /// Format string for output filename
    #[structopt(name = "FILE_FORMAT")]
    file_format: String,
}

pub struct ExportCommand;

impl CommandType for ExportCommand {
    type Options = ExportOptions;

    fn run(opt: &Options, exp_opt: &Self::Options) -> Result<()> {
        let n = Arc::new(N5Filesystem::open(&exp_opt.n5_path)?);
        let started = Instant::now();

        let data_attrs = Arc::new(n.get_dataset_attributes(&exp_opt.dataset)?);
        let slab_size = data_attrs.get_block_size()[2] as usize;

        let slab_min = exp_opt.z_min / slab_size; // Floor
        let slab_max: usize = exp_opt.z_max / slab_size +
            usize::from(exp_opt.z_max % slab_size > 0); // Ceiling

        let num_section_el =
                data_attrs.get_dimensions()[0] as usize *
                data_attrs.get_dimensions()[1] as usize;
        let dtype_size = data_attrs.get_data_type().size_of();

        let dataset = Arc::new(exp_opt.dataset.clone());
        let file_format = Arc::new(exp_opt.file_format.clone());

        let mut slab_img_buff = Vec::with_capacity(slab_size);
        for _ in 0..slab_size {
            slab_img_buff.push(RwLock::new(vec![0; num_section_el * dtype_size]));
        }
        let slab_img_buff = Arc::new(slab_img_buff);

        let pool = match opt.threads {
            Some(threads) => CpuPool::new(threads),
            None => CpuPool::new_num_cpus(),
        };
        let pbar = RwLock::new(default_progress_bar((slab_max - slab_min) as u64));

        for slab_coord in slab_min..slab_max {
            export_slab(
                &n,
                &dataset,
                &pool,
                &slab_img_buff,
                &file_format,
                &data_attrs,
                slab_coord,
                exp_opt.z_min,
                exp_opt.z_max)?;
            pbar.write().unwrap().inc(1);
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


fn dtype_to_color(dtype: &DataType) -> image::ColorType {
    match dtype {
        DataType::UINT8 => image::ColorType::Gray(8),
        DataType::UINT16 => image::ColorType::Gray(16),
        DataType::UINT32 => image::ColorType::Gray(32),
        DataType::UINT64 => image::ColorType::Gray(64),
        _ => unimplemented!(),
    }
}


fn export_slab<N5: N5Reader + Sync + Send + Clone + 'static>(
    n: &Arc<N5>,
    dataset: &Arc<String>,
    pool: &CpuPool,
    slab_img_buff: &Arc<Vec<RwLock<Vec<u8>>>>,
    file_format: &Arc<String>,
    data_attrs: &Arc<DatasetAttributes>,
    slab_coord: usize,
    z_min: usize,
    z_max: usize,
) -> Result<()> {

    let (slab_coord_iter, total_coords) = slab_coord_iter(&*data_attrs, 2, slab_coord as i64);
    let mut slab_coord_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(total_coords);

    let slab_z = slab_coord * data_attrs.get_block_size()[2] as usize;
    let slab_min = z_min.checked_sub(slab_z).unwrap_or(0);
    let slab_max = min(z_max - slab_z, data_attrs.get_block_size()[2] as usize);

    for coord in slab_coord_iter {
        let n = n.clone();
        let dataset = dataset.clone();
        let slab_img_buff = slab_img_buff.clone();
        let data_attrs = data_attrs.clone();
        slab_coord_jobs.push(pool.spawn_fn(move || {
            slab_block_dispatch(
                &*n,
                &*dataset,
                coord,
                &slab_img_buff,
                &*data_attrs,
                slab_min,
                slab_max)
        }));
    }

    futures::future::join_all(slab_coord_jobs).wait()?;

    let mut slab_load_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(slab_max - slab_min);

    for slab_ind in slab_min..slab_max {
        let data_attrs = data_attrs.clone();
        let slab_img_buff = slab_img_buff.clone();
        let mut params = std::collections::HashMap::new();
        params.insert("z".into(), slab_z + slab_ind);
        let filename = file_format.format(&params).unwrap();

        slab_load_jobs.push(pool.spawn_fn(move || {
            image::save_buffer(
                filename,
                &slab_img_buff[slab_ind].read().unwrap(),
                data_attrs.get_dimensions()[0] as u32,
                data_attrs.get_dimensions()[1] as u32,
                dtype_to_color(data_attrs.get_data_type()),
            ).unwrap();

            Ok(())
        }));
    }

    futures::future::join_all(slab_load_jobs).wait()?;

    Ok(())
}

fn slab_block_dispatch<N5>(
    n: &N5,
    dataset: &str,
    coord: Vec<i64>,
    slab_img_buff: &[RwLock<Vec<u8>>],
    data_attrs: &DatasetAttributes,
    slab_min: usize,
    slab_max: usize,
) -> Result<()>
where
    N5: N5Reader + Sync + Send + Clone + 'static {

    match *data_attrs.get_data_type() {
        DataType::UINT8 => {
            let block = n.read_block::<u8>(dataset, data_attrs, coord.clone());
            slab_block_reader::<u8>(
                &coord,
                block,
                slab_img_buff,
                data_attrs,
                slab_min,
                slab_max)
        },
        DataType::UINT16 => {
            let block = n.read_block::<u16>(dataset, data_attrs, coord.clone());
            slab_block_reader::<u16>(
                &coord,
                block,
                slab_img_buff,
                data_attrs,
                slab_min,
                slab_max)
        },
        DataType::UINT32 => {
            let block = n.read_block::<u32>(dataset, data_attrs, coord.clone());
            slab_block_reader::<u32>(
                &coord,
                block,
                slab_img_buff,
                data_attrs,
                slab_min,
                slab_max)
        },
        DataType::UINT64 => {
            let block = n.read_block::<u64>(dataset, data_attrs, coord.clone());
            slab_block_reader::<u64>(
                &coord,
                block,
                slab_img_buff,
                data_attrs,
                slab_min,
                slab_max)
        },
        _ => unimplemented!(),
    }
}

fn slab_block_reader<T>(
    coord: &[i64],
    block: Result<Option<VecDataBlock<T>>>,
    slab_img_buff: &[RwLock<Vec<u8>>],
    data_attrs: &DatasetAttributes,
    slab_min: usize,
    slab_max: usize,
) -> Result<()>
where
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
    let dtype_size = data_attrs.get_data_type().size_of();

    let byte_data: Vec<u8> = match block? {
        Some(block) => {
            let num_el = block.get_num_elements() as usize;
            let mut byte_data = Vec::with_capacity(num_el * dtype_size);
            block.write_data(&mut byte_data)?;
            assert_eq!(byte_data.len(), num_el * dtype_size);

            byte_data
        },
        None => {
            let num_el = crop_block_size.iter().product::<i32>() as usize;
            vec![0; num_el * dtype_size]
        },
    };

    let mut data_offset = dtype_size * slab_min *
        (crop_block_size[0] * crop_block_size[1]) as usize;
    let row_bytes = (crop_block_size[0] as usize) * dtype_size;

    for z in slab_min..min(slab_max, crop_block_size[2] as usize) {
        let mut offset = dtype_size *
            (data_attrs.get_dimensions()[0] * block_loc[1] + block_loc[0]) as usize;

        let mut img_write = slab_img_buff[z].write().unwrap();
        for _y in 0..crop_block_size[1] {
            img_write[offset..(offset + row_bytes)].copy_from_slice(
                &byte_data[data_offset..(data_offset + row_bytes)]);
            offset += dtype_size * data_attrs.get_dimensions()[0] as usize;
            data_offset += row_bytes;
        }
    }

    Ok(())
}
