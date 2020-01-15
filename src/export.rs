use super::*;

use std::cmp::min;

use n5::WriteableDataBlock;
use n5::{
    data_type_match,
    data_type_rstype_replace,
};
use strfmt::Format;

#[derive(StructOpt, Debug)]
pub struct ExportOptions {
    /// Input N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Input N5 dataset
    #[structopt(name = "DATASET")]
    dataset: String,
    /// Format string for output filename
    #[structopt(name = "FILE_FORMAT")]
    file_format: String,
    /// Minimum X for export (inclusive)
    #[structopt(long = "x_min")]
    x_min: Option<u64>,
    /// Maximum X for export (exclusive)
    #[structopt(long = "x_max")]
    x_max: Option<u64>,
    /// Minimum Y for export (inclusive)
    #[structopt(long = "y_min")]
    y_min: Option<u64>,
    /// Maximum Y for export (exclusive)
    #[structopt(long = "y_max")]
    y_max: Option<u64>,
    /// Minimum Z for export (inclusive)
    #[structopt(long = "z_min")]
    z_min: Option<u64>,
    /// Maximum Z for export (exclusive)
    #[structopt(long = "z_max")]
    z_max: Option<u64>,
}

pub struct ExportCommand;

impl CommandType for ExportCommand {
    type Options = ExportOptions;

    fn run(opt: &Options, exp_opt: &Self::Options) -> Result<()> {
        let n = Arc::new(N5Filesystem::open(&exp_opt.n5_path)?);
        let started = Instant::now();

        let data_attrs = Arc::new(n.get_dataset_attributes(&exp_opt.dataset)?);
        let slab_size = u64::from(data_attrs.get_block_size()[2]);

        let min = [
            exp_opt.x_min.unwrap_or(0),
            exp_opt.y_min.unwrap_or(0),
            exp_opt.z_min.unwrap_or(0),
        ];
        let max = [
            exp_opt
                .x_max
                .unwrap_or_else(|| data_attrs.get_dimensions()[0]),
            exp_opt
                .y_max
                .unwrap_or_else(|| data_attrs.get_dimensions()[1]),
            exp_opt
                .z_max
                .unwrap_or_else(|| data_attrs.get_dimensions()[2]),
        ];

        let slab_min = min[2] / slab_size; // Floor
        let slab_max = max[2] / slab_size + u64::from(max[2] % slab_size > 0); // Ceiling

        let num_section_el = ((max[0] - min[0]) * (max[1] - min[1])) as usize;
        let dtype_size = data_attrs.get_data_type().size_of();

        let dataset = Arc::new(exp_opt.dataset.clone());
        let file_format = Arc::new(exp_opt.file_format.clone());

        let mut slab_img_buff = Vec::with_capacity(slab_size as usize);
        for _ in 0..slab_size {
            slab_img_buff.push(RwLock::new(vec![0; num_section_el * dtype_size]));
        }
        let slab_img_buff = slab_img_buff.into();

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
                slab_coord as usize,
                &min,
                &max,
            )?;
            pbar.write().unwrap().inc(1);
        }

        pbar.write().unwrap().finish();

        let num_bytes = max
            .iter()
            .zip(min.iter())
            .map(|(&ma, &mi)| ma - mi)
            .product::<u64>() as usize
            * data_attrs.get_data_type().size_of();
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

fn dtype_to_color(dtype: DataType) -> image::ColorType {
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
    slab_img_buff: &Arc<[RwLock<Vec<u8>>]>,
    file_format: &Arc<String>,
    data_attrs: &Arc<DatasetAttributes>,
    slab_coord: usize,
    min_vox: &[u64; 3],
    max_vox: &[u64; 3],
) -> Result<()> {
    let (slab_coord_iter, total_coords) =
        bounded_slab_coord_iter(&*data_attrs, 2, slab_coord as u64, min_vox, max_vox);
    let mut slab_coord_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(total_coords);

    let slab_z = slab_coord as u64 * u64::from(data_attrs.get_block_size()[2]);
    let mut slab_min_i = *min_vox;
    let mut slab_max_i = *max_vox;
    slab_min_i[2] = std::cmp::max(min_vox[2] - slab_z, 0);
    // let slab_min = min[2].checked_sub(slab_z).unwrap_or(0);
    slab_max_i[2] = min(
        max_vox[2] - slab_z,
        u64::from(data_attrs.get_block_size()[2]),
    );

    let slab_min = [
        slab_min_i[0] as usize,
        slab_min_i[1] as usize,
        slab_min_i[2] as usize,
    ];
    let slab_max = [
        slab_max_i[0] as usize,
        slab_max_i[1] as usize,
        slab_max_i[2] as usize,
    ];

    for coord in slab_coord_iter {
        let n = n.clone();
        let dataset = dataset.clone();
        let slab_img_buff = slab_img_buff.clone();
        let data_attrs = data_attrs.clone();
        slab_coord_jobs.push(pool.spawn_fn(move || {
            slab_block_dispatch(
                &*n,
                &*dataset,
                coord.into(),
                &slab_img_buff,
                &*data_attrs,
                &slab_min,
                &slab_max,
            )
        }));
    }

    futures::future::join_all(slab_coord_jobs).wait()?;

    let mut slab_load_jobs: Vec<CpuFuture<_, std::io::Error>> =
        Vec::with_capacity(slab_max[2] - slab_min[2]);

    for slab_ind in slab_min[2]..slab_max[2] {
        let data_attrs = data_attrs.clone();
        let slab_img_buff = slab_img_buff.clone();
        let mut params = std::collections::HashMap::new();
        params.insert("z".into(), slab_z as usize + slab_ind);
        let filename = file_format.format(&params).unwrap();

        let width = (max_vox[0] - min_vox[0]) as u32;
        let height = (max_vox[1] - min_vox[1]) as u32;
        slab_load_jobs.push(pool.spawn_fn(move || {
            image::save_buffer(
                filename,
                &slab_img_buff[slab_ind as usize].read().unwrap(),
                width,
                height,
                dtype_to_color(*data_attrs.get_data_type()),
            )
            .unwrap();

            Ok(())
        }));
    }

    futures::future::join_all(slab_load_jobs).wait()?;

    Ok(())
}

fn slab_block_dispatch<N5>(
    n: &N5,
    dataset: &str,
    coord: GridCoord,
    slab_img_buff: &[RwLock<Vec<u8>>],
    data_attrs: &DatasetAttributes,
    slab_min: &[usize; 3],
    slab_max: &[usize; 3],
) -> Result<()>
where
    N5: N5Reader + Sync + Send + Clone + 'static,
{
    data_type_match! {
        *data_attrs.get_data_type(),
        {
            let block = n.read_block::<RsType>(dataset, data_attrs, coord.clone());
            slab_block_reader::<RsType>(
                &coord,
                block,
                slab_img_buff,
                data_attrs,
                slab_min,
                slab_max)
        }
    }
}

fn slab_block_reader<T>(
    coord: &[u64],
    block: Result<Option<VecDataBlock<T>>>,
    slab_img_buff: &[RwLock<Vec<u8>>],
    data_attrs: &DatasetAttributes,
    slab_min: &[usize; 3],
    slab_max: &[usize; 3],
) -> Result<()>
where
    T: DataTypeBounds,
    VecDataBlock<T>: n5::WriteableDataBlock,
{
    let block_loc = data_attrs
        .get_block_size()
        .iter()
        .cloned()
        .map(u64::from)
        .zip(coord.iter())
        .map(|(s, &i)| i * s)
        .collect::<Vec<_>>();
    let crop_block_size = data_attrs
        .get_dimensions()
        .iter()
        .zip(data_attrs.get_block_size().iter().cloned().map(u64::from))
        .zip(coord.iter())
        .map(|((&d, s), &c)| {
            let offset = c * s;
            (std::cmp::min((c + 1) * s, d) - offset) as u32
        })
        .collect::<Vec<_>>();
    let dtype_size = data_attrs.get_data_type().size_of();

    let block_min = slab_min
        .iter()
        .zip(block_loc.iter())
        .enumerate()
        .map(|(i, (&s, &b))| {
            if i == 2 {
                s
            } else {
                (s).saturating_sub(b as usize)
            }
        })
        .collect::<Vec<_>>();
    let block_max = slab_max
        .iter()
        .zip(block_loc.iter())
        .zip(crop_block_size.iter())
        .enumerate()
        .map(|(i, ((&ma, &bl), &cbs))| {
            if i == 2 {
                min(ma, cbs as usize)
            } else {
                min(ma - bl as usize, cbs as usize)
            }
        })
        .collect::<Vec<_>>();

    let byte_data: Vec<u8> = match block? {
        Some(block) => {
            let num_el = block.get_num_elements() as usize;
            let mut byte_data = Vec::with_capacity(num_el * dtype_size);
            block.write_data(&mut byte_data)?;
            assert_eq!(byte_data.len(), num_el * dtype_size);

            byte_data
        }
        None => {
            let num_el = crop_block_size.iter().product::<u32>() as usize;
            vec![0; num_el * dtype_size]
        }
    };

    let img_row_vox = slab_max[0] - slab_min[0];
    let img_row_bytes = img_row_vox * dtype_size;
    let img_block_row_bytes = (block_max[0] - block_min[0]) * dtype_size;
    let block_row_bytes = (crop_block_size[0] as usize) * dtype_size;

    for z in block_min[2]..block_max[2] {
        let mut data_offset = dtype_size
            * (z * ((crop_block_size[0] * crop_block_size[1]) as usize)
                + block_min[1] * crop_block_size[0] as usize
                + block_min[0]);
        let mut offset = dtype_size
            * (img_row_vox * (block_loc[1] as usize + block_min[1]).saturating_sub(slab_min[1])
                + (block_loc[0] as usize + block_min[0]).saturating_sub(slab_min[0]));

        let mut img_write = slab_img_buff[z].write().unwrap();
        for _y in block_min[1]..block_max[1] {
            img_write[offset..(offset + img_block_row_bytes)]
                .copy_from_slice(&byte_data[data_offset..(data_offset + img_block_row_bytes)]);
            offset += img_row_bytes;
            data_offset += block_row_bytes;
        }
    }

    Ok(())
}
