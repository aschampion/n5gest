use crate::common::*;

use std::convert::TryFrom;
use std::fs::File;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::RwLock;

use byteorder::{
    ByteOrder,
    NativeEndian,
};
use futures::Future;
use futures_cpupool::{
    CpuFuture,
    CpuPool,
};
use image::{
    DynamicImage,
    GenericImageView,
};
use n5::smallvec::smallvec;
use tiff::decoder::Decoder;

use crate::default_progress_bar;
use crate::iterator::{
    CoordIteratorFactory,
    GridSlabCoordIter,
};

use super::{
    color_to_dtype,
    slab_block_dispatch,
    BlockSizeParam,
};

#[derive(StructOpt, Debug)]
pub struct ImportTiffOptions {
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
    /// Tiff file to import
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,
    /// Uniform blocks of this value will not be written
    #[structopt(long = "elide_fill_value")]
    elide_fill_value: Option<String>,
}

pub struct ImportTiffCommand;

struct TiffMetadataIterator {
    path: PathBuf,
    decoder: Decoder<File>,
    current_image: usize,
    finished: bool,
}

impl TiffMetadataIterator {
    fn open(path: PathBuf) -> anyhow::Result<Self> {
        let file = File::open(&path)?;
        let decoder = Decoder::new(file)?;
        Ok(TiffMetadataIterator {
            path,
            decoder,
            current_image: 0,
            finished: false,
        })
    }

    fn rewind(&mut self) -> anyhow::Result<()> {
        self.decoder.goto_offset(0)?;
        // self.decoder = self.decoder.init()?;
        self.current_image = 0;
        self.finished = false;
        Ok(())
    }

    fn next_image(&mut self) -> anyhow::Result<bool> {
        if self.decoder.more_images() {
            self.decoder.next_image()?;
            self.current_image += 1;
            Ok(true)
        } else {
            self.finished = true;
            Ok(false)
        }
    }

    fn read_image(&mut self) -> anyhow::Result<DynamicImage> {
        let metadata = self.metadata()?;
        let buffer = self.decoder.read_image()?;

        let image_decoder = ImageBuffer { metadata, buffer };
        let image = DynamicImage::from_decoder(image_decoder)?;
        Ok(image)
    }

    fn read_image_index(&mut self, idx: usize) -> anyhow::Result<DynamicImage> {
        if self.current_image > idx {
            self.rewind()?;
        }
        while self.current_image < idx {
            if !self.next_image()? {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "Requested image beyond end of TIFF stack",
                )
                .into());
            }
        }

        self.read_image()
    }

    fn count_images(&mut self) -> anyhow::Result<usize> {
        while self.next_image()? {}
        Ok(self.current_image + 1)
    }

    fn metadata(&mut self) -> anyhow::Result<ImageMetadata> {
        Ok(ImageMetadata {
            dimensions: self.decoder.dimensions()?,
            color: tiff_color_to_image(self.decoder.colortype()?)?,
        })
    }
}

impl Clone for TiffMetadataIterator {
    fn clone(&self) -> Self {
        Self::open(self.path.clone()).unwrap()
    }
}

impl Iterator for TiffMetadataIterator {
    type Item = anyhow::Result<ImageMetadata>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        let item = self.metadata();
        if item.is_err() {
            return Some(item);
        }
        Some(self.next_image().and(item))
    }
}

fn tiff_color_to_image(color: tiff::ColorType) -> Result<image::ColorType> {
    match color {
        tiff::ColorType::Gray(8) => Ok(image::ColorType::L8),
        tiff::ColorType::Gray(16) => Ok(image::ColorType::L16),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported color: {:?}", color),
        )),
    }
}

struct ImageMetadata {
    dimensions: (u32, u32),
    color: image::ColorType,
}

struct ImageBuffer {
    metadata: ImageMetadata,
    buffer: tiff::decoder::DecodingResult,
}

// TODO: this duplicates functionality from `image` that it should eventually
// make more transparent
impl<'a> image::ImageDecoder<'a> for ImageBuffer {
    type Reader = std::io::Cursor<Vec<u8>>;

    fn dimensions(&self) -> (u32, u32) {
        self.metadata.dimensions
    }

    fn color_type(&self) -> image::ColorType {
        self.metadata.color
    }

    fn into_reader(self) -> image::ImageResult<Self::Reader> {
        let buf = match self.buffer {
            tiff::decoder::DecodingResult::U8(v) => v,
            tiff::decoder::DecodingResult::U16(v) => vec_u16_into_u8(v),
            tiff::decoder::DecodingResult::U32(_) => return Err(err_unknown_color_type(32)),
            tiff::decoder::DecodingResult::U64(_) => return Err(err_unknown_color_type(64)),
        };

        Ok(std::io::Cursor::new(buf))
    }

    fn read_image(self, buf: &mut [u8]) -> image::ImageResult<()> {
        assert_eq!(u64::try_from(buf.len()), Ok(self.total_bytes()));
        match self.buffer {
            tiff::decoder::DecodingResult::U8(v) => {
                buf.copy_from_slice(&v);
            }
            tiff::decoder::DecodingResult::U16(v) => {
                NativeEndian::write_u16_into(&v, buf);
            }
            tiff::decoder::DecodingResult::U32(_) => return Err(err_unknown_color_type(32)),
            tiff::decoder::DecodingResult::U64(_) => return Err(err_unknown_color_type(64)),
        }
        Ok(())
    }
}

// TODO: this duplicates functionality from `image` that it should eventually
// make more transparent.
fn err_unknown_color_type(value: u8) -> image::ImageError {
    image::ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
        image::ImageFormat::Tiff.into(),
        image::error::UnsupportedErrorKind::Color(image::ExtendedColorType::Unknown(value)),
    ))
}

// TODO: this duplicates functionality from `image` that it should eventually
// make more transparent.
fn vec_u16_into_u8(vec: Vec<u16>) -> Vec<u8> {
    let mut new = vec![0; vec.len() * std::mem::size_of::<u16>()];
    NativeEndian::write_u16_into(&vec[..], &mut new[..]);
    new
}

impl CommandType for ImportTiffCommand {
    type Options = ImportTiffOptions;

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

        let mut iter = TiffMetadataIterator::open(imp_opt.file.clone())?;
        let ref_img = iter.next().unwrap()?;
        let xy_dims = ref_img.dimensions;
        let dtype = color_to_dtype(ref_img.color);
        let z_dim = iter.count_images()?;

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

        let pool = match opt.threads {
            Some(threads) => CpuPool::new(threads),
            None => CpuPool::new_num_cpus(),
        };
        let pbar = RwLock::new(default_progress_bar(z_dim as u64));
        let elide_fill_value = imp_opt.elide_fill_value.clone().map(Arc::new);

        for (slab_coord, start) in (0..z_dim).step_by(slab_size).enumerate() {
            let end = std::cmp::min(z_dim, start + slab_size);
            import_slab(
                &n,
                &dataset,
                &pool,
                &iter,
                start..end,
                &slab_img_buff,
                &data_attrs,
                slab_coord,
                elide_fill_value.clone(),
            )?;
            pbar.write().unwrap().inc((end - start) as u64);
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

fn import_slab<N5: N5Writer + Sync + Send + Clone + 'static>(
    n: &Arc<N5>,
    dataset: &Arc<String>,
    pool: &CpuPool,
    tiff: &TiffMetadataIterator,
    slab: Range<usize>,
    slab_img_buff: &Arc<RwLock<Vec<Option<DynamicImage>>>>,
    data_attrs: &Arc<DatasetAttributes>,
    slab_coord: usize,
    elide_fill_value: Option<Arc<String>>,
) -> anyhow::Result<()> {
    let mut slab_load_jobs: Vec<CpuFuture<_, anyhow::Error>> = Vec::with_capacity(slab.len());
    {
        let mut buff_vec = slab_img_buff.write().unwrap();
        buff_vec.clear();
        buff_vec.resize(slab.len(), None);
    }

    for (i, coord) in slab.enumerate() {
        let mut tiff = tiff.clone();
        let data_attrs = data_attrs.clone();
        let slab_img_buff = slab_img_buff.clone();
        slab_load_jobs.push(pool.spawn_fn(move || {
            let image = tiff.read_image_index(coord)?;
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
        }));
    }

    futures::future::join_all(slab_load_jobs).wait()?;

    let coord_iter = GridSlabCoordIter {
        axis: 2,
        slab_coord: slab_coord as u64,
    };
    let (slab_coord_iter, total_coords) = coord_iter.coord_iter(&*data_attrs);
    let mut slab_coord_jobs: Vec<CpuFuture<_, std::io::Error>> = Vec::with_capacity(total_coords);

    for coord in slab_coord_iter {
        let n = n.clone();
        let dataset = dataset.clone();
        let slab_img_buff = slab_img_buff.clone();
        let data_attrs = data_attrs.clone();
        let elide_fill_value = elide_fill_value.clone();
        slab_coord_jobs.push(pool.spawn_fn(move || {
            let slab_read = slab_img_buff.read().unwrap();
            slab_block_dispatch(
                &*n,
                &*dataset,
                coord.into(),
                &slab_read,
                &*data_attrs,
                elide_fill_value,
            )
        }));
    }

    futures::future::join_all(slab_coord_jobs).wait()?;

    Ok(())
}
