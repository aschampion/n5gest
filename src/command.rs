use anyhow::Result;
use structopt::StructOpt;

use super::Options;

mod bench_read;
#[cfg(feature = "nightly")]
mod cast;
mod crop_blocks;
mod delete_uniform_blocks;
mod export;
mod import;
mod list;
mod map;
mod map_fold;
mod recompress;
mod stat;
mod validate_blocks;

#[derive(StructOpt, Debug)]
pub(super) enum Command {
    /// List all datasets under an N5 root.
    #[structopt(name = "ls")]
    List(list::ListOptions),
    /// Retrieve metadata about the number of blocks that exists and their
    /// timestamps.
    #[structopt(name = "stat")]
    Stat(stat::StatOptions),
    /// Benchmark reading an entire dataset.
    #[structopt(name = "bench-read")]
    BenchRead(bench_read::BenchReadOptions),
    /// Cast an existing dataset into a new dataset with a given
    /// data type.
    #[cfg(feature = "nightly")]
    #[structopt(name = "cast")]
    Cast(cast::CastOptions),
    /// Crop wrongly sized blocks to match dataset dimensions at the end of a
    /// given axis.
    #[structopt(name = "crop-blocks")]
    CropBlocks(crop_blocks::CropBlocksOptions),
    /// Delete blocks uniformly filled with a given value, such as empty blocks.
    #[structopt(name = "delete-uniform-blocks")]
    DeleteUniformBlocks(delete_uniform_blocks::DeleteUniformBlocksOptions),
    /// Export a sequence of image files from a series of z-sections.
    #[structopt(name = "export")]
    Export(export::ExportOptions),
    /// Import a sequence of image files as a series of z-sections into a 3D
    /// N5 dataset.
    #[structopt(name = "import")]
    Import(import::ImportOptions),
    /// Run simple math expressions mapping values to new datasets.
    /// For example, to clip values in a dataset:
    /// `map example.n5 dataset_in example.n5 dataset_out "min(128, x)"`
    /// Note that this converts back and forth to `f64` for the calculation.
    #[structopt(name = "map")]
    Map(map::MapOptions),
    /// Run simple math expressions as folds over blocks.
    /// For example, to find the maximum value in a positive dataset:
    /// `map-fold example.n5 dataset 0 "max(acc, x)"`
    #[structopt(name = "map-fold")]
    MapFold(map_fold::MapFoldOptions),
    /// Recompress an existing dataset into a new dataset with a given
    /// compression.
    #[structopt(name = "recompress")]
    Recompress(recompress::RecompressOptions),
    /// Report malformed blocks.
    #[structopt(name = "validate-blocks")]
    ValidateBlocks(validate_blocks::ValidateBlocksOptions),
}

pub(crate) trait CommandType {
    type Options: StructOpt;

    fn run(opt: &Options, com_opt: &Self::Options) -> Result<()>;
}

pub(super) fn dispatch(opt: &Options) -> Result<()> {
    #[rustfmt::skip]
    match opt.command {
        Command::List(ref ls_opt) =>
            list::ListCommand::run(opt, ls_opt)?,
        Command::Stat(ref st_opt) =>
            stat::StatCommand::run(opt, st_opt)?,
        Command::BenchRead(ref br_opt) =>
            bench_read::BenchReadCommand::run(opt, br_opt)?,
        #[cfg(feature = "nightly")]
        Command::Cast(ref cast_opt) =>
            cast::CastCommand::run(opt, cast_opt)?,
        Command::CropBlocks(ref crop_opt) =>
            crop_blocks::CropBlocksCommand::run(opt, crop_opt)?,
        Command::DeleteUniformBlocks(ref dub_opt) =>
            delete_uniform_blocks::DeleteUniformBlocksCommand::run(opt, dub_opt)?,
        Command::Export(ref exp_opt) =>
            export::ExportCommand::run(opt, exp_opt)?,
        Command::Import(ref imp_opt) =>
            import::ImportCommand::run(opt, imp_opt)?,
        Command::Map(ref m_opt) =>
            map::MapCommand::run(opt, m_opt)?,
        Command::MapFold(ref mf_opt) =>
            map_fold::MapFoldCommand::run(opt, mf_opt)?,
        Command::Recompress(ref com_opt) =>
            recompress::RecompressCommand::run(opt, com_opt)?,
        Command::ValidateBlocks(ref vb_opt) =>
            validate_blocks::ValidateBlocksCommand::run(opt, vb_opt)?,
    };

    Ok(())
}
