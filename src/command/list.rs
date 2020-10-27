use std::collections::HashMap;
use std::convert::TryFrom;

use itertools::Itertools;
use prettytable::Table;

use crate::common::*;
use crate::MetricPrefix;

#[derive(StructOpt, Debug)]
pub struct ListOptions {
    /// N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
    /// Group root path
    #[structopt(name = "GROUP", default_value = "")]
    group_path: String,
    #[structopt(long = "sort", possible_values = &Sorting::variants(), default_value = "numeric-suffix")]
    sorting: Sorting,
}

// TODO: Clap 3 will allow this mostly to be derived and documented.
#[derive(Debug)]
pub enum Sorting {
    Lexicographic,
    NumericSuffix,
    MaxVox,
    MaxBlocks,
    Dtype,
    Compression,
}

impl std::str::FromStr for Sorting {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        #[allow(deprecated, unused_imports)]
        match s {
            "lexicographic" => Ok(Self::Lexicographic),
            "numeric-suffix" => Ok(Self::NumericSuffix),
            "max-vox" => Ok(Self::MaxVox),
            "max-blocks" => Ok(Self::MaxBlocks),
            "dtype" => Ok(Self::Dtype),
            "compression" => Ok(Self::Compression),
            _ => Err(format!("valid values: {}", Self::variants().join(", "))),
        }
    }
}

impl Sorting {
    fn variants() -> &'static [&'static str] {
        &[
            "lexicographic",
            "numeric-suffix",
            "max-vox",
            "max-blocks",
            "dtype",
            "compression",
        ]
    }

    fn sort(
        &self,
        a: &(String, DatasetAttributes),
        b: &(String, DatasetAttributes),
    ) -> std::cmp::Ordering {
        match self {
            Sorting::Lexicographic => Ord::cmp(&a.0, &b.0),
            Sorting::NumericSuffix => {
                if let Some((left, right)) =
                    a.0.split('/')
                        .zip(b.0.split('/'))
                        .find(|(last, curr)| last != curr)
                {
                    let l = common_prefix_len(left, right);
                    let left = &left[l..];
                    let right = &right[l..];
                    if let (Ok(left), Ok(right)) = (left.parse::<u64>(), right.parse::<u64>()) {
                        return Ord::cmp(&left, &right);
                    }
                }

                Sorting::Lexicographic.sort(a, b)
            }
            Sorting::MaxVox => Ord::cmp(&a.1.get_num_elements(), &b.1.get_num_elements()),
            Sorting::MaxBlocks => Ord::cmp(&a.1.get_num_blocks(), &b.1.get_num_blocks()),
            Sorting::Dtype => Ord::cmp(
                &a.1.get_data_type().to_string(),
                &b.1.get_data_type().to_string(),
            ),
            Sorting::Compression => Ord::cmp(
                &a.1.get_compression().to_string(),
                &b.1.get_compression().to_string(),
            ),
        }
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(a, b)| a == b).count()
}

pub struct ListCommand;

impl CommandType for ListCommand {
    type Options = ListOptions;

    fn run(_opt: &Options, ls_opt: &Self::Options) -> anyhow::Result<()> {
        let n = N5Filesystem::open(&ls_opt.n5_path)?;
        let mut group_stack = vec![(
            ls_opt.group_path.clone(),
            n.list(&ls_opt.group_path)
                .with_context(|| {
                    format!(
                        "Failed to list container: {} group path: {}",
                        &ls_opt.n5_path, &ls_opt.group_path
                    )
                })?
                .into_iter(),
        )];

        let mut datasets = HashMap::new();

        while let Some((g_path, mut g_iter)) = group_stack.pop() {
            if let Some(next_item) = g_iter.next() {
                let path: String = if g_path.is_empty() {
                    next_item
                } else {
                    g_path.clone() + "/" + &next_item
                };
                group_stack.push((g_path, g_iter));
                if let Ok(ds_attr) = n.get_dataset_attributes(&path) {
                    datasets.insert(path, ds_attr);
                } else {
                    let next_g_iter = n.list(&path)?.into_iter();
                    group_stack.push((path, next_g_iter));
                }
            }
        }

        let mut table = Table::new();
        table.set_format(*prettytable::format::consts::FORMAT_CLEAN);
        table.set_titles(row![
            "Path",
            r -> "Dims",
            r -> "Max vox",
            r -> "Block",
            r -> "Grid dims",
            r -> "Max blocks",
            "Dtype",
            "Compression",
        ]);

        let mut last_path: Option<String> = None;
        let mut format_path = String::new();
        for (path, attr) in datasets
            .into_iter()
            .sorted_by(|a, b| ls_opt.sorting.sort(a, b))
        {
            let numel = attr.get_num_elements();
            let (numel, prefix) = MetricPrefix::reduce(numel);
            let numblocks = usize::try_from(attr.get_num_blocks()).unwrap();
            let (numblocks, nb_prefix) = MetricPrefix::reduce(numblocks);

            format_path.clear();
            format_path.push_str(&path);
            let start_same = last_path
                .map(|last_path| {
                    last_path
                        .split('/')
                        .zip(path.split('/'))
                        .take_while(|(last, curr)| last == curr)
                        .fold(0, |len, (p, _)| len + p.len() + 1)
                })
                .unwrap_or(0);

            format_path.insert_str(start_same, "\u{1b}[1m");
            format_path.push_str("\u{1b}[0m");
            last_path = Some(path);

            table.add_row(row![
                format_path,
                r -> format!("{:?}", attr.get_dimensions()),
                r -> format!("{} {}", numel, prefix),
                r -> format!("{:?}", attr.get_block_size()),
                r -> format!("{:?}", attr.get_grid_extent()),
                r -> format!("{} {}", numblocks, nb_prefix),
                format!("{:?}", attr.get_data_type()),
                attr.get_compression(),
            ]);
        }

        table.printstd();

        Ok(())
    }
}
