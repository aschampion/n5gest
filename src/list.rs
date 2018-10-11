use super::*;


#[derive(StructOpt, Debug)]
pub struct ListOptions {
    /// N5 root path
    #[structopt(name = "N5")]
    n5_path: String,
}

pub struct ListCommand;

impl CommandType for ListCommand {
    type Options = ListOptions;

    fn run(_opt: &Options, ls_opt: &Self::Options) -> Result<()> {
        let n = N5Filesystem::open(&ls_opt.n5_path).unwrap();
        let mut group_stack = vec![("".to_owned(), n.list("").unwrap().into_iter())];

        let mut datasets = vec![];

        while let Some((g_path, mut g_iter)) = group_stack.pop() {
            if let Some(next_item) = g_iter.next() {
                let path: String = if g_path.is_empty() {
                    next_item
                } else {
                    g_path.clone() + "/" + &next_item
                };
                group_stack.push((g_path, g_iter));
                if let Ok(ds_attr) = n.get_dataset_attributes(&path) {
                    datasets.push((path, ds_attr));
                } else {
                    let next_g_iter = n.list(&path).unwrap().into_iter();
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
            "Dtype",
            "Compression",
        ]);

        for (path, attr) in datasets {
            let numel = attr.get_dimensions().iter().map(|&n| n as usize).product();
            let (numel, prefix) = MetricPrefix::reduce(numel);
            table.add_row(row![
                b -> path,
                r -> format!("{:?}", attr.get_dimensions()),
                r -> format!("{} {}", numel, prefix),
                r -> format!("{:?}", attr.get_block_size()),
                format!("{:?}", attr.get_data_type()),
                attr.get_compression(),
            ]);
        }

        table.printstd();

        Ok(())
    }
}
