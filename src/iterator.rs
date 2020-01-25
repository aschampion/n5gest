use itertools::Itertools;
use n5::{
    smallvec::smallvec,
    DatasetAttributes,
    GridCoord,
};

pub(crate) trait CoordIteratorFactory {
    fn coord_iter(
        &self,
        data_attrs: &DatasetAttributes,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize);
}

pub(crate) struct DefaultCoordIter {}

impl CoordIteratorFactory for DefaultCoordIter {
    fn coord_iter(
        &self,
        data_attrs: &DatasetAttributes,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize) {
        let coord_iter = data_attrs.coord_iter();
        let total_coords = coord_iter.len();

        (Box::new(coord_iter), total_coords)
    }
}

pub(crate) struct VoxelBoundedSlabCoordIter {
    pub axis: usize,
    pub slab_coord: u64,
    pub min: GridCoord,
    pub max: GridCoord,
}

impl CoordIteratorFactory for VoxelBoundedSlabCoordIter {
    fn coord_iter(
        &self,
        data_attrs: &DatasetAttributes,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize) {
        // Necessary for moving into closures.

        let axis = self.axis;
        let slab_coord = self.slab_coord;
        let mut coord_ceil = self
            .max
            .iter()
            .zip(data_attrs.get_block_size().iter())
            .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
            .collect::<Vec<_>>();
        coord_ceil.remove(axis as usize);
        let mut coord_floor = self
            .min
            .iter()
            .zip(data_attrs.get_block_size().iter())
            .map(|(&d, &s)| d / u64::from(s))
            .collect::<Vec<_>>();
        coord_floor.remove(axis as usize);
        let total_coords = coord_floor
            .iter()
            .zip(coord_ceil.iter())
            .map(|(&min, &max)| max - min)
            .product::<u64>() as usize;

        let iter = coord_ceil
            .into_iter()
            .zip(coord_floor.into_iter())
            .map(|(c, f)| f..c)
            .multi_cartesian_product()
            .map(move |mut c| {
                c.insert(axis as usize, slab_coord);
                c
            });

        (Box::new(iter), total_coords)
    }
}

pub(crate) struct GridSlabCoordIter {
    pub axis: usize,
    pub slab_coord: u64,
}

impl CoordIteratorFactory for GridSlabCoordIter {
    fn coord_iter(
        &self,
        data_attrs: &DatasetAttributes,
    ) -> (Box<dyn Iterator<Item = Vec<u64>>>, usize) {
        VoxelBoundedSlabCoordIter {
            axis: self.axis,
            slab_coord: self.slab_coord,
            min: smallvec![0; data_attrs.get_ndim()],
            max: data_attrs.get_dimensions().into(),
        }
        .coord_iter(data_attrs)
    }
}
