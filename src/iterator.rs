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
    ) -> Box<dyn ExactSizeIterator<Item = Vec<u64>>>;
}

pub(crate) struct DefaultCoordIter {}

impl CoordIteratorFactory for DefaultCoordIter {
    fn coord_iter(
        &self,
        data_attrs: &DatasetAttributes,
    ) -> Box<dyn ExactSizeIterator<Item = Vec<u64>>> {
        let coord_iter = data_attrs.coord_iter();

        Box::new(coord_iter)
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
    ) -> Box<dyn ExactSizeIterator<Item = Vec<u64>>> {
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

        let iter = CoordRangeIterator::new(
            coord_ceil
                .into_iter()
                .zip(coord_floor.into_iter())
                .map(|(c, f)| f..c),
        )
        .map(move |mut c| {
            c.insert(axis as usize, slab_coord);
            c
        });

        Box::new(iter)
    }
}

/// A wrapper around `itertool`'s `MultiProduct` for a coordinate iterator
/// implementing `ExactSizeIterator`.
pub(crate) struct CoordRangeIterator {
    len: usize,
    mcp: itertools::structs::MultiProduct<std::ops::Range<u64>>,
}

impl CoordRangeIterator {
    /// Construct from an iterator over ranges for each axis.
    pub fn new(iter: impl Iterator<Item = std::ops::Range<u64>> + Clone) -> Self {
        Self {
            len: iter.clone().map(|r| r.end - r.start).product::<u64>() as usize,
            mcp: iter.multi_cartesian_product(),
        }
    }
}

impl Iterator for CoordRangeIterator {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        self.mcp.next()
    }
}

impl ExactSizeIterator for CoordRangeIterator {
    fn len(&self) -> usize {
        self.len
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
    ) -> Box<dyn ExactSizeIterator<Item = Vec<u64>>> {
        VoxelBoundedSlabCoordIter {
            axis: self.axis,
            slab_coord: self.slab_coord,
            min: smallvec![0; data_attrs.get_ndim()],
            max: data_attrs.get_dimensions().into(),
        }
        .coord_iter(data_attrs)
    }
}
