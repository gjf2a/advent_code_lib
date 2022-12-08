use crate::{map_width_height, nums2map, Position, RowMajorPositionIterator};
use bare_metal_modulo::*;
use std::{collections::HashMap, fmt::Display};

pub struct GridWorld {
    map: HashMap<Position, ModNumC<u32, 10>>,
    width: usize,
    height: usize,
}

impl GridWorld {
    pub fn from_file(filename: &str) -> anyhow::Result<Self> {
        let map = nums2map(filename)?;
        let (width, height) = map_width_height(&map);
        Ok(Self { map, width, height })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn value(&self, p: Position) -> Option<ModNumC<u32, 10>> {
        self.map.get(&p).copied()
    }

    pub fn position_iter(&self) -> RowMajorPositionIterator {
        RowMajorPositionIterator::new(self.width, self.height)
    }
}

impl Display for GridWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for p in self.position_iter() {
            if p.row > 0 && p.col == 0 {
                write!(f, "\n")?;
            }
            write!(f, "{}", self.value(p).unwrap().a())?;
        }
        Ok(())
    }
}
