use std::{collections::HashMap, fmt::Display};
use crate::{Position, nums2map, map_width_height, RowMajorPositionIterator};
use bare_metal_modulo::*;

pub struct GridWorld {
    map: HashMap<Position, ModNumC<u32, 10>>,
    width: usize,
    height: usize,
}

impl GridWorld {
    pub fn from_file(filename: &str) -> anyhow::Result<Self> {
        let map = nums2map(filename)?;
        let (width, height) = map_width_height(&map);
        Ok(Self {map, width, height})
    }

    pub fn width(&self) -> usize {self.width}

    pub fn height(&self) -> usize {self.height}

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

#[cfg(test)]
mod tests {
    use crate::GridWorld;

    #[test]
    pub fn test_read() {
        const FILENAME: &str = "num_grid.txt";
        let grid = GridWorld::from_file(FILENAME).unwrap();
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 10);
        let grid_str = std::fs::read_to_string(FILENAME).unwrap();
        assert_eq!(grid_str, format!("{}", grid));
    }
}