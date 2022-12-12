use crate::{map_width_height, Position, RowMajorPositionIterator, to_map};
use bare_metal_modulo::*;
use std::{collections::HashMap, fmt::Display};

pub type GridDigitWorld = GridWorld<ModNumC<u8, 10>>;
pub type GridCharWorld = GridWorld<char>;

pub trait CharDisplay {
    fn display(&self) -> char;
}

impl CharDisplay for ModNumC<u8, 10> {
    fn display(&self) -> char {
        (self.a() + '0' as u8) as char
    }
}

impl CharDisplay for char {
    fn display(&self) -> char {
        *self
    }
}

pub struct GridWorld<V> {
    map: HashMap<Position, V>,
    width: usize,
    height: usize,
}

impl GridDigitWorld {
    pub fn from_digit_file(filename: &str) -> anyhow::Result<GridDigitWorld> {
        Self::from_file(filename, |c| ModNumC::new(c.to_digit(10).unwrap() as u8))
    }
}

impl GridCharWorld {
    pub fn from_char_file(filename: &str) -> anyhow::Result<GridCharWorld> {
        Self::from_file(filename, |c| c)
    }
}

impl <V: Copy + Clone> GridWorld<V> {
    pub fn from_file<F: Fn(char) -> V>(filename: &str, reader: F) -> anyhow::Result<Self> {
        let map = to_map(filename, reader)?;
        let (width, height) = map_width_height(&map);
        Ok(Self { map, width, height })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn value(&self, p: Position) -> Option<V> {
        self.map.get(&p).copied()
    }

    pub fn position_iter(&self) -> RowMajorPositionIterator {
        RowMajorPositionIterator::new(self.width, self.height)
    }
}

impl<V: CharDisplay + Copy> Display for GridWorld<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for p in self.position_iter() {
            if p.row > 0 && p.col == 0 {
                write!(f, "\n")?;
            }
            write!(f, "{}", self.value(p).unwrap().display())?;
        }
        Ok(())
    }
}