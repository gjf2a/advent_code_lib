use crate::{map_width_height, Position, RowMajorPositionIterator, to_map};
use bare_metal_modulo::*;
use std::{collections::{HashMap, BTreeSet, BTreeMap}, fmt::{Debug, Display}};

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

#[derive(Debug, Clone, Eq, PartialEq)]
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

impl <V: Copy + Clone + Eq + PartialEq> GridWorld<V> {
    pub fn from_file<F: Fn(char) -> V>(filename: &str, reader: F) -> anyhow::Result<Self> {
        let map = to_map(filename, reader)?;
        let (width, height) = map_width_height(&map);
        Ok(Self { map, width, height })
    }

    pub fn in_bounds(&self, p: Position) -> bool {
        self.map.contains_key(&p)
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

    pub fn modify<M: FnMut(&mut V)>(&mut self, p: Position, mut modifier: M) {
        self.map.get_mut(&p).map(|v| modifier(v));
    }

    pub fn position_iter(&self) -> RowMajorPositionIterator {
        RowMajorPositionIterator::new(self.width, self.height)
    }

    pub fn position_value_iter(&self) -> impl Iterator<Item = (&Position, &V)> {
        self.map.iter()
    }

    pub fn position_value_iter_mut(&mut self) -> impl Iterator<Item = (&Position, &mut V)> {
        self.map.iter_mut()
    }

    pub fn positions_for(&self, item: V) -> BTreeSet<Position> {
        self.position_iter().filter(|p| self.value(*p).unwrap() == item).collect()
    }

    pub fn any_position_for(&self, item: V) -> Position {
        self.positions_for(item).iter().next().copied().unwrap()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

impl<V: CharDisplay + Copy + Eq + PartialEq> Display for GridWorld<V> {
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

#[derive(Clone, Debug, Default)]
pub struct InfiniteGrid<V: Copy + Clone + Debug + Default> {
    map: BTreeMap<Position, V>
}

impl<V: Copy + Clone + Debug + Default> InfiniteGrid<V> {
    pub fn get_pos(&self, p: Position) -> V {
        self.map.get(&p).copied().unwrap_or_default()
    }

    pub fn add_pos(&mut self, p: Position, value: V) {
        self.map.insert(p, value);
    }

    pub fn get(&self, x: isize, y: isize) -> V {
        self.get_pos(Position { row: y, col: x })
    }

    pub fn add(&mut self, x: isize, y: isize, value: V) {
        self.add_pos(Position {row: y, col: x }, value)
    }
}