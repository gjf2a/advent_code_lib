use crate::{map_width_height, to_map, Position, RowMajorPositionIterator};
use bare_metal_modulo::*;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Debug, Display}, str::FromStr,
};

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

impl FromStr for GridCharWorld {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut map = HashMap::new();
        for (row, line) in s.lines().enumerate() {
            for (col, value) in line.char_indices() {
                map.insert(Position::from((col as isize, row as isize)), value);
            }
        }
        let (width, height) = map_width_height(&map);
        Ok(Self {map, width, height})
    }
}

impl<V: Copy + Clone + Eq + PartialEq> GridWorld<V> {
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

    pub fn with_new_row<F: Fn(Position) -> V>(&self, row_num: isize, value: F) -> Self {
        let mut result = Self {
            width: self.width,
            height: self.height + 1,
            map: self.map.iter().map(|(p, v)| (if p.row < row_num {*p} else {Position {col: p.col, row: p.row + 1}}, *v)).collect()
        };
        for col in 0..result.width {
            let k = Position {row: row_num, col: col as isize};
            result.map.insert(k, value(k));
        }
        result
    }

    pub fn with_new_column<F: Fn(Position) -> V>(&self, col_num: isize, value: F) -> Self {
        let mut result = Self {
            width: self.width + 1,
            height: self.height,
            map: self.map.iter().map(|(p, v)| (if p.col < col_num {*p} else {Position {col: p.col + 1, row: p.row}}, *v)).collect()
        };
        for row in 0..result.height {
            let k = Position {row: row as isize, col: col_num};
            result.map.insert(k, value(k));
        }
        result
    }

    pub fn value(&self, p: Position) -> Option<V> {
        self.map.get(&p).copied()
    }

    pub fn get(&self, col: usize, row: usize) -> Option<V> {
        self.value(Position {row: row as isize, col: col as isize})
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
        self.position_iter()
            .filter(|p| self.value(*p).unwrap() == item)
            .collect()
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
pub struct InfiniteGrid<V: Copy + Clone + Debug + Default + Display> {
    map: BTreeMap<Position, V>,
}

impl<V: Copy + Clone + Debug + Default + Display> Display for InfiniteGrid<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ((x_start, y_start), (x_end, y_end)) = self.bounding_box();
        for y in y_start..=y_end {
            for x in x_start..=x_end {
                write!(f, "{}", self.get(x, y))?;
            }
            write!(f, "\n")?
        }
        Ok(())
    }
}

impl<V: Copy + Clone + Debug + Default + Display> InfiniteGrid<V> {
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
        self.add_pos(Position { row: y, col: x }, value)
    }

    pub fn move_square(&mut self, start: (isize, isize), movement: (isize, isize)) {
        let start = Position {
            col: start.0,
            row: start.1,
        };
        let offset = Position {
            col: movement.0,
            row: movement.1,
        };
        let value = self.map.remove(&start).unwrap_or_default();
        self.add_pos(start + offset, value);
    }

    pub fn bounding_box(&self) -> ((isize, isize), (isize, isize)) {
        ((self.min_x(), self.min_y()), (self.max_x(), self.max_y()))
    }

    pub fn min_x(&self) -> isize {
        self.map.keys().map(|k| k.col).min().unwrap()
    }

    pub fn max_x(&self) -> isize {
        self.map.keys().map(|k| k.col).max().unwrap()
    }

    pub fn min_y(&self) -> isize {
        self.map.keys().map(|k| k.row).min().unwrap()
    }

    pub fn max_y(&self) -> isize {
        self.map.keys().map(|k| k.row).max().unwrap()
    }
}
