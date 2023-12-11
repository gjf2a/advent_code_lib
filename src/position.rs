use crate::make_io_error;
use enum_iterator::{all, Sequence};
use std::collections::VecDeque;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, Div};
use std::str::FromStr;
use std::{io, mem};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
pub struct Position {
    pub row: isize,
    pub col: isize,
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.col, self.row)
    }
}

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            col: self.col + rhs.col,
            row: self.row + rhs.row,
        }
    }
}

impl Sub for Position {
    type Output = Position;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            col: self.col - rhs.col,
            row: self.row - rhs.row,
        }
    }
}

impl Mul<isize> for Position {
    type Output = Position;

    fn mul(self, rhs: isize) -> Self::Output {
        Self {
            col: self.col * rhs,
            row: self.row * rhs,
        }
    }
}

impl Div<isize> for Position {
    type Output = Position;

    fn div(self, rhs: isize) -> Self::Output {
        Self {
            col: self.col / rhs,
            row: self.row / rhs
        }
    }    
}

impl AddAssign for Position {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign<isize> for Position {
    fn mul_assign(&mut self, rhs: isize) {
        *self = *self * rhs;
    }
}

impl Position {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from(pair: (isize, isize)) -> Self {
        Self {
            col: pair.0,
            row: pair.1,
        }
    }

    pub fn grab_from(nums: &mut VecDeque<isize>) -> Self {
        let col = nums.pop_front().unwrap();
        let row = nums.pop_front().unwrap();
        Self { col, row }
    }

    pub fn update(&mut self, d: Dir) {
        let (nc, nr) = d.neighbor(self.col, self.row);
        self.col = nc;
        self.row = nr;
    }

    pub fn updated(&self, d: Dir) -> Self {
        let (nc, nr) = d.neighbor(self.col, self.row);
        Self { col: nc, row: nr }
    }

    pub fn next_in_grid(&self, width: usize, height: usize) -> Option<Position> {
        let mut result = self.clone();
        result.col += 1;
        if result.col == width as isize {
            result.col = 0;
            result.row += 1;
        }
        if result.row < height as isize {
            Some(result)
        } else {
            None
        }
    }

    pub fn manhattan_distance(&self, other: Position) -> usize {
        ((self.col - other.col).abs() + (self.row - other.row).abs()) as usize
    }

    pub fn is_manhattan_neighbor_of(&self, other: Position) -> bool {
        self.manhattan_distance(other) == 1
    }

    pub fn manhattan_neighbors(&self) -> impl Iterator<Item = Position> + '_ {
        all::<ManhattanDir>().map(|dir| *self + dir.position_offset())
    }

    pub fn neighbors(&self) -> impl Iterator<Item = Position> + '_ {
        all::<Dir>().map(|dir| *self + dir.position_offset())
    }
}

impl FromStr for Position {
    type Err = io::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() >= 2 && s.contains(',') {
            let mut chars = s.chars();
            let first = chars.next().unwrap();
            let last = chars.last().unwrap();
            if first == '(' && last == ')' {
                return Position::from_str(&s[1..s.len() - 1]);
            } else {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    return match parts[0]
                        .trim()
                        .parse::<isize>()
                        .and_then(|x| parts[1].trim().parse::<isize>().map(|y| (x, y)))
                    {
                        Ok(pair) => Ok(Position::from(pair)),
                        Err(e) => make_io_error(e.to_string().as_str()),
                    };
                }
            }
        }
        make_io_error(format!("\"{}\" is not a Position.", s).as_str())
    }
}

pub struct RowMajorPositionIterator {
    width: usize,
    height: usize,
    next: Option<Position>,
}

impl RowMajorPositionIterator {
    pub fn new(width: usize, height: usize) -> Self {
        RowMajorPositionIterator {
            width,
            height,
            next: Some(Position { col: 0, row: 0 }),
        }
    }

    pub fn in_bounds(&self) -> bool {
        self.next.map_or(false, |n| {
            n.col < self.width as isize && n.row < self.height as isize
        })
    }
}

impl Iterator for RowMajorPositionIterator {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let mut future = self
            .next
            .and_then(|p| p.next_in_grid(self.width, self.height));
        mem::swap(&mut future, &mut self.next);
        future
    }
}

pub struct OffsetRowMajorPositionIterator {
    min_col: isize,
    max_col: isize,
    max_row: isize,
    next: Option<Position>,
}

impl OffsetRowMajorPositionIterator {
    pub fn new(min_col: isize, min_row: isize, max_col: isize, max_row: isize) -> Self {
        OffsetRowMajorPositionIterator {
            min_col,
            max_col,
            max_row,
            next: Some(Position {
                col: min_col,
                row: min_row,
            }),
        }
    }

    fn next_in_grid(&self, p: Position) -> Option<Position> {
        let mut updated = Position::from((p.col + 1, p.row));
        if updated.col > self.max_col {
            updated.col = self.min_col;
            updated.row += 1;
            if updated.row > self.max_row {
                return None;
            }
        }
        Some(updated)
    }
}

impl Iterator for OffsetRowMajorPositionIterator {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let mut future = self.next.and_then(|p| self.next_in_grid(p));
        mem::swap(&mut future, &mut self.next);
        future
    }
}

pub fn indices_2d_vec<T>(width: usize, height: usize, func: fn(usize, usize) -> T) -> Vec<Vec<T>> {
    (0..height)
        .map(|y| (0..width).map(|x| func(x, y)).collect())
        .collect()
}

pub trait DirType {
    fn offset(&self) -> (isize, isize);

    fn next_position(&self, p: Position) -> Position {
        p + Position::from(self.offset())
    }

    fn position_offset(&self) -> Position {
        Position::from(self.offset())
    }

    fn neighbor(&self, col: isize, row: isize) -> (isize, isize) {
        let (d_col, d_row) = self.offset();
        (col + d_col, row + d_row)
    }

    fn position_neighbor(&self, col: isize, row: isize) -> Position {
        Position::from(self.neighbor(col, row))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Sequence)]
pub enum ManhattanDir {
    N,
    E,
    S,
    W,
}

impl DirType for ManhattanDir {
    fn offset(&self) -> (isize, isize) {
        match self {
            ManhattanDir::N => (0, -1),
            ManhattanDir::E => (1, 0),
            ManhattanDir::S => (0, 1),
            ManhattanDir::W => (-1, 0),
        }
    }
}

impl ManhattanDir {
    pub fn inverse(&self) -> ManhattanDir {
        match self {
            ManhattanDir::N => ManhattanDir::S,
            ManhattanDir::S => ManhattanDir::N,
            ManhattanDir::E => ManhattanDir::W,
            ManhattanDir::W => ManhattanDir::E,
        }
    }

    pub fn clockwise(&self) -> ManhattanDir {
        match self {
            ManhattanDir::N => ManhattanDir::E,
            ManhattanDir::E => ManhattanDir::S,
            ManhattanDir::S => ManhattanDir::W,
            ManhattanDir::W => ManhattanDir::N,
        }
    }

    pub fn counterclockwise(&self) -> ManhattanDir {
        match self {
            ManhattanDir::N => ManhattanDir::W,
            ManhattanDir::W => ManhattanDir::S,
            ManhattanDir::S => ManhattanDir::E,
            ManhattanDir::E => ManhattanDir::N,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Sequence)]
pub enum Dir {
    N,
    Ne,
    E,
    Se,
    S,
    Sw,
    W,
    Nw,
}

impl DirType for Dir {
    fn offset(&self) -> (isize, isize) {
        match self {
            Dir::N => (0, -1),
            Dir::Ne => (1, -1),
            Dir::E => (1, 0),
            Dir::Se => (1, 1),
            Dir::S => (0, 1),
            Dir::Sw => (-1, 1),
            Dir::W => (-1, 0),
            Dir::Nw => (-1, -1),
        }
    }
}

impl Dir {
    pub fn right(&self) -> Dir {
        match self {
            Dir::N => Dir::Ne,
            Dir::Ne => Dir::E,
            Dir::E => Dir::Se,
            Dir::Se => Dir::S,
            Dir::S => Dir::Sw,
            Dir::Sw => Dir::W,
            Dir::W => Dir::Nw,
            Dir::Nw => Dir::N,
        }
    }

    pub fn rotated_degrees(&self, degrees: isize) -> Dir {
        let mut steps = normalize_degrees(degrees) / 45;
        let mut result = *self;
        while steps > 0 {
            steps -= 1;
            result = result.right();
        }
        result
    }
}

pub fn normalize_degrees(degrees: isize) -> isize {
    let mut degrees = degrees;
    while degrees < 0 {
        degrees += 360;
    }
    degrees % 360
}
