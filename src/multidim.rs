use std::{
    cmp::{max, min},
    fmt::Display,
    iter::Sum,
    ops::{Add, Index, Neg, Sub, IndexMut},
    str::FromStr,
};

use bare_metal_modulo::NumType;

use crate::{ManhattanDir, Dir};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Point<N: NumType, const S: usize> {
    coords: [N; S],
}

impl Point<isize,2> {
    pub fn manhattan_move(&mut self, dir: ManhattanDir) {
        match dir {
            ManhattanDir::N => self[1] -= 1,
            ManhattanDir::E => self[0] += 1,
            ManhattanDir::S => self[1] += 1,
            ManhattanDir::W => self[0] -= 1,
        }
    }

    pub fn manhattan_moved(&self, dir: ManhattanDir) -> Self {
        let mut result = *self;
        result.manhattan_move(dir);
        result
    }

    pub fn dir_moved(&self, dir: Dir) -> Self {
        *self + Self::new(match dir {
            Dir::N => [0, -1],
            Dir::Ne => [1, -1],
            Dir::E => [1, 0],
            Dir::Se => [1, 1],
            Dir::S => [0, 1],
            Dir::Sw => [-1, 1],
            Dir::W => [-1, 0],
            Dir::Nw => [-1, -1],
        })
    }
}

impl<N: NumType, const S: usize> Point<N, S> {
    pub fn new(coords: [N; S]) -> Self {
        Self { coords }
    }

    pub fn values(&self) -> impl Iterator<Item = N> + '_ {
        self.coords.iter().copied()
    }

    pub fn from_iter<I: Iterator<Item = N>>(items: I) -> Self {
        let mut result = Self::default();
        for (i, item) in items.enumerate() {
            result.coords[i] = item;
        }
        result
    }

    pub fn min_max_points<I: Iterator<Item = Point<N, S>>>(
        mut points: I,
    ) -> Option<(Point<N, S>, Point<N, S>)> {
        if let Some(init) = points.next() {
            let init = (init, init);
            Some(points.fold(init, |a, b| {
                (
                    Self::from_iter((0..S).map(|i| min(a.0[i], b[i]))),
                    Self::from_iter((0..S).map(|i| max(a.1[i], b[i]))),
                )
            }))
        } else {
            None
        }
    }

    pub fn bounding_box<I: Iterator<Item = Point<N, S>>>(points: I) -> Option<Vec<Point<N, S>>> {
        Self::min_max_points(points).map(|(ul, lr)| {
            let mut result = vec![];
            Self::bb_help(&mut result, [N::default(); S], &ul, &lr, 0);
            result
        })
    }

    fn bb_help(
        result: &mut Vec<Point<N, S>>,
        coords: [N; S],
        a: &Point<N, S>,
        b: &Point<N, S>,
        start: usize,
    ) {
        if start == coords.len() {
            result.push(Point::new(coords));
        } else {
            for use_a in [true, false] {
                let mut copied = coords;
                copied[start] = (if use_a { a } else { b })[start];
                Self::bb_help(result, copied, a, b, start + 1);
            }
        }
    }
}

impl<N: NumType + num::traits::Signed + Sum<N>, const S: usize> Point<N, S> {
    pub fn manhattan_distance(&self, other: &Point<N, S>) -> N {
        (0..S).map(|i| (self[i] - other[i]).abs()).sum()
    }

    pub fn manhattan_neighbors(&self) -> Vec<Point<N, S>> {
        let mut result = vec![];
        for sign in [N::one(), -N::one()] {
            for pos in 0..S {
                let mut n = self.clone();
                n.coords[pos] += sign;
                result.push(n);
            }
        }
        result
    }

    pub fn adjacent(&self, other: &Point<N, S>) -> bool {
        self.manhattan_distance(other) == N::one()
    }
}

impl<N: NumType, const S: usize> Index<usize> for Point<N, S> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl<N: NumType, const S: usize> IndexMut<usize> for Point<N, S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.coords[index]
    }
}

impl<N: NumType, const S: usize> Default for Point<N, S> {
    fn default() -> Self {
        Self {
            coords: [N::default(); S],
        }
    }
}

impl<N: NumType, const S: usize> Add for Point<N, S> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = Self::default();
        for i in 0..S {
            result.coords[i] = self.coords[i] + rhs.coords[i];
        }
        result
    }
}

impl<N: NumType + Neg<Output = N>, const S: usize> Neg for Point<N, S> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = Self::default();
        for i in 0..S {
            result.coords[i] = -self.coords[i];
        }
        result
    }
}

impl<N: NumType + Neg<Output = N>, const S: usize> Sub for Point<N, S> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

impl<N: NumType, const S: usize> Display for Point<N, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.coords.iter().enumerate() {
            write!(f, "{}", c)?;
            if i < self.coords.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, ")")
    }
}

impl<N: NumType + FromStr, const S: usize> FromStr for Point<N, S>
where
    <N as FromStr>::Err: 'static + Sync + Send + std::error::Error,
{
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut coords = [N::default(); S];
        let mut s = s;
        if s.starts_with('(') && s.ends_with(')') {
            s = &s[1..s.len() - 1];
        }
        for (i, coord) in s.split(",").enumerate() {
            coords[i] = coord.trim().parse()?;
        }
        Ok(Self { coords })
    }
}
