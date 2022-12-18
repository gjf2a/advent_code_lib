use std::{
    cmp::{max, min},
    fmt::Display,
    ops::{Add, Index, Neg, Sub},
    str::FromStr,
};

use bare_metal_modulo::NumType;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Point<N: NumType, const S: usize> {
    coords: [N; S],
}

impl<N: NumType, const S: usize> Index<usize> for Point<N, S> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl<N: NumType, const S: usize> Default for Point<N, S> {
    fn default() -> Self {
        Self {
            coords: [N::default(); S],
        }
    }
}

impl<N: NumType, const S: usize> Point<N, S> {
    pub fn new(coords: [N; S]) -> Self {
        Self { coords }
    }

    pub fn from_iter<I: Iterator<Item = N>>(items: I) -> Self {
        let mut result = Self::default();
        for (i, item) in items.enumerate() {
            result.coords[i] = item;
        }
        result
    }

    pub fn bounding_box<I: Iterator<Item = Point<N, S>>>(
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

    pub fn x(&self) -> N {
        self.coords[0]
    }

    pub fn y(&self) -> N {
        self.coords[1]
    }

    pub fn z(&self) -> N {
        self.coords[2]
    }

    pub fn adjacent(&self, other: &Point<N, S>) -> bool {
        let all_but_1 = (0..S)
            .filter(|i| self.coords[*i] == other.coords[*i])
            .count()
            == S - 1;
        let touch = (0..S)
            .filter(|i| {
                self.coords[*i] == other.coords[*i] + N::one()
                    || self.coords[*i] + N::one() == other.coords[*i]
            })
            .count()
            == 1;
        all_but_1 && touch
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
