use std::{str::FromStr, fmt::Display};

use bare_metal_modulo::NumType;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Point<N: NumType, const S: usize> {
    pub coords: [N; S]
}

impl<N: NumType, const S: usize> Point<N, S> {
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
        let all_but_1 = (0..S).filter(|i| self.coords[*i] == other.coords[*i]).count() == S - 1;
        let touch = (0..S).filter(|i| self.coords[*i] == other.coords[*i] + N::one() || self.coords[*i] + N::one() == other.coords[*i]).count() == 1;
        all_but_1 && touch
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

impl<N: NumType + FromStr, const S: usize> FromStr for Point<N, S> where <N as FromStr>::Err: 'static + Sync + Send + std::error::Error {
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
        Ok(Self {coords})
    }
}