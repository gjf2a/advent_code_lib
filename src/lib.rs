use std::slice::Iter;
use std::{io, fs, mem};
use std::io::{BufRead, Lines, BufReader};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Add, Mul, AddAssign, MulAssign};
use std::fs::File;
use enum_iterator::IntoEnumIterator;

pub fn all_lines_wrap(filename: &str) -> io::Result<Lines<BufReader<File>>> {
    Ok(io::BufReader::new(fs::File::open(filename)?).lines())
}

pub fn all_lines(filename: &str) -> io::Result<impl Iterator<Item=String>> {
    Ok(all_lines_wrap(filename)?.map(|line| line.unwrap()))
}

pub fn for_each_line<F: FnMut(&str) -> io::Result<()>>(filename: &str, mut line_processor: F) -> io::Result<()> {
    for line in all_lines(filename)? {
        line_processor(line.as_str())?;
    }
    Ok(())
}

pub fn file2nums(filename: &str) -> io::Result<Vec<isize>> {
    Ok(all_lines(filename)?.map(|line| line.parse::<isize>().unwrap()).collect())
}

pub fn pass_counter<F: Fn(&str) -> bool>(filename: &str, passes_check: F) -> io::Result<String> {
    Ok(all_lines(filename)?
        .filter(|line| passes_check(line.as_str()))
        .count().to_string())
}

pub trait ExNihilo {
    fn create() -> Self;
}

impl <K:Ord,V> ExNihilo for BTreeMap<K,V> {
    fn create() -> Self {BTreeMap::new()}
}

impl <T:Ord> ExNihilo for BTreeSet<T> {
    fn create() -> Self {BTreeSet::new()}
}

pub struct MultiLineObjects<T: Eq+PartialEq+Clone+ExNihilo> {
    objects: Vec<T>
}

impl <T: Eq+PartialEq+Clone+ExNihilo> MultiLineObjects<T> {
    pub fn new() -> Self {
        MultiLineObjects {objects: vec![T::create()]}
    }

    pub fn from_file<P: FnMut(&mut T,&str)>(filename: &str, proc: &mut P) -> io::Result<Self> {
        let mut result = MultiLineObjects::new();
        for_each_line(filename, |line| Ok({
            result.add_line(line, proc);
        }))?;
        Ok(result)
    }

    pub fn add_line<P: FnMut(&mut T,&str)>(&mut self, line: &str, proc: &mut P) {
        let line = line.trim();
        if line.len() == 0 {
            self.objects.push(T::create());
        } else {
            let end = self.objects.len() - 1;
            proc(&mut self.objects[end], line);
        }
    }

    pub fn objects(&self) -> Vec<T> {self.objects.clone()}

    pub fn iter(&self) -> Iter<T> {
        self.objects.iter()
    }

    pub fn count_matching<P: Fn(&T) -> bool>(&self, predicate: P) -> usize {
        self.iter()
            .filter(|m| predicate(*m))
            .count()
    }
}

#[derive(Debug,Copy,Clone,Eq,PartialEq)]
pub struct Position {
    pub col: isize,
    pub row: isize
}

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Position {col: self.col + rhs.col, row: self.row + rhs.row}
    }
}

impl Mul<isize> for Position {
    type Output = Position;

    fn mul(self, rhs: isize) -> Self::Output {
        Position {col: self.col * rhs, row: self.row * rhs}
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
    pub fn from(pair: (isize,isize)) -> Self {
        Position {col: pair.0, row: pair.1}
    }

    pub fn update(&mut self, d: Dir) {
        let (nc, nr) = d.neighbor(self.col, self.row);
        self.col = nc;
        self.row = nr;
    }

    pub fn updated(&self, d: Dir) -> Self {
        let (nc, nr) = d.neighbor(self.col, self.row);
        Position {col: nc, row: nr}
    }

    pub fn next_in_grid(&self, width: usize, height: usize) -> Option<Position> {
        let mut result = self.clone();
        result.col += 1;
        if result.col == width as isize {
            result.col = 0;
            result.row += 1;
        }
        if result.row < height as isize {Some(result)} else {None}
    }
}

pub struct RowMajorPositionIterator {
    width: usize, height: usize, next: Option<Position>
}

impl RowMajorPositionIterator {
    pub fn new(width: usize, height: usize) -> Self {
        RowMajorPositionIterator {width, height, next: Some(Position {col: 0, row: 0})}
    }

    pub fn in_bounds(&self) -> bool {
        self.next.map_or(false, |n| n.col < self.width as isize && n.row < self.height as isize)
    }
}

impl Iterator for RowMajorPositionIterator {
    type Item = Position;

    fn next(&mut self) -> Option<Self::Item> {
        let mut future = self.next.and_then(|p| p.next_in_grid(self.width, self.height));
        mem::swap(&mut future, &mut self.next);
        future
    }
}

pub fn indices_2d_vec<T>(width: usize, height: usize, func: fn(usize,usize)->T) -> Vec<Vec<T>> {
    (0..height)
        .map(|y| (0..width)
            .map(|x| func(x, y))
            .collect())
        .collect()
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd,IntoEnumIterator)]
pub enum ManhattanDir {
    N, E, S, W
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd,IntoEnumIterator)]
pub enum Dir {
    N, Ne, E, Se, S, Sw, W, Nw
}

impl Dir {
    pub fn neighbor(&self, col: isize, row: isize) -> (isize,isize) {
        let (d_col, d_row) = self.offset();
        (col + d_col, row + d_row)
    }

    pub fn right(&self) -> Dir {
        match self {
            Dir::N => Dir::Ne,
            Dir::Ne => Dir::E,
            Dir::E => Dir::Se,
            Dir::Se => Dir::S,
            Dir::S => Dir::Sw,
            Dir::Sw => Dir::W,
            Dir::W => Dir::Nw,
            Dir::Nw => Dir::N
        }
    }

    pub fn offset(&self) -> (isize,isize) {
        match self {
            Dir::N  => ( 0, -1),
            Dir::Ne => ( 1, -1),
            Dir::E  => ( 1,  0),
            Dir::Se => ( 1,  1),
            Dir::S  => ( 0,  1),
            Dir::Sw => (-1,  1),
            Dir::W  => (-1,  0),
            Dir::Nw => (-1, -1)
        }
    }

    pub fn position_offset(&self) -> Position {
        Position::from(self.offset())
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
    while degrees < 0 {degrees += 360;}
    degrees % 360
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;
    use Dir::*;

    #[test]
    fn test_nums() {
        let nums = file2nums("test_1.txt").unwrap();
        assert_eq!(*nums.iter().min().unwrap(), 111);
        assert_eq!(*nums.iter().max().unwrap(), 2010);
    }

    #[test]
    fn test_pred_count() {
        let min_1500 = pass_counter("test_1.txt", |n| n.parse::<isize>().unwrap() >= 1500).unwrap();
        assert_eq!(min_1500, "185");
    }

    fn to_set(strs: Vec<&str>) -> BTreeSet<String> {
        BTreeSet::from_iter(strs.iter().map(|s| s.to_string()))
    }

    #[test]
    fn test_multiline() {
        let objs: MultiLineObjects<BTreeSet<String>> = MultiLineObjects::from_file(
            "test_2.txt",
            &mut |set: &mut BTreeSet<String>, line| {set.insert(line.to_owned());}).unwrap();
        let mut iter = objs.iter();

        let obj = iter.next().unwrap();
        assert_eq!(*obj, to_set(vec!["abc"]));

        let obj = iter.next().unwrap();
        assert_eq!(*obj, to_set(vec!["a", "b", "c"]));

        let obj = iter.next().unwrap();
        assert_eq!(*obj, to_set(vec!["ab", "ac"]));

        let obj = iter.next().unwrap();
        assert_eq!(*obj, to_set(vec!["a"]));

        let obj = iter.next().unwrap();
        assert_eq!(*obj, to_set(vec!["b"]));
    }

    #[test]
    fn test_dir() {
        assert_eq!(Dir::into_enum_iter().collect::<Vec<Dir>>(), vec![N,Ne,E,Se,S,Sw,W,Nw]);
        assert_eq!(Dir::into_enum_iter().map(|d| d.neighbor(4, 4)).collect::<Vec<(isize,isize)>>(),
                   vec![(4, 3), (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (3, 4), (3, 3)]);
        let mut p = Position {col: 3, row: 2};
        p.update(Dir::Nw);
        assert_eq!(p, Position {col: 2, row: 1});
        p.update(Dir::Se);
        assert_eq!(p, Position {col: 3, row: 2});
        assert_eq!(p.updated(Dir::Ne), Position {col: 4, row: 1});

        let ps: Vec<Position> = RowMajorPositionIterator::new(2, 3).collect();
        let targets = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)];
        assert_eq!(ps.len(), targets.len());
        assert!((0..targets.len()).all(|i| Position {col: targets[i].0, row: targets[i].1} == ps[i]));

        assert_eq!(Dir::N.rotated_degrees(90), Dir::E);
        assert_eq!(Dir::N.rotated_degrees(180), Dir::S);
        assert_eq!(Dir::N.rotated_degrees(270), Dir::W);
        assert_eq!(Dir::N.rotated_degrees(360), Dir::N);
        assert_eq!(Dir::N.rotated_degrees(-90), Dir::W);
        assert_eq!(Dir::E.rotated_degrees(180), Dir::W);
        assert_eq!(Dir::E.rotated_degrees(-180), Dir::W);
    }

    #[test]
    fn test_pos_math() {
        let mut p = Position::from((2, 3));
        assert_eq!(Position::from((4, 6)), p * 2);
        assert_eq!(Position::from((3, 5)), p + Position::from((1, 2)));

        p += Position::from((2, 5));
        assert_eq!(Position::from((4, 8)), p);

        p *= 3;
        assert_eq!(Position::from((12, 24)), p);
    }

    #[test]
    fn test_normalize_degrees() {
        assert_eq!(normalize_degrees(90), 90);
        assert_eq!(normalize_degrees(-90), 270);
        assert_eq!(normalize_degrees(180), normalize_degrees(-180));
        assert_eq!(normalize_degrees(360), 0);
    }

    #[test]
    fn test_indices_2d() {
        let v: Vec<Vec<(usize,usize)>> = indices_2d_vec(3, 2, |x, y| (x, y));
        assert_eq!(v, vec![vec![(0, 0), (1, 0), (2, 0)], vec![(0, 1), (1, 1), (2, 1)]]);
    }
}
