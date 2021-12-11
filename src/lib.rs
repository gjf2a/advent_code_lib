use std::slice::Iter;
use std::{io, fs, mem, env};
use std::io::{BufRead, Lines, BufReader};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::Debug;
use std::ops::{Add, Mul, AddAssign, MulAssign, Sub};
use std::fs::File;
use std::hash::Hash;
use std::str::FromStr;
use enum_iterator::IntoEnumIterator;
use trait_set::trait_set;

pub fn generic_main(title: &str, other_args: &[&str], optional_args: &[&str],
                    code: fn(Vec<String>) -> io::Result<()>) -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 + other_args.len() {
        println!("Usage: {} filename {} [{}]", title, other_args.join(" "), optional_args.join(" "));
        Ok(())
    } else {
        code(args)
    }
}

pub fn all_lines_wrap(filename: &str) -> io::Result<Lines<BufReader<File>>> {
    Ok(io::BufReader::new(fs::File::open(filename)?).lines())
}

pub fn all_lines(filename: &str) -> io::Result<impl Iterator<Item=String>> {
    Ok(all_lines_wrap(filename)?.map(|line| line.unwrap()))
}

pub fn first_line_only_numbers<N: FromStr>(filename: &str) -> io::Result<Vec<N>>
    where <N as FromStr>::Err: Debug {
    Ok(line2numbers(all_lines(filename)?.next().unwrap().as_str()))
}

pub fn line2numbers_iter<N: FromStr>(line: &str) -> impl Iterator<Item=N> + '_
    where <N as FromStr>::Err: Debug {
    line.split(',').map(|s| s.parse().unwrap())
}

pub fn line2numbers<N: FromStr>(line: &str) -> Vec<N> where <N as FromStr>::Err: Debug {
    line2numbers_iter(line).collect::<Vec<N>>()
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
    objects: Vec<T>,
    line_parser: fn(&mut T, &str)
}

impl <T: Eq+PartialEq+Clone+ExNihilo> MultiLineObjects<T> {
    pub fn new(line_parser: fn(&mut T, &str)) -> Self {
        MultiLineObjects {objects: vec![T::create()], line_parser}
    }

    pub fn from_file(filename: &str, line_parser: fn(&mut T, &str)) -> io::Result<Self> {
        Ok(MultiLineObjects::from_iterator(all_lines(filename)?, line_parser))
    }

    pub fn from_iterator<'a, C: IntoIterator<Item=String>>(iter: C, line_parser: fn(&mut T, &str)) -> Self {
        let mut result = MultiLineObjects::new(line_parser);
        for line in iter {
            result.add_line(line.as_str());
        }
        result
    }

    pub fn add_line(&mut self, line: &str) {
        let line = line.trim();
        if line.len() == 0 {
            self.objects.push(T::create());
        } else {
            let end = self.objects.len() - 1;
            (self.line_parser)(&mut self.objects[end], line);
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

pub fn make_io_error<T>(message: &str) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, message))
}

#[derive(Debug,Copy,Clone,Eq,PartialEq,Ord,PartialOrd,Hash)]
pub struct Position {
    pub row: isize,
    pub col: isize
}

impl Add for Position {
    type Output = Position;

    fn add(self, rhs: Self) -> Self::Output {
        Position {col: self.col + rhs.col, row: self.row + rhs.row}
    }
}

impl Sub for Position {
    type Output = Position;

    fn sub(self, rhs: Self) -> Self::Output {
        Position {col: self.col - rhs.col, row: self.row - rhs.row}
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
    pub fn new() -> Self {
        Position::from((0, 0))
    }

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

    pub fn manhattan_distance(&self, other: Position) -> usize {
        ((self.col - other.col).abs() + (self.row - other.row).abs()) as usize
    }

    pub fn is_manhattan_neighbor_of(&self, other: Position) -> bool {
        self.manhattan_distance(other) == 1
    }

    pub fn manhattan_neighbors(&self) -> impl Iterator<Item=Position> + '_ {
        ManhattanDir::into_enum_iter().map(|dir| *self + dir.position_offset())
    }

    pub fn neighbors(&self) -> impl Iterator<Item=Position> + '_ {
        Dir::into_enum_iter().map(|dir| *self + dir.position_offset())
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
                return Position::from_str(&s[1..s.len() - 1])
            } else {
                let parts: Vec<_> = s.split(',').collect();
                if parts.len() == 2 {
                    return match parts[0].trim().parse::<isize>()
                        .and_then(|x| parts[1].trim().parse::<isize>().map(|y| (x, y))) {
                        Ok(pair) => Ok(Position::from(pair)),
                        Err(e) => make_io_error(e.to_string().as_str())
                    }
                }
            }
        }
        make_io_error(format!("\"{}\" is not a Position.", s).as_str())
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

pub trait DirType {
    fn offset(&self) -> (isize,isize);

    fn next(&self, p: Position) -> Position {
        p + Position::from(self.offset())
    }

    fn position_offset(&self) -> Position {
        Position::from(self.offset())
    }

    fn neighbor(&self, col: isize, row: isize) -> (isize,isize) {
        let (d_col, d_row) = self.offset();
        (col + d_col, row + d_row)
    }

    fn position_neighbor(&self, col: isize, row: isize) -> Position {
        Position::from(self.neighbor(col, row))
    }
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd,IntoEnumIterator)]
pub enum ManhattanDir {
    N, E, S, W
}

impl DirType for ManhattanDir {
    fn offset(&self) -> (isize, isize) {
        match self {
            ManhattanDir::N => (0, -1),
            ManhattanDir::E => (1, 0),
            ManhattanDir::S => (0, 1),
            ManhattanDir::W => (-1, 0)
        }
    }
}

impl ManhattanDir {
    pub fn inverse(&self) -> ManhattanDir {
        match self {
            ManhattanDir::N => ManhattanDir::S,
            ManhattanDir::S => ManhattanDir::N,
            ManhattanDir::E => ManhattanDir::W,
            ManhattanDir::W => ManhattanDir::E
        }
    }

    pub fn clockwise(&self) -> ManhattanDir {
        match self {
            ManhattanDir::N => ManhattanDir::E,
            ManhattanDir::S => ManhattanDir::W,
            ManhattanDir::E => ManhattanDir::S,
            ManhattanDir::W => ManhattanDir::N
        }
    }
}

#[derive(Debug,Clone,Copy,Eq,PartialEq,Ord,PartialOrd,IntoEnumIterator)]
pub enum Dir {
    N, Ne, E, Se, S, Sw, W, Nw
}

impl DirType for Dir {
    fn offset(&self) -> (isize,isize) {
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
            Dir::Nw => Dir::N
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
    while degrees < 0 {degrees += 360;}
    degrees % 360
}

trait_set! {
    pub trait SearchNode = Hash + Eq + Clone;
}

pub trait SearchQueue<T> {
    fn new() -> Self;
    fn enqueue(&mut self, item: &T);
    fn dequeue(&mut self) -> Option<T>;
}

impl <T:Clone> SearchQueue<T> for VecDeque<T> {
    fn new() -> Self {VecDeque::new()}
    fn enqueue(&mut self, item: &T) {self.push_back(item.clone());}
    fn dequeue(&mut self) -> Option<T> {self.pop_front()}
}

pub struct ParentMapQueue<T: SearchNode, Q: SearchQueue<T>> {
    parent_map: HashMap<T, Option<T>>,
    queue: Q,
    last_dequeued: Option<T>
}

impl <T: SearchNode, Q: SearchQueue<T>> SearchQueue<T> for ParentMapQueue<T, Q> {
    fn new() -> Self {
        ParentMapQueue {parent_map: HashMap::new(), queue: Q::new(), last_dequeued: None}
    }

    fn enqueue(&mut self, item: &T) {
        if !self.parent_map.contains_key(item) {
            self.parent_map.insert(item.clone(), self.last_dequeued.clone());
            self.queue.enqueue(item);
        }
    }

    fn dequeue(&mut self) -> Option<T> {
        let dequeued = self.queue.dequeue();
        self.last_dequeued = dequeued.clone();
        dequeued
    }
}

pub fn search<T, S, Q>(start_value: &T, add_successors: S) -> Q
    where T: Clone, Q: SearchQueue<T>, S: Fn(&T, &mut Q) {
    let mut open_list = Q::new();
    open_list.enqueue(start_value);
    loop {
        match open_list.dequeue() {
            Some(candidate) => {
                add_successors(&candidate, &mut open_list);
            }
            None => return open_list
        }
    }
}

pub fn breadth_first_search<T,S>(start_value: &T, add_successors: S) -> HashMap<T,Option<T>>
    where T: SearchNode, S: Fn(&T, &mut ParentMapQueue<T, VecDeque<T>>) {
    search(start_value, add_successors).parent_map
}

pub fn path_back_from<T: SearchNode>(end: &T, parent_map: &HashMap<T,Option<T>>) -> VecDeque<T> {
    let mut path = VecDeque::new();
    let mut current = end;
    loop {
        path.push_front(current.clone());
        match parent_map.get(&current).and(None) {
            None => break,
            Some(parent) => {current = parent;}
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
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
    fn test_one_line_nums() {
        let nums = first_line_only_numbers::<usize>("test_3.txt").unwrap();
        assert_eq!(nums, vec![16,1,2,0,4,2,7,1,2,14]);
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
            |set: &mut BTreeSet<String>, line: &str| {set.insert(line.to_owned());}).unwrap();
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

        let q = p;
        p *= 3;
        assert_eq!(Position::from((12, 24)), p);

        assert_eq!(p - q, Position::from((8, 16)));
    }

    #[test]
    fn test_pos_order() {
        let set: BTreeSet<Position> = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)].iter().map(|p| Position::from(*p)).collect();
        let seq1: Vec<Position> = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)].iter().map(|p| Position::from(*p)).collect();
        let seq2: Vec<Position> = set.iter().copied().collect();
        assert_eq!(seq1, seq2);
    }

    #[test]
    fn test_position_from_str() {
        for (x, y) in [(22, 34), (-3, -2)].iter() {
            for s in [
                format!("{},{}", x, y),
                format!("({},{})", x, y),
                format!("{}, {}", x, y),
                format!("({}, {})", x, y)].iter() {
                println!("Testing Position from \"{}\"", s);
                assert_eq!(Position::from((*x, *y)), s.parse().unwrap());
            }
        }
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

    #[test]
    fn test_manhattan() {
        let p = Position::new();
        for (d, (x, y)) in ManhattanDir::into_enum_iter().zip([(0, -1), (1, 0), (0, 1), (-1, 0)].iter()) {
            let next = d.next(p);
            assert_eq!(next, Position::from((*x, *y)));
            let inverse = d.inverse().next(next);
            assert_eq!(inverse, p);
        }

        let mut d1 = ManhattanDir::N;
        for d2 in ManhattanDir::into_enum_iter() {
            assert_eq!(d1, d2);
            d1 = d1.clockwise();
        }
        assert_eq!(d1, ManhattanDir::N);
    }

    #[test]
    fn test_manhattan_neighbors() {
        let p = Position::new();
        let mut seen = HashSet::new();
        for n in p.manhattan_neighbors() {
            println!("{:?} ({})", n, n.manhattan_distance(p));
            assert!(n.is_manhattan_neighbor_of(p));
            seen.insert(n);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn test_diagonal_neighbors() {
        let p = Position::new();
        let mut seen = HashSet::new();
        for n in p.neighbors() {
            println!("{:?}", n);
            assert!((n.col - p.col).abs() <= 1);
            assert!((n.row - p.row).abs() <= 1);
            seen.insert(n);
        }
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn test_bfs() {
        println!("Test BFS");
        let max_dist = 2;
        let start_value = Position::new();
        println!("Starting BFS");
        let paths_back =
            breadth_first_search(&start_value, |p, q|
                for n in p.manhattan_neighbors()
                    .filter(|n| n.manhattan_distance(start_value) <= max_dist) {
                    q.enqueue(&n);
                });
        println!("Search complete.");
        assert_eq!(paths_back.len(), 13);
        for node in paths_back.keys() {
            let len = path_back_from(node, &paths_back).len();
            println!("From {:?}: {}", node, len);
            assert!(len - 1 <= max_dist);
        }
    }
}
