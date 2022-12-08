mod searchers;
mod position;
mod failed_a_star;
mod grid;

use std::slice::Iter;
use std::{io, fs, env};
use std::io::{BufRead, Lines, BufReader};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::str::FromStr;
use bare_metal_modulo::ModNumC;
use enum_iterator::IntoEnumIterator;

pub use crate::searchers::*;
pub use crate::position::*;
pub use crate::grid::*;

pub fn advent_main(other_args: &[&str], optional_args: &[&str],
                   code: fn(Vec<String>) -> io::Result<()>) -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 + other_args.len() {
        println!("Usage: {} filename {} [{}]", args[0], other_args.join(" "), optional_args.join(" "));
        Ok(())
    } else {
        code(args)
    }
}

pub fn simpler_main(code: fn(&str) -> anyhow::Result<()>) -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} filename", args[0]);
        Ok(())
    } else {
        code(args[1].as_str())
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

pub fn nums2map(filename: &str) -> io::Result<HashMap<Position, ModNumC<u32, 10> >> {
    let mut num_map = HashMap::new();
    for (row, line) in all_lines(filename)?.enumerate() {
        for (col, value) in line.chars().enumerate() {
            num_map.insert(Position::from((col as isize, row as isize)),
                           ModNumC::new(value.to_digit(10).unwrap()));
        }
    }
    Ok(num_map)
}

pub fn map_width_height<V>(map: &HashMap<Position,V>) -> (usize, usize) {
    let max = map.keys().max().unwrap();
    let min = map.keys().min().unwrap();
    ((max.col - min.col + 1) as usize, (max.row - min.row + 1) as usize)
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

pub fn make_inner_io_error(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub fn make_io_error<T>(message: &str) -> io::Result<T> {
    Err(make_inner_io_error(message))
}

pub fn assert_io_error(condition: bool, message: &str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        make_io_error(message)
    }
}

pub fn assert_token<T: Copy + Eq + Display>(next_token: Option<T>, target: T) -> io::Result<()> {
    if let Some(token) = next_token {
        assert_io_error(token == target,
                        format!("Token '{}' did not match expected '{}'", token, target).as_str())
    } else {
        make_io_error(format!("Looking for {}, but no tokens left.", target).as_str())
    }
}

#[derive(Debug, Clone)]
pub struct SingleListNode<T: Clone> {
    value: T,
    addr: usize,
    next: Option<usize>
}

impl <T: Clone> SingleListNode<T> {
    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn iter<'a>(&self, arena: &'a Arena<T>) -> SingleListIterator<'a, T> {
        SingleListIterator {pointer: Some(self.addr), arena}
    }
}

#[derive(Debug, Clone)]
pub struct Arena<T: Clone> {
    memory: Vec<SingleListNode<T>>
}

impl <T: Clone> Arena<T> {
    pub fn new() -> Self {
        Arena {memory: Vec::new()}
    }

    pub fn get(&self, location: usize) -> &SingleListNode<T> {
        &self.memory[location]
    }

    pub fn iter_from(&self, location: usize) -> SingleListIterator<T> {
        self.get(location).iter(&self)
    }

    pub fn alloc(&mut self, value: T, next: Option<usize>) -> usize {
        let addr = self.memory.len();
        self.memory.push(SingleListNode {value, next, addr} );
        addr
    }
}

pub struct SingleListIterator<'a, T: Clone> {
    pointer: Option<usize>,
    arena: &'a Arena<T>
}

impl <'a, T: Clone> Iterator for SingleListIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.pointer {
            None => None,
            Some(current) => {
                self.pointer = self.arena.get(current).next;
                Some(&self.arena.get(current).value)
            }
        }
    }
}

pub fn combinations_of<F:FnMut(&[usize; O]), const O: usize>(inner: usize, func: &mut F) {
    let values = [0; O];
    combo_help(O - 1, inner, &values, func);
}

fn combo_help<F:FnMut(&[usize; O]), const O: usize>(outer: usize, inner: usize, partial: &[usize; O], func: &mut F) {
    for j in 0..inner {
        let mut partial = partial.clone();
        partial[outer] = j;
        if outer == 0 {
            func(&partial)
        } else {
            combo_help(outer - 1, inner, &partial, func);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;
    use std::iter::FromIterator;
    use hash_histogram::HashHistogram;
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
    fn test_offset_row_major_position_iterator() {
        let iter = OffsetRowMajorPositionIterator::new(-1, -2, 1, -1);
        let expected = [(-1, -2), (0, -2), (1, -2), (-1, -1), (0, -1), (1, -1)];
        for (p, (expect_col, expect_row)) in iter.zip(expected.iter()) {
            let expect = Position::from((*expect_col, *expect_row));
            assert_eq!(expect, p);
        }
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
            breadth_first_search(&start_value, |p, q| {
                for n in p.manhattan_neighbors()
                    .filter(|n| n.manhattan_distance(start_value) <= max_dist) {
                    q.enqueue(&n);
                }
                ContinueSearch::Yes
            });
        println!("Search complete.");
        assert_eq!(paths_back.len(), 13);
        for node in paths_back.keys() {
            let len = paths_back.path_back_from(node).len();
            println!("From {:?}: {}", node, len);
            assert!(len - 1 <= max_dist);
        }
    }

    #[test]
    fn graph_test() {
        let mut graph = AdjacencySets::new();
        for (a, b) in [("start", "A"), ("start", "b"), ("A", "c"), ("A", "b"), ("b", "d"), ("A", "end"), ("b", "end")] {
            graph.connect2(a, b);
        }
        let keys = graph.keys().collect::<Vec<_>>();
        assert_eq!(keys, vec!["A", "b", "c", "d", "end", "start"]);
        let parent_map =
            breadth_first_search(&"start".to_string(),
                                 |node, q| {
                                     graph.neighbors_of(node).unwrap().iter().for_each(|n| q.enqueue(n));
                                     ContinueSearch::Yes});
        let parent_map_str = format!("{:?}", parent_map);
        assert_eq!(parent_map_str.as_str(), r#"ParentMap { parents: {"start": None, "A": Some("start"), "b": Some("start"), "c": Some("A"), "end": Some("A"), "d": Some("b")}, last_dequeued: Some("d") }"#);
        let path = parent_map.path_back_from(&"end".to_string());
        let path_str = format!("{:?}", path);
        assert_eq!(path_str, r#"["start", "A", "end"]"#);
    }

    #[test]
    fn test_linked_list() {
        let mut arena = Arena::new();
        let mut next = None;
        let nums = [1, 2, 3, 4, 5];
        for i in nums.iter() {
            next = Some(arena.alloc(i, next));
        }
        let start = next.unwrap();
        for (retrieved, original) in arena.get(start).iter(&arena).zip(nums.iter().rev()) {
            assert_eq!(**retrieved, *original);
        }
    }

    #[test]
    fn test_num_map() {
        let map = nums2map("num_grid.txt").unwrap();
        for (p, value) in [((0, 0), 1), ((9, 0), 2), ((2, 1), 8), ((7, 8), 5), ((8, 7), 3)] {
            let p = Position::from(p);
            assert_eq!(*map.get(&p).unwrap(), value);
        }
    }

    #[test]
    fn test_combinations() {
        let mut roll_counts = HashHistogram::new();
        combinations_of(6, &mut |arr: &[usize; 3]| {
            roll_counts.bump(&arr.iter().map(|n| *n + 1).sum::<usize>());
        });
        for (num, count) in [(12, 25), (13, 21), (18, 1), (5, 6), (10, 27), (6, 10), (7, 15),
            (8, 21), (16, 6), (11, 27), (3, 1), (17, 3), (14, 15), (4, 3), (15, 10), (9, 25)] {
            assert_eq!(roll_counts.count(&num), count);
        }
    }
}
