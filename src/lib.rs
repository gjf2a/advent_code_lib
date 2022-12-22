mod failed_a_star;
mod grid;
mod multidim;
mod position;
mod searchers;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::slice::Iter;
use std::str::FromStr;
use std::{env, fs, io};

pub use crate::grid::*;
pub use crate::multidim::*;
pub use crate::position::*;
pub use crate::searchers::*;

pub fn advent_main(
    other_args: &[&str],
    optional_args: &[&str],
    code: fn(Vec<String>) -> io::Result<()>,
) -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 + other_args.len() {
        println!(
            "Usage: {} filename {} [{}]",
            args[0],
            other_args.join(" "),
            optional_args.join(" ")
        );
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

pub fn all_lines(filename: &str) -> io::Result<impl Iterator<Item = String>> {
    Ok(all_lines_wrap(filename)?.map(|line| line.unwrap()))
}

pub fn first_line_only_numbers<N: FromStr>(filename: &str) -> io::Result<Vec<N>>
where
    <N as FromStr>::Err: Debug,
{
    Ok(line2numbers(all_lines(filename)?.next().unwrap().as_str()))
}

pub fn line2numbers_iter<N: FromStr>(line: &str) -> impl Iterator<Item = N> + '_
where
    <N as FromStr>::Err: Debug,
{
    line.split(',').map(|s| s.parse().unwrap())
}

pub fn line2numbers<N: FromStr>(line: &str) -> Vec<N>
where
    <N as FromStr>::Err: Debug,
{
    line2numbers_iter(line).collect::<Vec<N>>()
}

pub fn for_each_line<F: FnMut(&str) -> io::Result<()>>(
    filename: &str,
    mut line_processor: F,
) -> io::Result<()> {
    for line in all_lines(filename)? {
        line_processor(line.as_str())?;
    }
    Ok(())
}

pub fn file2nums(filename: &str) -> io::Result<Vec<isize>> {
    Ok(all_lines(filename)?
        .map(|line| line.parse::<isize>().unwrap())
        .collect())
}

pub fn keep_only<F: Fn(char) -> bool>(check: F, s: String) -> String {
    s.chars().map(|c| if check(c) { c } else { ' ' }).collect()
}

pub fn keep_digits(s: String) -> String {
    keep_only(|c| c.is_digit(10) || c == '-', s)
}

pub fn all_nums_from<N: FromStr>(s: String) -> VecDeque<N> {
    keep_digits(s)
        .split_whitespace()
        .map(|s| s.parse::<N>().ok().unwrap())
        .collect()
}

pub fn all_positions_from(s: String) -> VecDeque<Position> {
    let mut nums = all_nums_from(s);
    let mut result = VecDeque::new();
    assert!(nums.len() % 2 == 0);
    while nums.len() > 0 {
        result.push_back(Position::grab_from(&mut nums));
    }
    result
}

pub fn to_map<V, F: Fn(char) -> V>(
    filename: &str,
    reader: F,
) -> anyhow::Result<HashMap<Position, V>> {
    let mut result = HashMap::new();
    for (row, line) in all_lines(filename)?.enumerate() {
        for (col, value) in line.chars().enumerate() {
            result.insert(Position::from((col as isize, row as isize)), reader(value));
        }
    }
    Ok(result)
}

pub fn map_width_height<V>(map: &HashMap<Position, V>) -> (usize, usize) {
    let max = map.keys().max().unwrap();
    let min = map.keys().min().unwrap();
    (
        (max.col - min.col + 1) as usize,
        (max.row - min.row + 1) as usize,
    )
}

pub fn pass_counter<F: Fn(&str) -> bool>(filename: &str, passes_check: F) -> io::Result<String> {
    Ok(all_lines(filename)?
        .filter(|line| passes_check(line.as_str()))
        .count()
        .to_string())
}

pub trait ExNihilo {
    fn create() -> Self;
}

impl<K: Ord, V> ExNihilo for BTreeMap<K, V> {
    fn create() -> Self {
        BTreeMap::new()
    }
}

impl<T: Ord> ExNihilo for BTreeSet<T> {
    fn create() -> Self {
        BTreeSet::new()
    }
}

pub struct MultiLineObjects<T: Eq + PartialEq + Clone + ExNihilo> {
    objects: Vec<T>,
    line_parser: fn(&mut T, &str),
}

impl<T: Eq + PartialEq + Clone + ExNihilo> MultiLineObjects<T> {
    pub fn new(line_parser: fn(&mut T, &str)) -> Self {
        MultiLineObjects {
            objects: vec![T::create()],
            line_parser,
        }
    }

    pub fn from_file(filename: &str, line_parser: fn(&mut T, &str)) -> io::Result<Self> {
        Ok(MultiLineObjects::from_iterator(
            all_lines(filename)?,
            line_parser,
        ))
    }

    pub fn from_iterator<'a, C: IntoIterator<Item = String>>(
        iter: C,
        line_parser: fn(&mut T, &str),
    ) -> Self {
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

    pub fn objects(&self) -> Vec<T> {
        self.objects.clone()
    }

    pub fn iter(&self) -> Iter<T> {
        self.objects.iter()
    }

    pub fn count_matching<P: Fn(&T) -> bool>(&self, predicate: P) -> usize {
        self.iter().filter(|m| predicate(*m)).count()
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
        assert_io_error(
            token == target,
            format!("Token '{}' did not match expected '{}'", token, target).as_str(),
        )
    } else {
        make_io_error(format!("Looking for {}, but no tokens left.", target).as_str())
    }
}

#[derive(Debug, Clone)]
pub struct SingleListNode<T: Clone> {
    value: T,
    addr: usize,
    next: Option<usize>,
}

impl<T: Clone> SingleListNode<T> {
    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn iter<'a>(&self, arena: &'a Arena<T>) -> SingleListIterator<'a, T> {
        SingleListIterator {
            pointer: Some(self.addr),
            arena,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Arena<T: Clone> {
    memory: Vec<SingleListNode<T>>,
}

impl<T: Clone> Arena<T> {
    pub fn new() -> Self {
        Arena { memory: Vec::new() }
    }

    pub fn get(&self, location: usize) -> &SingleListNode<T> {
        &self.memory[location]
    }

    pub fn iter_from(&self, location: usize) -> SingleListIterator<T> {
        self.get(location).iter(&self)
    }

    pub fn alloc(&mut self, value: T, next: Option<usize>) -> usize {
        let addr = self.memory.len();
        self.memory.push(SingleListNode { value, next, addr });
        addr
    }
}

pub struct SingleListIterator<'a, T: Clone> {
    pointer: Option<usize>,
    arena: &'a Arena<T>,
}

impl<'a, T: Clone> Iterator for SingleListIterator<'a, T> {
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

pub fn combinations_of<F: FnMut(&[usize; O]), const O: usize>(inner: usize, func: &mut F) {
    let values = [0; O];
    combo_help(O - 1, inner, &values, func);
}

fn combo_help<F: FnMut(&[usize; O]), const O: usize>(
    outer: usize,
    inner: usize,
    partial: &[usize; O],
    func: &mut F,
) {
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
    use super::*;
    use bare_metal_modulo::ModNumC;
    use enum_iterator::all;
    use hash_histogram::HashHistogram;
    use std::collections::HashSet;
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
        assert_eq!(nums, vec![16, 1, 2, 0, 4, 2, 7, 1, 2, 14]);
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
        let objs: MultiLineObjects<BTreeSet<String>> =
            MultiLineObjects::from_file("test_2.txt", |set: &mut BTreeSet<String>, line: &str| {
                set.insert(line.to_owned());
            })
            .unwrap();
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
        assert_eq!(
            all::<Dir>().collect::<Vec<Dir>>(),
            vec![N, Ne, E, Se, S, Sw, W, Nw]
        );
        assert_eq!(
            all::<Dir>()
                .map(|d| d.neighbor(4, 4))
                .collect::<Vec<(isize, isize)>>(),
            vec![
                (4, 3),
                (5, 3),
                (5, 4),
                (5, 5),
                (4, 5),
                (3, 5),
                (3, 4),
                (3, 3)
            ]
        );
        let mut p = Position { col: 3, row: 2 };
        p.update(Dir::Nw);
        assert_eq!(p, Position { col: 2, row: 1 });
        p.update(Dir::Se);
        assert_eq!(p, Position { col: 3, row: 2 });
        assert_eq!(p.updated(Dir::Ne), Position { col: 4, row: 1 });

        let ps: Vec<Position> = RowMajorPositionIterator::new(2, 3).collect();
        let targets = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)];
        assert_eq!(ps.len(), targets.len());
        assert!((0..targets.len()).all(|i| Position {
            col: targets[i].0,
            row: targets[i].1
        } == ps[i]));

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
        let set: BTreeSet<Position> = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            .iter()
            .map(|p| Position::from(*p))
            .collect();
        let seq1: Vec<Position> = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
            .iter()
            .map(|p| Position::from(*p))
            .collect();
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
                format!("({}, {})", x, y),
            ]
            .iter()
            {
                println!("Testing Position from \"{}\"", s);
                assert_eq!(Position::from((*x, *y)), s.parse().unwrap());
            }
            assert_eq!(
                format!("({}, {})", x, y),
                format!("{}", Position::from((*x, *y)))
            );
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
        let v: Vec<Vec<(usize, usize)>> = indices_2d_vec(3, 2, |x, y| (x, y));
        assert_eq!(
            v,
            vec![vec![(0, 0), (1, 0), (2, 0)], vec![(0, 1), (1, 1), (2, 1)]]
        );
    }

    #[test]
    fn test_manhattan() {
        let p = Position::new();
        for (d, (x, y)) in all::<ManhattanDir>().zip([(0, -1), (1, 0), (0, 1), (-1, 0)].iter()) {
            let next = d.next_position(p);
            assert_eq!(next, Position::from((*x, *y)));
            let inverse = d.inverse().next_position(next);
            assert_eq!(inverse, p);
        }

        let mut d1 = ManhattanDir::N;
        for d2 in all::<ManhattanDir>() {
            assert_eq!(d1, d2);
            d1 = d1.clockwise();
            assert_eq!(d1.counterclockwise(), d2);
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
        let paths_back = breadth_first_search(&start_value, |p, q| {
            for n in p
                .manhattan_neighbors()
                .filter(|n| n.manhattan_distance(start_value) <= max_dist)
            {
                q.enqueue(&n);
            }
            ContinueSearch::Yes
        });
        println!("Search complete.");
        assert_eq!(paths_back.len(), 13);
        for node in paths_back.keys() {
            let len = paths_back.path_back_from(node).unwrap().len();
            println!("From {:?}: {}", node, len);
            assert!(len - 1 <= max_dist);
        }
    }

    #[test]
    fn graph_test() {
        let mut graph = AdjacencySets::new();
        for (a, b) in [
            ("start", "A"),
            ("start", "b"),
            ("A", "c"),
            ("A", "b"),
            ("b", "d"),
            ("A", "end"),
            ("b", "end"),
        ] {
            graph.connect2(a, b);
        }
        let keys = graph.keys().collect::<Vec<_>>();
        assert_eq!(keys, vec!["A", "b", "c", "d", "end", "start"]);
        let parent_map = breadth_first_search(&"start".to_string(), |node, q| {
            graph
                .neighbors_of(node)
                .unwrap()
                .iter()
                .for_each(|n| q.enqueue(n));
            ContinueSearch::Yes
        });
        let parent_map_str = format!("{:?}", parent_map);
        assert_eq!(
            parent_map_str.as_str(),
            r#"ParentMap { parents: {"start": None, "A": Some("start"), "b": Some("start"), "c": Some("A"), "end": Some("A"), "d": Some("b")}, last_dequeued: Some("d") }"#
        );
        let path = parent_map.path_back_from(&"end".to_string()).unwrap();
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
    fn test_combinations() {
        let mut roll_counts = HashHistogram::new();
        combinations_of(6, &mut |arr: &[usize; 3]| {
            roll_counts.bump(&arr.iter().map(|n| *n + 1).sum::<usize>());
        });
        for (num, count) in [
            (12, 25),
            (13, 21),
            (18, 1),
            (5, 6),
            (10, 27),
            (6, 10),
            (7, 15),
            (8, 21),
            (16, 6),
            (11, 27),
            (3, 1),
            (17, 3),
            (14, 15),
            (4, 3),
            (15, 10),
            (9, 25),
        ] {
            assert_eq!(roll_counts.count(&num), count);
        }
    }

    #[test]
    pub fn test_grid_world() {
        const FILENAME: &str = "num_grid.txt";
        let grid = GridDigitWorld::from_digit_file(FILENAME).unwrap();
        assert_eq!(grid.len(), 100);
        assert_eq!(grid.width(), 10);
        assert_eq!(grid.height(), 10);
        let grid_str = std::fs::read_to_string(FILENAME)
            .unwrap()
            .replace("\r\n", "\n");
        assert_eq!(grid_str, format!("{}", grid));

        for (p, value) in [
            ((0, 0), 1),
            ((9, 0), 2),
            ((2, 1), 8),
            ((7, 8), 5),
            ((8, 7), 3),
        ] {
            let p = Position::from(p);
            assert_eq!(grid.value(p).unwrap(), value);
            assert!(grid.positions_for(ModNumC::new(value)).contains(&p));
            assert!(grid.in_bounds(p));
        }

        for p in [(10, 8), (-1, 0), (12, 12), (8, 10)] {
            let p = Position::from(p);
            assert!(!grid.in_bounds(p));
        }

        let original = grid.clone();
        let mut grid = grid;
        for p in grid.position_iter() {
            let old = grid.value(p).unwrap();
            grid.modify(p, |v| *v += 1);
            assert_eq!(old + 1, grid.value(p).unwrap());
        }

        for (_, value) in grid.position_value_iter_mut() {
            *value -= 1;
        }

        assert_eq!(original, grid);
    }

    #[test]
    fn test_infinite_grid() {
        let mut grid: InfiniteGrid<u8> = InfiniteGrid::default();
        let values: HashMap<(isize, isize), u8> =
            [((1, 1), 5), ((3, -1), 4)].iter().copied().collect();
        for ((x, y), value) in values.iter() {
            grid.add(*x, *y, *value);
        }
        for x in -10..=10 {
            for y in -10..=10 {
                let target = values.get(&(x, y)).copied().unwrap_or_default();
                assert_eq!(target, grid.get(x, y));
            }
        }

        assert_eq!(grid.bounding_box(), ((1, -1), (3, 1)));

        grid.move_square((1, 1), (-1, 1));
        assert_eq!(grid.get(1, 1), 0);
        assert_eq!(grid.get(0, 2), 5);

        assert_eq!(grid.bounding_box(), ((0, -1), (3, 2)));
        assert_eq!(format!("{grid}"), "0004\n0000\n0000\n5000\n");
    }

    #[test]
    fn test_nums_from() {
        let examples = [
            ("  Starting items: 79, 98", vec![79, 98]),
            ("  Starting items: 54, 65, 75, 74", vec![54, 65, 75, 74]),
            (
                "Sensor at x=2, y=18: closest beacon is at x=-2, y=15",
                vec![2, 18, -2, 15],
            ),
        ];

        for (ex, nums) in examples.iter() {
            let result: VecDeque<isize> = all_nums_from(ex.to_string());
            assert_eq!(result.len(), nums.len());
            for i in 0..result.len() {
                assert_eq!(result[i], nums[i]);
            }
        }
    }

    #[test]
    fn test_positions_from() {
        let examples = [
            (
                "Sensor at x=20, y=14: closest beacon is at x=25, y=17",
                vec![(20, 14), (25, 17)],
            ),
            (
                "Sensor at x=2, y=18: closest beacon is at x=-2, y=15",
                vec![(2, 18), (-2, 15)],
            ),
        ];

        for (ex, pairs) in examples.iter() {
            let mut result = all_positions_from(ex.to_string());
            for pair in pairs.iter() {
                let p = Position::from(*pair);
                assert_eq!(p, result.pop_front().unwrap());
            }
        }
    }

    #[test]
    fn test_point_parse() {
        for p in ["1,2,3"] {
            let point = p.parse::<Point<i64, 3>>().unwrap();
            assert_eq!(format!("({p})"), format!("{point}"));
        }
        for p in ["(1,2,3)"] {
            let point = p.parse::<Point<i64, 3>>().unwrap();
            assert_eq!(format!("{p}"), format!("{point}"));
        }

        let p = "1,2,3".parse::<Point<i64, 3>>().unwrap();
        assert_eq!(p, "(1, 2, 3)".parse::<Point<i64, 3>>().unwrap());
        assert_eq!(p[0], 1);
        assert_eq!(p[1], 2);
        assert_eq!(p[2], 3);
    }

    #[test]
    fn test_point_adjacent() {
        for (p1, p2, outcome) in [
            ("1,1,1", "2,1,1", true),
            ("1,1,1", "1,1,1", false),
            ("1,1,1", "3,1,1", false),
            ("1,1,1", "0,1,1", true),
            ("1,0,1", "1,1,1", true),
        ] {
            let p1: Point<i64, 3> = p1.parse().unwrap();
            let p2: Point<i64, 3> = p2.parse().unwrap();
            assert_eq!(p1.adjacent(&p2), outcome);
            assert_eq!(p2.adjacent(&p1), outcome);
        }
    }

    #[test]
    fn test_point_math() {
        let p1: Point<i64, 3> = Point::new([1, 2, 3]);
        let p2: Point<i64, 3> = Point::new([4, 5, 6]);
        let add = p1 + p2;
        assert_eq!(add, Point::new([5, 7, 9]));
        let sub = p1 - p2;
        assert_eq!(sub, Point::new([-3, -3, -3]));
    }

    #[test]
    fn test_bounding_coords() {
        let pts = ["1,0,-1", "-1,0,1", "0,-1,0"];
        let bbox = Point::<i64, 3>::min_max_points(pts.iter().map(|p| p.parse().unwrap())).unwrap();
        let lo: Point<i64, 3> = Point::new([-1, -1, -1]);
        let hi: Point<i64, 3> = Point::new([1, 0, 1]);
        assert_eq!((lo, hi), bbox);
    }

    #[test]
    fn test_bounding_box() {
        let pts = ["1,0,-1", "-1,0,1", "0,-1,0"];
        let bbox = Point::<i64, 3>::bounding_box(pts.iter().map(|p| p.parse().unwrap())).unwrap();
        assert_eq!(bbox.len(), 8);
        assert_eq!(format!("{bbox:?}"), "[Point { coords: [-1, -1, -1] }, Point { coords: [-1, -1, 1] }, Point { coords: [-1, 0, -1] }, Point { coords: [-1, 0, 1] }, Point { coords: [1, -1, -1] }, Point { coords: [1, -1, 1] }, Point { coords: [1, 0, -1] }, Point { coords: [1, 0, 1] }]");
    }

    #[test]
    fn test_order() {
        let mut pts: Vec<Point<i64, 3>> = ["-1,0,0", "0,-1,-1", "1,0,-1", "-1,0,1", "0,-1,0"]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        pts.sort();
        assert_eq!("[Point { coords: [-1, 0, 0] }, Point { coords: [-1, 0, 1] }, Point { coords: [0, -1, -1] }, Point { coords: [0, -1, 0] }, Point { coords: [1, 0, -1] }]", format!("{pts:?}"));
    }

    #[test]
    fn test_point_iter() {
        let p: Point<i64, 3> = "3, -1, 2".parse().unwrap();
        assert_eq!(-1, p.values().min().unwrap());
        assert_eq!(3, p.values().max().unwrap());
    }

    #[test]
    fn test_point_move() {
        let mut p: Point<isize, 2> = "0, 0".parse().unwrap();
        for (m, ex) in [
            (ManhattanDir::N, "0, -1"),
            (ManhattanDir::N, "0, -2"),
            (ManhattanDir::E, "1, -2"),
            (ManhattanDir::E, "2, -2"),
            (ManhattanDir::S, "2, -1"),
            (ManhattanDir::W, "1, -1"),
        ] {
            p.manhattan_move(m);
            assert_eq!(p, ex.parse().unwrap());
        }
    }
}
