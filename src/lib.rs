#[macro_use] extern crate maplit;

use std::slice::Iter;
use std::{io, fs};
use std::io::{Lines, BufReader, BufRead};
use std::fs::File;
use std::collections::{BTreeMap, BTreeSet};
use std::collections::btree_map::Keys;

pub fn all_lines(filename: &str) -> io::Result<Lines<BufReader<File>>> {
    Ok(io::BufReader::new(fs::File::open(filename)?).lines())
}

pub fn for_each_line<F: FnMut(&str) -> io::Result<()>>(filename: &str, mut line_processor: F) -> io::Result<()> {
    for line in all_lines(filename)? {
        line_processor(line?.as_str())?;
    }
    Ok(())
}

pub fn file2nums(filename: &str) -> io::Result<Vec<isize>> {
    Ok(all_lines(filename)?.map(|line| line.unwrap().parse::<isize>().unwrap()).collect())
}

pub fn pass_counter<F: Fn(&str) -> bool>(filename: &str, passes_check: F) -> io::Result<String> {
    Ok(all_lines(filename)?
        .filter(|line| line.as_ref().map_or(false, |line| passes_check(line.as_str())))
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

#[derive(Clone,Debug,Eq,Ord,PartialOrd,PartialEq)]
pub struct StringGraph {
    node2nodes: BTreeMap<String,BTreeSet<String>>
}

impl StringGraph {
    pub fn new() -> Self {StringGraph {node2nodes: BTreeMap::new()}}

    pub fn add_if_absent(&mut self, name: &str) {
        if !self.node2nodes.contains_key(name) {
            self.node2nodes.insert(name.to_string(), BTreeSet::new());
        }
    }

    pub fn add_edge(&mut self, start: &str, end: &str) {
        self.add_if_absent(start);
        self.add_if_absent(end);
        self.node2nodes.get_mut(start).unwrap().insert(end.to_string());
    }

    pub fn all_node_names(&self) -> Keys<String,BTreeSet<String>> {
        self.node2nodes.keys()
    }

    pub fn all_successors_of(&self, name: &str) -> BTreeSet<String> {
        let mut visited = BTreeSet::new();
        if self.node2nodes.contains_key(name) {
            let mut open_list: Vec<String> = self.node2nodes.get(name).unwrap().iter()
                .map(|s| s.clone())
                .collect();
            while open_list.len() > 0 {
                let candidate = open_list.pop().unwrap();
                if !visited.contains(candidate.as_str()) {
                    self.node2nodes.get(candidate.as_str()).unwrap().iter()
                        .for_each(|s| open_list.push(s.clone()));
                    visited.insert(candidate);
                }
            }
        }
        visited
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::FromIterator;

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
    pub fn test_string_graph() {
        let mut sg = StringGraph::new();
        [("a", "b"), ("b", "c"), ("c", "d"), ("b", "e")].iter()
            .for_each(|(a, b)| sg.add_edge(*a, *b));
        [("a", btreeset!("b", "c", "d", "e")), ("b", btreeset!("c", "d", "e")),
            ("c", btreeset!("d")), ("d", btreeset!()), ("e", btreeset!())].iter()
            .for_each(|(k, s)| {
            assert_eq!(sg.all_successors_of(k),
                       s.iter().map(|s| s.to_string()).collect::<BTreeSet<String>>());
        });
    }
}
