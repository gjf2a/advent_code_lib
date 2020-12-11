use std::slice::Iter;
use std::{io, fs};
use std::io::{Lines, BufReader, BufRead};
use std::fs::File;
use std::collections::{BTreeMap, BTreeSet};

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

#[derive(Debug,Clone,Copy,Eq,PartialEq)]
pub enum Dir {
    N, Ne, E, Se, S, Sw, W, Nw
}

impl Dir {
    pub fn neighbor(&self, col: usize, row: usize) -> (isize,isize) {
        let (d_col, d_row) = match self {
            Dir::N  => ( 0, -1),
            Dir::Ne => (-1, -1),
            Dir::E  => (-1,  0),
            Dir::Se => (-1,  1),
            Dir::S  => ( 0,  1),
            Dir::Sw => ( 1,  1),
            Dir::W  => ( 1,  0),
            Dir::Nw => ( 1, -1)
        };
        (col as isize + d_col, row as isize + d_row)
    }
}

pub struct DirIter {
    d: Option<Dir>
}

impl DirIter {
    pub fn new() -> Self {DirIter {d: Some(Dir::N)}}
}

impl Iterator for DirIter {
    type Item = Dir;

    fn next(&mut self) -> Option<Self::Item> {
        match self.d {
            None => None,
            Some(d) => {
                self.d = match d {
                    Dir::N  => Some(Dir::Ne),
                    Dir::Ne => Some(Dir::E),
                    Dir::E  => Some(Dir::Se),
                    Dir::Se => Some(Dir::S),
                    Dir::S  => Some(Dir::Sw),
                    Dir::Sw => Some(Dir::W),
                    Dir::W  => Some(Dir::Nw),
                    Dir::Nw => None
                };
                Some(d)
            }
        }
    }
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
        assert_eq!(DirIter::new().collect::<Vec<Dir>>(), vec![N,Ne,E,Se,S,Sw,W,Nw]);
        assert_eq!(DirIter::new().map(|d| d.neighbor(4, 4)).collect::<Vec<(isize,isize)>>(),
                   vec![(4, 3), (3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (5, 3)]);
    }
}
