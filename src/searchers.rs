use std::collections::{BinaryHeap, BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;
use common_macros::b_tree_set;
use trait_set::trait_set;
use derive_getters::Getters;

trait_set! {
    pub trait SearchNode = Ord + Eq + Clone + Debug;
}

pub trait SearchQueue<T> {
    fn new() -> Self;
    fn enqueue(&mut self, item: &T);
    fn dequeue(&mut self) -> Option<T>;
    fn len(&self) -> usize;
}

impl <T:Clone+Debug> SearchQueue<T> for VecDeque<T> {
    fn new() -> Self {VecDeque::new()}
    fn enqueue(&mut self, item: &T) {self.push_back(item.clone());}
    fn dequeue(&mut self) -> Option<T> {self.pop_front()}
    fn len(&self) -> usize {self.len()}
}

impl <T:Clone+Debug> SearchQueue<T> for Vec<T> {
    fn new() -> Self {Vec::new()}
    fn enqueue(&mut self, item: &T) {self.push(item.clone());}
    fn dequeue(&mut self) -> Option<T> {self.pop()}
    fn len(&self) -> usize {self.len()}
}

impl <T:SearchNode> SearchQueue<T> for BinaryHeap<T> {
    fn new() -> Self {BinaryHeap::new()}
    fn enqueue(&mut self, item: &T) {self.push(item.clone());}
    fn dequeue(&mut self) -> Option<T> {self.pop()}
    fn len(&self) -> usize {self.len()}
}

#[derive(Debug, Clone)]
pub struct ParentMapQueue<T: SearchNode, Q: SearchQueue<T>> {
    queue: Q,
    last_dequeued: Option<T>,
    parent_map: BTreeMap<T, Option<T>>
}

impl <T: SearchNode, Q: SearchQueue<T>> ParentMapQueue<T, Q> {
    pub fn parent_of(&self, item: &T) -> &Option<T> {
        self.parent_map.get(item).unwrap_or(&None)
    }

    pub fn path_back_from(&self, end: &T) -> VecDeque<T> {
        path_back_from(end, &self.parent_map)
    }
}

impl <T: SearchNode, Q: SearchQueue<T>> SearchQueue<T> for ParentMapQueue<T, Q> {
    fn new() -> Self {
        ParentMapQueue {parent_map: BTreeMap::new(), queue: Q::new(), last_dequeued: None}
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

    fn len(&self) -> usize {self.queue.len()}
}

#[derive(Clone, Debug, Getters)]
pub struct SearchResult<Q> {
    enqueued: usize,
    dequeued: usize,
    open_list: Q
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ContinueSearch {
    Yes, No
}

pub fn search<T, S, Q>(mut open_list: Q, mut add_successors: S) -> SearchResult<Q>
    where T: Clone, Q: SearchQueue<T>, S: FnMut(&T, &mut Q) -> ContinueSearch {
    let mut enqueued = open_list.len();
    let mut dequeued = 0;
    loop {
        match open_list.dequeue() {
            Some(candidate) => {
                dequeued += 1;
                let before = open_list.len();
                let cont = add_successors(&candidate, &mut open_list);
                assert!(open_list.len() >= before);
                enqueued += open_list.len() - before;
                if cont == ContinueSearch::No {
                    break;
                }
            }
            None => break
        }
    }
    SearchResult {enqueued, dequeued, open_list}
}

pub fn breadth_first_search<T,S>(start_value: &T, add_successors: S) -> BTreeMap<T,Option<T>>
    where T: SearchNode, S: FnMut(&T, &mut ParentMapQueue<T, VecDeque<T>>) -> ContinueSearch {
    let mut open_list = ParentMapQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors).open_list.parent_map
}

pub fn path_back_from<T: SearchNode>(end: &T, parent_map: &BTreeMap<T,Option<T>>) -> VecDeque<T> {
    let mut path = VecDeque::new();
    let mut current = end;
    loop {
        path.push_front(current.clone());
        match parent_map.get(&current).unwrap() {
            None => break,
            Some(parent) => {current = parent;}
        }
    }
    path
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AdjacencySets {
    graph: BTreeMap<String,BTreeSet<String>>
}

impl AdjacencySets {
    pub fn new() -> Self {
        AdjacencySets {graph: BTreeMap::new()}
    }

    pub fn keys(&self) -> impl Iterator<Item=&str> {
        self.graph.keys().map(|s| s.as_str())
    }

    pub fn neighbors_of(&self, node: &str) -> Option<&BTreeSet<String>> {
        self.graph.get(node)
    }

    pub fn connect2(&mut self, start: &str, end: &str) {
        self.connect(start, end);
        self.connect(end, start);
    }

    pub fn connect(&mut self, start: &str, end: &str) {
        match self.graph.get_mut(start) {
            None => {
                self.graph.insert(start.to_string(), b_tree_set! {end.to_string()});
            }
            Some(connections) => {
                connections.insert(end.to_string());
            }
        }
    }
}

//fn depth_first_search<T: SearchNode>