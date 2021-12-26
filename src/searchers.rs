use std::cmp::Ordering;
use std::hash::Hash;
use std::ops::Add;
use std::collections::{BinaryHeap, BTreeMap, BTreeSet, VecDeque};
use std::fmt::{Debug, Display};
use common_macros::b_tree_set;
use trait_set::trait_set;
use derive_getters::Getters;
use indexmap::{IndexMap, IndexSet};
use priority_queue::PriorityQueue;

// Searchers to provide:
//
// DFS based on Advent of Code 2021, Day 12
// * Provide:
//   * a successor function
//   * a path-so-far filter

trait_set! {
    pub trait SearchNode = Eq + Clone + Debug + Hash;
    pub trait Priority = Eq + PartialEq + Add<Output=Self> + Ord + PartialOrd + Display + Copy + Clone + Debug;
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

#[derive(Ord, Eq, PartialEq, Copy, Clone, Debug, Hash)]
pub struct AStarCost<N: Priority> {
    cost_so_far: N,
    estimate_to_goal: N
}

impl <N: Priority> AStarCost<N> {
    pub fn new(cost_so_far: N, estimate_to_goal: N) -> Self {
        AStarCost {cost_so_far, estimate_to_goal}
    }

    pub fn cost_so_far(&self) -> N {self.cost_so_far}

    pub fn total_estimate(&self) -> N {self.cost_so_far + self.estimate_to_goal}
}

impl <N: Priority> PartialOrd for AStarCost<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.total_estimate().cmp(&other.total_estimate()).reverse())
    }
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct AStarNode<C: Priority, T: SearchNode> {
    item: T,
    cost: AStarCost<C>
}

impl <C: Priority, T: SearchNode> AStarNode<C, T> {
    pub fn new(item: T, cost: AStarCost<C>) -> Self { AStarNode {item, cost}}
}

impl <C: Priority, T: SearchNode> PartialOrd for AStarNode<C, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl <C: Priority, T: SearchNode> Ord for AStarNode<C, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

// In the style that Mark Goadrich constructed
pub struct AStarQueueM<C: Priority, T: SearchNode> {
    queue: BinaryHeap<AStarNode<C, T>>,
    visited: IndexSet<T>
}

impl <C: Priority, T: SearchNode> SearchQueue<(AStarCost<C>, T)> for AStarQueueM<C, T> {
    fn new() -> Self {
        AStarQueueM {queue: BinaryHeap::new(), visited: IndexSet::new()}
    }

    fn enqueue(&mut self, item: &(AStarCost<C>, T)) {
        let (cost, value) = item;
        if !self.visited.contains(value) {
            self.queue.push(AStarNode::new(value.clone(), cost.clone()));
        }
    }

    fn dequeue(&mut self) -> Option<(AStarCost<C>, T)> {
        self.queue.pop().map(|node| {
            self.visited.insert(node.item.clone());
            (node.cost, node.item)
        })
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

pub struct AStarQueue<C: Priority, T: SearchNode> {
    queue: PriorityQueue<T, AStarCost<C>>,
    visited: IndexSet<T>
}

impl <C: Priority, T: SearchNode> SearchQueue<(AStarCost<C>, T)> for AStarQueue<C, T> {
    fn new() -> Self {
        AStarQueue {queue: PriorityQueue::new(), visited: IndexSet::new()}
    }

    fn enqueue(&mut self, item: &(AStarCost<C>, T)) {
        let (cost, value) = item;
        match self.queue.get_priority(value) {
            None => {if !self.visited.contains(value) {self.queue.push(value.clone(), *cost);}},
            Some(old_cost) => {if cost > old_cost {self.queue.change_priority(value, *cost);}}
        }
    }

    fn dequeue(&mut self) -> Option<(AStarCost<C>, T)> {
        self.queue.pop().map(|(item, cost)| {self.visited.insert(item.clone()); (cost, item)})
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

pub struct AStarQueueShift<C: Priority, T: SearchNode> {
    queue: PriorityQueue<T, AStarCost<C>>,
    parents: ParentMap<T>
}

impl <C: Priority, T: SearchNode> SearchQueue<(AStarCost<C>, T)> for AStarQueueShift<C, T> {
    fn new() -> Self {
        AStarQueueShift {queue: PriorityQueue::new(), parents: ParentMap::new()}
    }

    fn enqueue(&mut self, item: &(AStarCost<C>, T)) {
        let (cost, value) = item;
        if match self.queue.get_priority(value) {
            None => {self.queue.push(value.clone(), *cost); true},
            Some(old_cost) => {
                let changing = cost > old_cost;
                if changing {self.queue.change_priority(value, *cost);}
                changing
            }
        } {
            self.parents.add(value.clone());
        }
    }

    fn dequeue(&mut self) -> Option<(AStarCost<C>, T)> {
        self.queue.pop().map(|(item, cost)| (cost, item))
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[derive(Debug, Clone)]
pub struct ParentMap<T: SearchNode> {
    parents: IndexMap<T, Option<T>>,
    last_dequeued: Option<T>
}

impl <T:SearchNode> ParentMap<T> {
    pub fn new() -> Self {
        ParentMap {parents: IndexMap::new(), last_dequeued: None}
    }

    pub fn len(&self) -> usize {self.parents.len()}

    pub fn keys(&self) -> impl Iterator<Item=&T> {self.parents.keys()}

    pub fn parent_of(&self, item: &T) -> &Option<T> {
        self.parents.get(item).unwrap_or(&None)
    }

    pub fn visited(&self, item: &T) -> bool {
        self.parents.contains_key(item)
    }

    pub fn path_back_from(&self, end: &T) -> VecDeque<T> {
        path_back_from(end, &self.parents)
    }

    pub fn add(&mut self, item: T) {
        self.parents.insert(item, self.last_dequeued.clone());
    }

    pub fn set_last_dequeued(&mut self, item: Option<T>) {
        self.last_dequeued = item;
    }
}

#[derive(Debug, Clone)]
pub struct ParentMapQueue<T: SearchNode, Q: SearchQueue<T>> {
    queue: Q,
    parent_map: ParentMap<T>
}

impl <T: SearchNode, Q: SearchQueue<T>> ParentMapQueue<T, Q> {
    pub fn parent_of(&self, item: &T) -> &Option<T> {self.parent_map.parent_of(item)}

    pub fn path_back_from(&self, end: &T) -> VecDeque<T> {self.parent_map.path_back_from(end)}
}

impl <T: SearchNode, Q: SearchQueue<T>> SearchQueue<T> for ParentMapQueue<T, Q> {
    fn new() -> Self {
        ParentMapQueue {parent_map: ParentMap::new(), queue: Q::new()}
    }

    fn enqueue(&mut self, item: &T) {
        if !self.parent_map.visited(item) {
            self.parent_map.add(item.clone());
            self.queue.enqueue(item);
        }
    }

    fn dequeue(&mut self) -> Option<T> {
        let dequeued = self.queue.dequeue();
        self.parent_map.set_last_dequeued(dequeued.clone());
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

pub fn breadth_first_search<T,S>(start_value: &T, add_successors: S) -> ParentMap<T>
    where T: SearchNode, S: FnMut(&T, &mut ParentMapQueue<T, VecDeque<T>>) -> ContinueSearch {
    let mut open_list = ParentMapQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors).open_list.parent_map
}

pub fn best_first_search<T, S, C>(start_value: &(AStarCost<C>, T), add_successors: S) -> SearchResult<AStarQueue<C, T>>
    where T: SearchNode, C: Priority, S: FnMut(&(AStarCost<C>, T), &mut AStarQueue<C, T>) -> ContinueSearch {
    let mut open_list = AStarQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors)
}

pub fn best_first_search_shift<T, S, C>(start_value: &(AStarCost<C>, T), add_successors: S) -> SearchResult<AStarQueueShift<C, T>>
    where T: SearchNode, C: Priority, S: FnMut(&(AStarCost<C>, T), &mut AStarQueueShift<C, T>) -> ContinueSearch {
    let mut open_list = AStarQueueShift::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors)
}

pub fn best_first_search_m<T, S, C>(start_value: &(AStarCost<C>, T), add_successors: S) -> SearchResult<AStarQueueM<C, T>>
    where T: SearchNode, C: Priority, S: FnMut(&(AStarCost<C>, T), &mut AStarQueueM<C, T>) -> ContinueSearch {
    let mut open_list = AStarQueueM::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors)
}

pub fn depth_first_search<T,S>(start_value: &T, add_successors: S) -> ParentMap<T>
    where T: SearchNode, S: FnMut(&T, &mut ParentMapQueue<T, Vec<T>>) -> ContinueSearch {
    let mut open_list = ParentMapQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors).open_list.parent_map
}

pub fn path_back_from<T: SearchNode>(end: &T, parent_map: &IndexMap<T,Option<T>>) -> VecDeque<T> {
    let mut path = VecDeque::new();
    let mut current = end;
    loop {
        path.push_front(current.clone());
        match parent_map.get(current).unwrap() {
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