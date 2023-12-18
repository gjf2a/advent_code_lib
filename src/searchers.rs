use common_macros::b_tree_set;
use derive_getters::Getters;
use indexmap::IndexMap;
use num::Num;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::Add;
use trait_set::trait_set;

// Searchers to provide:
//
// DFS based on Advent of Code 2021, Day 12
// * Provide:
//   * a successor function
//   * a path-so-far filter

trait_set! {
    pub trait SearchNode = Eq + Clone + Debug + Hash;
    pub trait Priority = Num + Eq + PartialEq + Add<Output=Self> + Ord + PartialOrd + Display + Copy + Clone + Debug;
}

pub trait SearchQueue<T> {
    fn new() -> Self;
    fn enqueue(&mut self, item: &T);
    fn dequeue(&mut self) -> Option<T>;
    fn len(&self) -> usize;
}

impl<T: Clone + Debug> SearchQueue<T> for VecDeque<T> {
    fn new() -> Self {
        VecDeque::new()
    }
    fn enqueue(&mut self, item: &T) {
        self.push_back(item.clone());
    }
    fn dequeue(&mut self) -> Option<T> {
        self.pop_front()
    }
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: Clone + Debug> SearchQueue<T> for Vec<T> {
    fn new() -> Self {
        Vec::new()
    }
    fn enqueue(&mut self, item: &T) {
        self.push(item.clone());
    }
    fn dequeue(&mut self) -> Option<T> {
        self.pop()
    }
    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Debug, Clone)]
struct VisitTracker<C, T> {
    visited: HashMap<T, C>,
}

impl<T: SearchNode, C: Priority> VisitTracker<C, T> {
    fn new() -> Self {
        VisitTracker {
            visited: HashMap::new(),
        }
    }

    fn should_visit(&self, node: &AStarNode<C, T>) -> bool {
        self.visited
            .get(&node.item)
            .map_or(true, |prev_count| node.cost_so_far() < *prev_count)
    }

    fn record_visit(&mut self, node: &AStarNode<C, T>) {
        self.visited.insert(node.item.clone(), node.cost_so_far());
    }
}

#[derive(Clone, Debug)]
pub struct AStarQueue<C: Priority, T: SearchNode> {
    queue: BinaryHeap<AStarNode<C, T>>,
    parents: ParentMap<T>,
    visited: VisitTracker<C, T>,
    last_cost: Option<C>,
}

impl<T: SearchNode, C: Priority> SearchQueue<AStarNode<C, T>> for AStarQueue<C, T> {
    fn new() -> Self {
        AStarQueue {
            queue: BinaryHeap::new(),
            parents: ParentMap::new(),
            visited: VisitTracker::new(),
            last_cost: None,
        }
    }

    fn enqueue(&mut self, node: &AStarNode<C, T>) {
        if self.visited.should_visit(node) {
            self.visited.record_visit(node);
            self.parents.add(node.item.clone());
            self.queue.push(node.clone());
        }
    }

    fn dequeue(&mut self) -> Option<AStarNode<C, T>> {
        let result = self.queue.pop();
        if let Some(node) = &result {
            self.last_cost = Some(node.cost_so_far());
            self.parents.set_last_dequeued(Some(node.item().clone()));
        }
        result
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[derive(Eq, PartialEq, Copy, Clone, Debug, Hash)]
pub struct AStarCost<N: Priority> {
    cost_so_far: N,
    estimate_to_goal: N,
}

impl<N: Priority> AStarCost<N> {
    pub fn new(cost_so_far: N, estimate_to_goal: N) -> Self {
        AStarCost {
            cost_so_far,
            estimate_to_goal,
        }
    }

    pub fn cost_so_far(&self) -> N {
        self.cost_so_far
    }

    pub fn total_estimate(&self) -> N {
        self.cost_so_far + self.estimate_to_goal
    }
}

impl<N: Priority> PartialOrd for AStarCost<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.total_estimate()
            .partial_cmp(&other.total_estimate())
            .map(|ord| ord.reverse())
    }
}

impl<N: Priority> Ord for AStarCost<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct AStarNode<C: Priority, T: SearchNode> {
    item: T,
    cost: AStarCost<C>,
}

impl<C: Priority, T: SearchNode> AStarNode<C, T> {
    pub fn new(item: T, cost: AStarCost<C>) -> Self {
        AStarNode { item, cost }
    }

    pub fn cost_so_far(&self) -> C {
        self.cost.cost_so_far
    }

    pub fn total_estimate(&self) -> C {
        self.cost.total_estimate()
    }

    pub fn item(&self) -> &T {
        &self.item
    }
}

impl<C: Priority, T: SearchNode> PartialOrd for AStarNode<C, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl<C: Priority, T: SearchNode> Ord for AStarNode<C, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct ParentMap<T: SearchNode> {
    parents: IndexMap<T, Option<T>>,
    last_dequeued: Option<T>,
}

impl<T: SearchNode> ParentMap<T> {
    pub fn new() -> Self {
        ParentMap {
            parents: IndexMap::new(),
            last_dequeued: None,
        }
    }

    pub fn len(&self) -> usize {
        self.parents.len()
    }

    pub fn keys(&self) -> impl Iterator<Item = &T> {
        self.parents.keys()
    }

    pub fn parent_of(&self, item: &T) -> &Option<T> {
        self.parents.get(item).unwrap_or(&None)
    }

    pub fn visited(&self, item: &T) -> bool {
        self.parents.contains_key(item)
    }

    pub fn path_back_from(&self, end: &T) -> Option<VecDeque<T>> {
        path_back_from(end, &self.parents)
    }

    pub fn add(&mut self, item: T) {
        self.parents.insert(item, self.last_dequeued.clone());
    }

    pub fn set_last_dequeued(&mut self, item: Option<T>) {
        if item.is_some() {
            self.last_dequeued = item;
        }
    }

    pub fn get_last_dequeued(&self) -> &Option<T> {
        &self.last_dequeued
    }
}

#[derive(Debug, Clone)]
pub struct ParentMapQueue<T: SearchNode, Q: SearchQueue<T>> {
    queue: Q,
    parent_map: ParentMap<T>,
}

impl<T: SearchNode, Q: SearchQueue<T>> ParentMapQueue<T, Q> {
    pub fn parent_of(&self, item: &T) -> &Option<T> {
        self.parent_map.parent_of(item)
    }

    pub fn path_back_from(&self, end: &T) -> Option<VecDeque<T>> {
        self.parent_map.path_back_from(end)
    }
}

impl<T: SearchNode, Q: SearchQueue<T>> SearchQueue<T> for ParentMapQueue<T, Q> {
    fn new() -> Self {
        ParentMapQueue {
            parent_map: ParentMap::new(),
            queue: Q::new(),
        }
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

    fn len(&self) -> usize {
        self.queue.len()
    }
}

#[derive(Clone, Debug, Getters)]
pub struct SearchResult<Q> {
    enqueued: usize,
    dequeued: usize,
    open_list: Q,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ContinueSearch {
    Yes,
    No,
}

pub fn search<T, S, Q>(mut open_list: Q, mut add_successors: S) -> SearchResult<Q>
where
    T: Clone,
    Q: SearchQueue<T>,
    S: FnMut(&T, &mut Q) -> ContinueSearch,
{
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
            None => break,
        }
    }
    SearchResult {
        enqueued,
        dequeued,
        open_list,
    }
}

pub fn breadth_first_search<T, S>(start_value: &T, add_successors: S) -> ParentMap<T>
where
    T: SearchNode,
    S: FnMut(&T, &mut ParentMapQueue<T, VecDeque<T>>) -> ContinueSearch,
{
    let mut open_list = ParentMapQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors).open_list.parent_map
}

pub fn heuristic_search<T, P, C, G, H, S>(
    start_value: T,
    node_cost: P,
    at_goal: G,
    heuristic: H,
    get_successors: S,
) -> SearchResult<AStarQueue<C, T>>
where
    T: SearchNode,
    C: Priority,
    P: Fn(&T) -> C,
    G: Fn(&T) -> bool,
    H: Fn(&T) -> C,
    S: Fn(&T) -> Vec<T>,
{
    let cost = AStarCost {
        cost_so_far: C::zero(),
        estimate_to_goal: heuristic(&start_value),
    };
    let start_value = AStarNode::new(start_value, cost);
    best_first_search(&start_value, |n, s| {
        if at_goal(n.item()) {
            ContinueSearch::No
        } else {
            for succ in get_successors(n.item()) {
                let cost = AStarCost {
                    cost_so_far: node_cost(n.item()),
                    estimate_to_goal: heuristic(n.item()),
                };
                s.enqueue(&AStarNode::new(succ, cost));
            }
            ContinueSearch::Yes
        }
    })
}

pub fn heuristic_search_path_check<T, P, C, G, H, A, S>(
    start_value: T,
    node_cost: P,
    at_goal: G,
    heuristic: H,
    path_approved: A,
    get_successors: S,
) -> SearchResult<AStarQueue<C, T>>
where
    T: SearchNode,
    C: Priority,
    P: Fn(&T) -> C,
    G: Fn(&T) -> bool,
    H: Fn(&T) -> C,
    A: Fn(VecDeque<T>) -> bool,
    S: Fn(&T) -> Vec<T>,
{
    let cost = AStarCost {
        cost_so_far: C::zero(),
        estimate_to_goal: heuristic(&start_value),
    };
    let start_value = AStarNode::new(start_value, cost);
    best_first_search(&start_value, |n, s| {
        if s.parents
            .path_back_from(&n.item())
            .map_or(true, |path| path_approved(path))
        {
            if at_goal(n.item()) {
                return ContinueSearch::No;
            } else {
                for succ in get_successors(n.item()) {
                    let cost = AStarCost {
                        cost_so_far: node_cost(n.item()),
                        estimate_to_goal: heuristic(n.item()),
                    };

                    s.enqueue(&AStarNode::new(succ, cost));
                }
            }
        } 
        ContinueSearch::Yes
    })
}

pub fn best_first_search<T, S, C>(
    start_value: &AStarNode<C, T>,
    add_successors: S,
) -> SearchResult<AStarQueue<C, T>>
where
    T: SearchNode,
    C: Priority,
    S: FnMut(&AStarNode<C, T>, &mut AStarQueue<C, T>) -> ContinueSearch,
{
    let mut open_list = AStarQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors)
}

impl<C: Priority, T: SearchNode> SearchResult<AStarQueue<C, T>> {
    pub fn cost(&self) -> Option<C> {
        self.open_list.last_cost
    }

    pub fn node_at_goal(&self) -> Option<T> {
        self.path().and_then(|d| d.back().cloned())
    }

    pub fn path(&self) -> Option<VecDeque<T>> {
        self.open_list
            .parents
            .get_last_dequeued()
            .as_ref()
            .and_then(|last| self.open_list.parents.path_back_from(last))
    }
}

pub fn depth_first_search<T, S>(start_value: &T, add_successors: S) -> ParentMap<T>
where
    T: SearchNode,
    S: FnMut(&T, &mut ParentMapQueue<T, Vec<T>>) -> ContinueSearch,
{
    let mut open_list = ParentMapQueue::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors).open_list.parent_map
}

pub fn path_back_from<T: SearchNode>(
    end: &T,
    parent_map: &IndexMap<T, Option<T>>,
) -> Option<VecDeque<T>> {
    let mut path = VecDeque::new();
    let mut current = end;
    loop {
        path.push_front(current.clone());
        match parent_map.get(current) {
            None => return None,
            Some(parent) => match parent {
                None => return Some(path),
                Some(parent) => {
                    current = parent;
                }
            },
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AdjacencySets {
    graph: BTreeMap<String, BTreeSet<String>>,
}

impl AdjacencySets {
    pub fn new() -> Self {
        AdjacencySets {
            graph: BTreeMap::new(),
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
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
                self.graph
                    .insert(start.to_string(), b_tree_set! {end.to_string()});
            }
            Some(connections) => {
                connections.insert(end.to_string());
            }
        }
    }
}

//fn depth_first_search<T: SearchNode>
