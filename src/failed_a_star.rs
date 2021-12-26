//use std::collections::BinaryHeap;
use indexmap::IndexSet;
use priority_queue::PriorityQueue;
use crate::{AStarCost, Priority, SearchNode, /*AStarNode, ContinueSearch, search, SearchResult, */ParentMap, SearchQueue};

// This is an archive of failed A* implementations.
// These are flawed in a couple of ways:
//
// * Takes too long
// * Provides worse-than-optimal plans on Advent 2021, Day 15.

/*
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

pub fn best_first_search_m<T, S, C>(start_value: &(AStarCost<C>, T), add_successors: S) -> SearchResult<AStarQueueM<C, T>>
    where T: SearchNode, C: Priority, S: FnMut(&(AStarCost<C>, T), &mut AStarQueueM<C, T>) -> ContinueSearch {
    let mut open_list = AStarQueueM::new();
    open_list.enqueue(start_value);
    search(open_list, add_successors)
}


 */

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
/*
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

 */
