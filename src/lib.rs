/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Rurel is a flexible, reusable reinforcement learning (Q learning) implementation in Rust.
//!
//! Implement the [Agent](mdp/trait.Agent.html) and [State](mdp/trait.State.html) traits for your
//! process, then create an [AgentTrainer](struct.AgentTrainer.html) and train it for your process.
//!
//! # Basic Example
//!
//! The following example defines the `State` as a position on a 21x21 2D matrix. The `Action`s
//! that can be taken are: go up, go down, go left and go right. Positions closer to (10, 10) are
//! assigned a higher reward.
//!
//! After training, the AgentTrainer will have assigned higher values to actions which move closer
//! to (10, 10).
//!
//! ```
//! use rurel::mdp::{State, Agent};
//!
//! #[derive(PartialEq, Eq, Hash, Clone)]
//! struct MyState { x: i32, y: i32 }
//! #[derive(PartialEq, Eq, Hash, Clone)]
//! struct MyAction { dx: i32, dy: i32 }
//!
//! impl State for MyState {
//!     type A = MyAction;
//!     fn reward(&self) -> f64 {
//!         // Negative Euclidean distance
//!         -((((10 - self.x).pow(2) + (10 - self.y).pow(2)) as f64).sqrt())
//!     }
//!     fn actions(&self) -> Vec<MyAction> {
//!         vec![MyAction { dx: 0, dy: -1 },    // up
//!              MyAction { dx: 0, dy: 1 }, // down
//!              MyAction { dx: -1, dy: 0 },    // left
//!              MyAction { dx: 1, dy: 0 }, // right
//!         ]
//!     }
//! }
//!
//! struct MyAgent { state: MyState }
//! impl Agent<MyState> for MyAgent {
//!     fn current_state(&self) -> &MyState {
//!         &self.state
//!     }
//!     fn take_action(&mut self, action: &MyAction) -> () {
//!         match action {
//!             &MyAction { dx, dy } => {
//!                 self.state = MyState {
//!                     x: (((self.state.x + dx) % 21) + 21) % 21, // (x+dx) mod 21
//!                     y: (((self.state.y + dy) % 21) + 21) % 21, // (y+dy) mod 21
//!                 }
//!             }
//!         }
//!     }
//! }
//!
//! use rurel::AgentTrainer;
//! use rurel::strategy::learn::QLearning;
//! use rurel::strategy::explore::RandomExploration;
//! use rurel::strategy::terminate::FixedIterations;
//!
//! let mut trainer = AgentTrainer::new();
//! let mut agent = MyAgent { state: MyState { x: 0, y: 0 }};
//! trainer.train(&mut agent,
//!               &QLearning::new(0.2, 0.01, 2.),
//!               &mut FixedIterations::new(100000),
//!               &RandomExploration::new());
//!
//! // Test to see if it worked
//! let test_state = MyState { x: 10, y: 9 };
//! let go_up = MyAction { dx: 0, dy: -1 };
//! let go_down = MyAction { dx: 0, dy: 1};
//! // Going down is better than going up
//! assert!(trainer.expected_value(&test_state, &go_down)
//!     > trainer.expected_value(&test_state, &go_up));
//! ```

pub mod mdp;
pub mod strategy;

use mdp::{Agent, State};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::Hash,
    hash::Hasher,
    marker::PhantomData,
};
use strategy::explore::ExplorationStrategy;
use strategy::learn::LearningStrategy;
use strategy::terminate::TerminationStrategy;
#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct StateHash(u64);

impl<S> From<&S> for StateHash
where
    S: State,
{
    fn from(state: &S) -> Self {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        StateHash(hasher.finish())
    }
}

impl From<u64> for StateHash {
    fn from(v: u64) -> Self {
        StateHash(v)
    }
}

impl From<StateHash> for u64 {
    fn from(sh: StateHash) -> Self {
        sh.0
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct ActionHash(u64);

impl From<u64> for ActionHash {
    fn from(v: u64) -> Self {
        ActionHash(v)
    }
}

impl From<ActionHash> for u64 {
    fn from(ah: ActionHash) -> Self {
        ah.0
    }
}

impl<A> From<&A> for ActionHash
where
    A: Eq + Hash + Clone,
{
    fn from(action: &A) -> Self {
        let mut hasher = DefaultHasher::new();
        action.hash(&mut hasher);
        ActionHash(hasher.finish())
    }
}

/// An `AgentTrainer` can be trained for using a certain [Agent](mdp/trait.Agent.html). After
/// training, the `AgentTrainer` contains learned knowledge about the process, and can be queried
/// for this. For example, you can ask the `AgentTrainer` the expected values of all possible
/// actions in a given state.
pub struct AgentTrainer<S> {
    q: HashMap<u64, HashMap<u64, f64>>,
    state_marker: PhantomData<S>,
}

impl<S> AgentTrainer<S>
where
    S: State,
{
    pub fn new() -> AgentTrainer<S> {
        AgentTrainer {
            q: HashMap::new(),
            state_marker: PhantomData,
        }
    }
    /// Fetches the learned values for the given state, by `Action`, or `None` if no value was
    /// learned.
    pub fn expected_values(&self, state: &S) -> Option<&HashMap<u64, f64>> {
        // XXX: make associated const with empty map and remove Option?
        self.q.get(&StateHash::from(state).0)
    }
    /// Fetches the learned value for the given `Action` in the given `State`, or `None` if no
    /// value was learned.
    pub fn expected_value(&self, state: &S, action: &S::A) -> Option<f64> {
        self.q
            .get(&StateHash::from(state).0)
            .and_then(|m| m.get(&ActionHash::from(action).0).copied())
    }
    /// Returns a clone of the entire learned state to be saved or used elsewhere.
    pub fn export_state(&self) -> HashMap<u64, HashMap<u64, f64>> {
        self.q.clone()
    }
    /// Imports a state, completely replacing any learned progress
    pub fn import_state(&mut self, q: HashMap<u64, HashMap<u64, f64>>) {
        self.q = q;
    }
    /// Trains this [AgentTrainer] using the given [ExplorationStrategy], [LearningStrategy] and
    /// [Agent] until the [TerminationStrategy] decides to stop.
    pub fn train(
        &mut self,
        agent: &mut dyn Agent<S>,
        learning_strategy: &dyn LearningStrategy<S>,
        termination_strategy: &mut dyn TerminationStrategy<S>,
        exploration_strategy: &dyn ExplorationStrategy<S>,
    ) {
        loop {
            let s_t = agent.current_state().clone();
            let action = exploration_strategy.pick_action(agent);

            // current action value
            let s_t_next = agent.current_state();
            let r_t_next = s_t_next.reward();

            let v = {
                let old_value = self
                    .q
                    .get(&StateHash::from(&s_t).0)
                    .and_then(|m| m.get(&ActionHash::from(&action).0));
                learning_strategy.value(
                    &self.q.get(&StateHash::from(s_t_next).0),
                    &old_value,
                    r_t_next,
                )
            };

            self.q
                .entry(StateHash::from(&s_t).0)
                .or_insert_with(HashMap::new)
                .insert(ActionHash::from(&action).0, v);

            if termination_strategy.should_stop(&s_t_next) {
                break;
            }
        }
    }
}

impl<S: mdp::State> Default for AgentTrainer<S> {
    fn default() -> Self {
        Self::new()
    }
}
