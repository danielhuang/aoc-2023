#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]
#![feature(file_create_new)]
#![feature(iter_array_chunks)]
#![feature(const_for)]
#![feature(box_patterns)]
#![feature(closure_track_caller)]
#![feature(backtrace_frames)]
#![feature(return_position_impl_trait_in_trait)]
// #![feature(return_position_impl_trait_in_trait)]

pub use ::tap::*;
pub use btree_vec::BTreeVec;
pub use cached::proc_macro::cached;
pub use derive_more::{Add, AddAssign, Sub, SubAssign, Sum};
pub use itertools::Itertools;
use multimap::MultiMap;
pub use num::*;
use owo_colors::OwoColorize;
pub use pathfinding::directed::astar::*;
pub use pathfinding::directed::bfs::*;
pub use pathfinding::directed::count_paths::*;
pub use pathfinding::directed::cycle_detection::*;
pub use pathfinding::directed::dfs::*;
pub use pathfinding::directed::dijkstra::*;
pub use pathfinding::directed::edmonds_karp::*;
pub use pathfinding::directed::fringe::*;
pub use pathfinding::directed::idastar::*;
pub use pathfinding::directed::iddfs::*;
pub use pathfinding::directed::strongly_connected_components::*;
pub use pathfinding::directed::topological_sort::*;
pub use pathfinding::directed::yen::*;
pub use pathfinding::grid::*;
pub use pathfinding::kuhn_munkres::*;
pub use pathfinding::undirected::connected_components::*;
pub use pathfinding::undirected::kruskal::*;
pub use pathfinding::utils::*;
pub use prime_factorization::*;
use regex::{Captures, Match, Regex};
use reqwest::blocking::Client;
pub use rustc_hash::{FxHashMap, FxHashSet};
use serde::de::DeserializeOwned;
pub use serde::{Deserialize, Serialize};
use serde_json::Value;
pub use std::any::Any;
use std::array;
use std::backtrace::{Backtrace, BacktraceFrame};
pub use std::cmp::Ordering;
pub use std::collections::*;
pub use std::fmt::{Debug, Display};
use std::fs;
use std::fs::metadata;
pub use std::fs::{read_to_string, File};
pub use std::hash::Hash;
use std::io::Read;
pub use std::io::Write;
pub use std::iter::from_fn;
pub use std::ops::Mul;
use std::ops::RangeBounds;
pub use std::process::{Command, Stdio};
use std::ptr::null;
use std::str::{FromStr, Split};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, SystemTimeError};
use std::time::{Duration, Instant};
pub use std::{env, io};
use std::collections::hash_map::Iter;
use std::hash::BuildHasherDefault;
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout};

pub mod cartesian;
pub mod defaultmap;
pub mod printer;

pub use crate::cartesian::*;
pub use crate::defaultmap::*;
pub use crate::printer::*;

use mimalloc::MiMalloc;
use reqwest::{Request, RequestBuilder};
use rustc_hash::FxHasher;
use terminal_size::{Height, Width};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn fetch(url: &str) -> Result<String, reqwest::Error> {
    Client::new()
        .get(url)
        .header(
            "cookie",
            format!("session={}", env::var("AOC_SESSION").unwrap()),
        )
        .header(
            "user-agent",
            "github.com/danielhuang/aoc-2023 - hello@danielh.cc",
        )
        .send()?
        .error_for_status()?
        .text()
}

#[cfg(debug_assertions)]
pub const DEBUG: bool = true;

#[cfg(not(debug_assertions))]
pub const DEBUG: bool = false;

static SUBMITTED: Mutex<bool> = Mutex::new(false);

static START_TS: Mutex<Option<Instant>> = Mutex::new(None);

fn read_clipboard() -> Option<String> {
    let mut cmd: Child = Command::new("xclip")
        .arg("-o")
        .arg("clip")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout: ChildStdout = cmd.stdout.take().unwrap();
    cmd.wait().unwrap();
    let mut s: String = String::new();
    match stdout.read_to_string(&mut s) {
        Ok(_) => Some(s),
        Err(e) => {
            dbg!(e);
            None
        }
    }
}

fn day() -> u8 {
    let exe: String = env::args().next().unwrap();
    let exe_path: &Path = Path::new(&exe);
    let day_str: &str = exe_path.file_stem().unwrap().to_str().unwrap();

    // Print the value of day_str for debugging
    println!("Day String: {:?}", day_str);

    // Attempt to parse day_str as u8
    match day_str.parse::<u8>() {
        Ok(parsed_day) => parsed_day,
        Err(err) => {
            eprintln!("Failed to parse day as u8: {:?}", err);
            // Handle the error, you might want to return a default value or panic.
            // For now, we'll panic to show that an error occurred.
            panic!("Failed to parse day as u8");
        }
    }
}


fn write_atomic(filename: &str, data: &str) {
    let tmp: u128 = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let tmp: String = format!("{filename}.{}", tmp);
    File::create_new(&tmp)
        .unwrap()
        .write_all(data.as_bytes())
        .unwrap();
    fs::rename(tmp, filename).unwrap();
}

pub fn load_input() -> String {
    better_panic::install();

    let input: String = {
        let sample: String = read_to_string(format!("src/bin/{}.sample.txt", day())).unwrap();
        if sample.trim().is_empty() {
            println!("{}", "reading sample input from clipboard!!".red().bold());
            read_clipboard().unwrap()
        } else {
            println!("{}", "using saved sample input".blue().bold());
            sample
        }
    };

    //     } else {
    //         let url = format!("https://adventofcode.com/2023/day/{}/input", day());
    //         let path = format!("target/{}.input.txt", day());
    //         let input = match read_to_string(&path) {
    //             Ok(x) => x,
    //             Err(e) => {
    //                 println!("{e:?}");
    //                 print!("Downloading input... ");
    //                 io::stdout().flush().unwrap();
    //                 match fetch(&url) {
    //                     Ok(input) => {
    //                         write_atomic(&path, &input);
    //                         println!("done!");
    //                         input
    //                     }
    //                     Err(e) => {
    //                         dbg!(e);
    //                         println!("testing session cookie...");
    //                         assert!(fetch("https://adventofcode.com/2023")
    //                             .unwrap()
    //                             .contains("[Log Out]"));
    //                         panic!("cookie works, input missing!")
    //                     }
    //                 }
    //             }
    //         };
    //         let submitted_path = format!("target/{}.html", day());
    //         let submitted = match metadata(&submitted_path) {
    //             Ok(_) => true,
    //             Err(_) => {
    //                 let page = fetch(&format!("https://adventofcode.com/2023/day/{}", day())).unwrap();
    //                 if page.contains(
    //                     "Both parts of this puzzle are complete! They provide two gold stars: **",
    //                 ) {
    //                     write_atomic(&submitted_path, &page);
    //                     true
    //                 } else {
    //                     false
    //                 }
    //             }
    //         };
    //         *SUBMITTED.lock().unwrap() = submitted;
    //         input

    *START_TS.lock().unwrap() = Some(Instant::now());

    println!(
        "loaded input: {} chars, {} lines, {} paras",
        input.len(),
        input.lines().count(),
        input.split("\n\n").count()
    );

    let mut lines = input.lines();
    if let Some(line) = lines.next() {
        println!("{}", line.blue());
    }
    let last = lines.next_back();
    if let Some(_) = lines.next() {
        println!("(... {} more lines)", lines.count() + 1);
    }
    if let Some(line) = last {
        println!("{}", line.blue());
    }
    bar();

    input
}

pub fn cp(x: impl Display) {
    let elapsed: Duration = START_TS.lock().unwrap().unwrap().elapsed();
    let elapsed: String = format!("{:?}", elapsed);

    static COPIES: Mutex<usize> = Mutex::new(0);
    let mut copies: MutexGuard<usize> = COPIES.lock().unwrap();
    if *copies >= 2 {
        println!("value: {}", x.red().bold());
        panic!("already copied twice");
    }
    *copies += 1;

    println!(
        "value: {} (debug mode, not copying) took {}",
        x.blue().bold(),
        elapsed.yellow()
    );

    *START_TS.lock().unwrap() = Some(Instant::now());
}

pub fn force_copy(x: &impl Display) {
    // Copy it twice to work around a bug.
    for _ in 0..2 {
        let mut cmd: Child = Command::new("xclip")
            .arg("-sel")
            .arg("clip")
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin: ChildStdin = cmd.stdin.take().unwrap();
        stdin.write_all(x.to_string().as_bytes()).unwrap();
        stdin.flush().unwrap();
        drop(stdin);
        cmd.wait().unwrap();
    }
}

pub fn cp1(x: impl Display) {
    cp(x);
    panic!("exiting after copy")
}

pub trait CollectionExt<T> {
    fn min_c(&self) -> T;
    fn max_c(&self) -> T;
    fn iter_c(&self) -> impl Iterator<Item = T>;
}

impl<T: Clone + Ord, C: IntoIterator<Item = T> + Clone> CollectionExt<T> for C {
    fn min_c(&self) -> T {
        self.clone().into_iter().min().unwrap()
    }

    fn max_c(&self) -> T {
        self.clone().into_iter().max().unwrap()
    }

    fn iter_c(&self) -> impl Iterator<Item = T> {
        self.clone().into_iter()
    }
}

pub fn collect_2d<T>(s: impl IntoIterator<Item = impl IntoIterator<Item = T>>) -> Vec<Vec<T>> {
    s.into_iter().map(|x| x.into_iter().collect()).collect()
}

pub fn transpose_vec<T: Default + Clone>(s: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(s.iter().map(|x: &Vec<T>| x.len()).all_equal());
    assert!(!s.is_empty());
    assert!(!s[0].is_empty());
    let mut result: Vec<Vec<T>>  = vec![vec![T::default(); s.len()]; s[0].len()];
    for (i, row) in s.iter().cloned().enumerate() {
        for (j, x) in row.iter().cloned().enumerate() {
            result[j][i] = x;
        }
    }
    result
}

pub fn transpose<T: Default + Clone>(
    s: impl IntoIterator<Item = impl IntoIterator<Item = T>>,
) -> Vec<Vec<T>> {
    transpose_vec(collect_2d(s))
}

pub fn set_n<T: Eq + Hash + Clone>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
) -> HashSet<T> {
    let a: HashSet<_> = a.into_iter().collect();
    let b: HashSet<_> = b.into_iter().collect();
    a.intersection(&b).cloned().collect()
}

pub fn set_u<T: Eq + Hash + Clone>(
    a: impl IntoIterator<Item = T>,
    b: impl IntoIterator<Item = T>,
) -> HashSet<T> {
    let a: HashSet<_> = a.into_iter().collect();
    let b: HashSet<_> = b.into_iter().collect();
    a.union(&b).cloned().collect()
}

pub trait ExtraItertools: IntoIterator + Sized {
    fn collect_set(self) -> HashSet<Self::Item>
    where
        Self::Item: Eq + Hash,
    {
        self.ii().collect()
    }

    fn cset(self) -> HashSet<Self::Item>
    where
        Self::Item: Eq + Hash,
    {
        self.ii().collect()
    }

    fn collect_string(self) -> String
    where
        Self::Item: Display,
    {
        self.ii().map(|x| x.to_string()).collect()
    }

    fn cii(&self) -> std::vec::IntoIter<Self::Item>
    where
        Self: Clone,
    {
        self.clone().into_iter().collect_vec().into_iter()
    }

    fn ii(self) -> <Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    fn test(
        self,
        mut pass: impl FnMut(&Self::Item) -> bool,
        mut fail: impl FnMut(&Self::Item) -> bool,
    ) -> bool {
        for item in self {
            if pass(&item) {
                return true;
            }
            if fail(&item) {
                return false;
            }
        }
        unreachable!("the iterator does not pass or fail");
    }

    fn one(&self) -> Self::Item
    where
        Self: Clone,
    {
        let mut iter = self.cii();
        let item:<Self as IntoIterator>::Item = iter.next().unwrap();
        assert!(iter.next().is_none());

        item
    }

    fn cv(self) -> Vec<Self::Item> {
        self.into_iter().collect()
    }

    fn cbv(self) -> BTreeVec<Self::Item> {
        self.into_iter().collect()
    }

    fn cbt(self) -> BTreeVec<Self::Item> {
        self.into_iter().collect()
    }

    fn ca<const N: usize>(self) -> [Self::Item; N]
    where
        <Self as IntoIterator>::Item: Debug,
    {
        self.cv().try_into().unwrap()
    }

    fn sumi(self) -> Self::Item
    where
        <Self as IntoIterator>::Item: std::ops::Add<Output = Self::Item> + Default,
    {
        self.ii().fold(Default::default(), |a: <Self as IntoIterator>::Item, b| a + b)
    }
}

impl<T: IntoIterator + Sized + Clone> ExtraItertools for T {}

pub fn freqs<T: Hash + Eq>(i: impl IntoIterator<Item = T>) -> DefaultHashMap<T, usize> {
    let mut result: DefaultHashMap<T, usize> = DefaultHashMap::new(0);
    for x in i {
        result[x] += 1;
    }
    result
}

pub trait SignedExt: Signed {
    fn with_abs(self, f: impl FnOnce(Self) -> Self) -> Self {
        f(self.abs()) * self.signum()
    }
}

impl<T: Signed> SignedExt for T {}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Snailfish {
    Num(i64),
    Array(Vec<Snailfish>),
}

impl Snailfish {
    pub fn from_value(v: &Value) -> Self {
        match v {
            Value::Number(x) => Self::Num(x.as_i64().unwrap()),
            Value::Array(x) => Self::Array(x.iter().map(Self::from_value).collect_vec()),
            _ => unreachable!("invalid"),
        }
    }
}

impl FromStr for Snailfish {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json::from_str(s)
    }
}

impl PartialOrd for Snailfish {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Snailfish {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Snailfish::Num(a), Snailfish::Num(b)) => a.cmp(b),
            (Snailfish::Num(a), Snailfish::Array(b)) => vec![Snailfish::Num(*a)].cmp(b),
            (Snailfish::Array(a), Snailfish::Num(b)) => a.cmp(&vec![Snailfish::Num(*b)]),
            (Snailfish::Array(a), Snailfish::Array(b)) => a.cmp(b),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IntervalEdge {
    Start,
    End,
}

#[derive(Default, Debug, Clone)]
pub struct Intervals {
    bounds: BTreeMap<i64, IntervalEdge>,
}

impl Intervals {
    fn remove_between(&mut self, range: impl RangeBounds<i64>) {
        let (start, end) = (range.start_bound(), range.end_bound());
        match (start, end) {
            (Bound::Excluded(s), Bound::Excluded(e)) if s == e => {
                return;
            }
            (Bound::Included(s) | Bound::Excluded(s), Bound::Included(e) | Bound::Excluded(e))
                if s > e =>
            {
                return;
            }
            _ => {}
        }
        for to_remove in self.bounds.range(range).map(|x: (&i64, &IntervalEdge)| *x.0).collect_vec() {
            self.bounds.remove(&to_remove);
        }
    }

    pub fn add(&mut self, start: i64, end: i64) {
        match (self.is_inside(start), self.is_inside(end)) {
            (true, true) => {
                self.remove_between((start + 1)..=end);
            }
            (true, false) => {
                self.remove_between((start + 1)..=end);
                self.bounds.insert(end, IntervalEdge::End);
            }
            (false, true) => {
                self.remove_between((start + 1)..=end);
                self.bounds.insert(start, IntervalEdge::Start);
            }
            (false, false) => {
                self.remove_between((start + 1)..=end);
                self.bounds.insert(start, IntervalEdge::Start);
                self.bounds.insert(end, IntervalEdge::End);
            }
        }
    }

    pub fn remove(&mut self, start: i64, end: i64) {
        match (self.is_inside(start), self.is_inside(end)) {
            (true, true) => {
                self.remove_between(start..end);
                self.bounds.insert(start, IntervalEdge::End);
                self.bounds.insert(end, IntervalEdge::Start);
            }
            (true, false) => {
                self.remove_between(start..end);
                self.bounds.insert(start, IntervalEdge::End);
            }
            (false, true) => {
                self.remove_between(start..end);
                self.bounds.insert(end, IntervalEdge::Start);
            }
            (false, false) => {
                self.remove_between(start..end);
            }
        }
    }

    pub fn remove_one(&mut self, x: i64) {
        self.remove(x, x + 1);
    }

    pub fn is_inside(&self, x: i64) -> bool {
        if let Some(edge) = self.bounds.range(..=x).next_back() {
            edge.1 == &IntervalEdge::Start
        } else {
            false
        }
    }

    pub fn covered_size(&self) -> i64 {
        let mut total: i64 = 0;
        for (left, right) in self.bounds.iter().tuple_windows() {
            match left.1 {
                IntervalEdge::Start => {
                    total += right.0 - left.0;
                }
                IntervalEdge::End => {}
            }
        }
        total
    }
}

pub fn bfs2<T: Clone + Hash + Eq, I: IntoIterator<Item = T>>(
    start: T,
    mut find_nexts: impl FnMut(usize, T) -> I,
) -> impl Iterator<Item = (usize, T)> {
    let mut edge: VecDeque<T> = VecDeque::new();
    let mut seen: HashSet<T> = HashSet::new();

    seen.insert(start.clone());
    edge.push_back(start);

    let mut i: usize = 0;

    from_fn(move || {
        let mut result: Vec<(usize, T)> = vec![];
        for _ in 0..edge.len() {
            let item: T = edge.pop_front()?;
            let nexts: I = find_nexts(i, item.clone());
            for next in nexts {
                seen.insert(next.clone());
                edge.push_back(next);
            }
            result.push((i, item));
        }
        i += 1;
        if result.is_empty() {
            return None;
        }
        Some(result)
    })
    .flatten()
}

pub fn sometimes() -> bool {
    static PREV: Mutex<Option<Instant>> = Mutex::new(None);
    static COUNT: AtomicUsize = AtomicUsize::new(0);

    let count: usize = COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let mut s: MutexGuard<Option<Instant>> = PREV.lock().unwrap();
    let result: bool = s.is_none() || s.is_some_and(|x: Instant| x.elapsed() > Duration::from_millis(250));
    if result {
        println!("sometimes count: {count}");
        *s = Some(Instant::now());
    }
    result
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cuboid<const N: usize> {
    // points, inclusive
    pub min: [i64; N],
    pub max: [i64; N],
}

impl<const N: usize> Cuboid<N> {
    pub fn length(&self, dim: usize) -> i64 {
        self.max[dim] - self.min[dim]
    }

    pub fn lengths(&self) -> [i64; N] {
        array::from_fn(|dim: usize| self.length(dim))
    }

    pub fn size(&self) -> i64 {
        self.lengths().into_iter().product()
    }

    pub fn volume(&self) -> i64 {
        self.size()
    }

    pub fn surface_area(&self) -> i64 {
        2 * (0..N)
            .map(|k: usize| {
                (0..N)
                    .filter(|&j: &usize| k != j)
                    .map(|j: usize| self.length(j))
                    .product::<i64>()
            })
            .sum::<i64>()
    }

    fn assert(&self) {
        for dim in 0..N {
            assert!(self.max[dim] >= self.min[dim]);
        }
    }

    pub fn resize(&self, amount: i64) -> Self {
        let mut new: Cuboid<N> = *self;
        for dim in 0..N {
            new.min[dim] -= amount;
            new.max[dim] += amount;
        }
        new.assert();
        new
    }

    pub fn grow(&self) -> Self {
        self.resize(1)
    }

    pub fn shrink(&self) -> Self {
        self.resize(-1)
    }

    pub fn contains_point(&self, p: Point<N>) -> bool {
        (0..N).all(|dim: usize| (self.min[dim]..=self.max[dim]).contains(&p.0[dim]))
    }

    pub fn contains(&self, p: impl Boundable<N>) -> bool {
        p.points().all(|x: Point<N>| self.contains_point(x))
    }

    fn things_inside(&self, length_add: i64) -> Vec<[i64; N]> {
        let total: i64 = (0..N).map(|x: usize| self.length(x) + length_add).product();
        let mut output: Vec<[i64; N]> = vec![];
        for mut n in 0..total {
            let mut pos: [i64; N] = self.min;
            for (dim, x) in pos.iter_mut().enumerate() {
                *x += n % (self.length(dim) + length_add);
                n /= self.length(dim) + length_add;
            }
            output.push(pos);
        }
        output
    }

    pub fn points(&self) -> Vec<Point<N>> {
        self.things_inside(1).into_iter().map(Point).collect()
    }

    pub fn cells(&self) -> Vec<Cell<N>> {
        self.things_inside(0).into_iter().map(Cell).collect()
    }

    pub fn corner_cells(&self) -> [Cell<N>; 2] {
        [Cell(self.min), Cell(self.max.map(|x| x - 1))]
    }

    pub fn corner_points(&self) -> [Point<N>; 2] {
        [Point(self.min), Point(self.max)]
    }

    pub fn wrap<T: Cartesian<N>>(&self, value: T) -> T {
        T::new(array::from_fn(|dim: usize| {
            let relative: i64 = value[dim] - self.min[dim];
            let wrapped_relative: i64 = relative.rem_euclid(self.length(dim));
            wrapped_relative + self.min[dim]
        }))
    }
}

pub fn bounds_points<const N: usize>(i: impl IntoIterator<Item = Point<N>>) -> Cuboid<N> {
    let mut i = i.into_iter();

    let first: Point<N> = i.next().expect("must have at least one point");
    let mut bounds = Cuboid {
        min: first.0,
        max: first.0,
    };

    for c in i {
        for (dim, val) in c.0.into_iter().enumerate() {
            if val < bounds.min[dim] {
                bounds.min[dim] = val;
            }
            if val > bounds.max[dim] {
                bounds.max[dim] = val;
            }
        }
    }

    bounds.assert();

    bounds
}

pub fn bounds<const N: usize>(i: impl IntoIterator<Item = impl Boundable<N>>) -> Cuboid<N> {
    bounds_points(i.into_iter().flat_map(|x| x.points().collect_vec()))
}

pub trait DefaultHashMapExt<K, V> {
    fn find(&self, f: impl FnMut(V) -> bool) -> Vec<K>;
    fn findv(&self, value: V) -> Vec<K>;
    fn undef(&self) -> HashMap<K, V>;
    fn invert(&self) -> DefaultHashMap<V, Vec<K>>
    where
        V: Clone + Eq + Hash,
        K: Clone;
    fn uninvert(&self) -> HashMap<<V as IntoIterator>::Item, K>
    where
        K: Clone + Eq + Hash,
        V: IntoIterator,
        <V as IntoIterator>::Item: Clone,
        <V as IntoIterator>::Item: Eq + Hash;
}

impl<K: Eq + Hash + Clone, V: Clone + PartialEq> DefaultHashMapExt<K, V> for DefaultHashMap<K, V> {
    fn find(&self, mut f: impl FnMut(V) -> bool) -> Vec<K> {
        let mut l: Vec<K> = vec![];
        for (k, v) in self.iter() {
            if f(v.clone()) {
                l.push(k.clone());
            }
        }
        l
    }

    fn findv(&self, value: V) -> Vec<K> {
        self.find(|v: V| v == value)
    }

    fn undef(&self) -> HashMap<K, V> {
        self.iter()
            .filter(|(_, v)| **v != self.default)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn invert(&self) -> DefaultHashMap<V, Vec<K>>
    where
        V: Clone + Eq + Hash,
        K: Clone,
    {
        let mut map: DefaultHashMap<V, Vec<K>> = DefaultHashMap::new(vec![]);
        for (k, v) in self.iter() {
            map[v.clone()].push(k.clone());
        }
        map
    }

    fn uninvert(&self) -> HashMap<<V as IntoIterator>::Item, K>
    where
        K: Clone + Eq + Hash,
        V: IntoIterator,
        <V as IntoIterator>::Item: Clone,
        <V as IntoIterator>::Item: Eq + Hash,
    {
        let mut map: HashMap<<V as IntoIterator>::Item, K> = HashMap::new();
        for (k, v) in self.iter() {
            for v in v.clone().into_iter() {
                map.insert(v.clone(), k.clone());
            }
        }
        map
    }
}

#[derive(Default, PartialEq, Eq, Clone, Debug)]
pub struct DisjointSet<T: Hash + Eq + Clone + Debug> {
    map: HashMap<T, T>,
    added: HashSet<T>,
}

impl<T: Hash + Eq + Clone + Debug> DisjointSet<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            added: HashSet::new(),
        }
    }

    pub fn using(to_add: impl IntoIterator<Item = T>) -> Self {
        let mut s: DisjointSet<T> = Self::new();
        for x in to_add {
            s.add(x);
        }
        s
    }

    fn find_root(&mut self, x: T) -> T {
        if let Some(parent) = self.map.get(&x) {
            let root: T = self.find_root(parent.clone());
            self.map.insert(x, root.clone()).unwrap();
            root
        } else {
            x
        }
    }

    pub fn add(&mut self, x: T) {
        self.added.insert(x);
    }

    /// Returns true if the set was changed, false if the items were already joined.
    pub fn join(&mut self, a: T, b: T) -> bool {
        self.add(a.clone());
        self.add(b.clone());
        let a: T = self.find_root(a);
        let b: T = self.find_root(b);
        if a == b {
            false
        } else {
            assert!(self.map.insert(a, b).is_none());
            true
        }
    }

    pub fn joined(&mut self, a: T, b: T) -> bool {
        self.find_root(a) == self.find_root(b)
    }

    pub fn all(&self) -> Vec<T> {
        self.added.cii().cv()
    }

    pub fn sets(&mut self) -> Vec<Vec<T>> {
        let mut seen_roots: MultiMap<T, T> = MultiMap::new();
        for x in self.all() {
            seen_roots.insert(self.find_root(x.clone()), x.clone());
        }
        seen_roots.cii().map(|(_, x)| x).collect_vec()
    }
}

pub fn grab_nums<const N: usize>(s: &str) -> [i64; N] {
    s.grab().ints().ca()
}

pub fn grab_unums<const N: usize>(s: &str) -> [usize; N] {
    s.grab().uints().ca()
}

pub trait DisplayExt: Display {
    fn grab(&self) -> String {
        self.to_string()
    }

    fn tos(&self) -> String {
        self.to_string()
    }

    #[track_caller]
    fn int(&self) -> i64 {
        self.to_string().trim().parse().unwrap_or_else(
            #[track_caller]
            |_| panic!("tried to parse {} as int", self),
        )
    }

    #[track_caller]
    fn uint(&self) -> usize {
        self.to_string().trim().parse().unwrap_or_else(
            #[track_caller]
            |_| panic!("tried to parse {} as uint", self),
        )
    }

    #[track_caller]
    fn velocity(&self) -> Vector2 {
        charvel(self.to_string().chars().next().unwrap())
    }

    fn ints(&self) -> Vec<i64> {
        self.to_string()
            .split(|c: char| !c.is_numeric() && c != '-')
            .filter_map(|x: &str| {
                if x.is_empty() {
                    None
                } else {
                    Some(x.trim().int())
                }
            })
            .collect_vec()
    }

    fn uints(&self) -> Vec<usize> {
        self.to_string()
            .split(|c: char| !c.is_numeric())
            .filter_map(|x: &str| {
                if x.is_empty() {
                    None
                } else {
                    Some(x.trim().uint())
                }
            })
            .collect_vec()
    }

    fn uints2(&self) -> Vec<i64> {
        self.to_string()
            .split(|c: char| !c.is_numeric())
            .filter_map(|x: &str| {
                if x.is_empty() {
                    None
                } else {
                    Some(x.trim().int())
                }
            })
            .collect_vec()
    }

    fn words(&self) -> Vec<String> {
        self.to_string()
            .split_whitespace()
            .map(|x: &str| x.to_string())
            .collect()
    }

    fn paragraphs(&self) -> Vec<String> {
        self.to_string()
            .split("\n\n")
            .map(|x: &str| x.to_string())
            .collect()
    }

    fn paras(&self) -> Vec<String> {
        self.paragraphs()
    }

    fn digits(&self) -> Vec<i64> {
        self.to_string()
            .chars()
            .filter_map(|x: char| x.to_digit(10).map(|x: u32| x as i64))
            .collect()
    }

    fn json<T: DeserializeOwned>(&self) -> T {
        serde_json::from_str(&self.to_string()).unwrap()
    }

    fn alphanumeric_words(&self) -> Vec<String> {
        self.to_string()
            .replace(|x: char| !x.is_alphanumeric(), "")
            .words()
    }

    fn regex(&self, regex: &str) -> Vec<Vec<String>> {
        #[cached]
        fn compile_regex(regex: String) -> Regex {
            Regex::new(&regex).unwrap()
        }

        let regex: Regex = compile_regex(regex.to_string());
        regex
            .captures_iter(&self.to_string())
            .map(|x: Captures| {
                x.iter()
                    .skip(1)
                    .map(|x: Option<Match>| x.unwrap().as_str().to_string())
                    .collect_vec()
            })
            .collect_vec()
    }
}

impl<T: Display> DisplayExt for T {}

#[derive(Default, Clone)]
pub struct OpaqueId {
    inner: Option<Arc<()>>,
}

impl OpaqueId {
    pub fn new() -> Self {
        Self {
            inner: Some(Arc::new(())),
        }
    }

    fn as_ptr(&self) -> *const () {
        self.inner.as_ref().map(Arc::as_ptr).unwrap_or(null())
    }

    fn as_num(&self) -> usize {
        self.as_ptr() as usize
    }
}

impl PartialEq for OpaqueId {
    fn eq(&self, other: &Self) -> bool {
        self.as_num() == other.as_num()
    }
}

impl Eq for OpaqueId {}

impl Ord for OpaqueId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_num().cmp(&other.as_num())
    }
}

impl PartialOrd for OpaqueId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for OpaqueId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_num().hash(state);
    }
}

pub fn fresh() -> usize {
    static STATE: AtomicUsize = AtomicUsize::new(0);
    STATE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub fn memo_dfs<N, FN, IN, FS>(start: N, mut successors: FN, mut success: FS) -> Option<Vec<N>>
where
    N: Eq + Hash + Clone,
    FN: FnMut(&N) -> IN,
    IN: IntoIterator<Item = N>,
    FS: FnMut(&N) -> bool,
{
    fn step<N, FN, IN, FS>(
        path: &mut Vec<N>,
        successors: &mut FN,
        success: &mut FS,
        cache: &mut FxHashSet<N>,
    ) -> bool
    where
        N: Eq + Hash + Clone,
        FN: FnMut(&N) -> IN,
        IN: IntoIterator<Item = N>,
        FS: FnMut(&N) -> bool,
    {
        if cache.contains(path.last().unwrap()) {
            return false;
        }
        if success(path.last().unwrap()) {
            true
        } else {
            let successors_it: IN = successors(path.last().unwrap());
            for n in successors_it {
                if !path.contains(&n) {
                    path.push(n);
                    if step(path, successors, success, cache) {
                        return true;
                    }
                    path.pop();
                }
            }
            cache.insert(path.last().cloned().unwrap());
            false
        }
    }

    let mut path: Vec<N> = vec![start];
    let mut cache: HashSet<N, BuildHasherDefault<FxHasher>> = FxHashSet::default();
    step(&mut path, &mut successors, &mut success, &mut cache).then_some(path)
}

pub fn factorize(n: i64) -> Vec<i64> {
    Factorization::run(n as u64)
        .factors
        .into_iter()
        .map(|x: u64| x as i64)
        .collect()
}

pub fn parse_grid<T: Clone>(
    s: &str,
    mut f: impl FnMut(char) -> T,
    default: T,
) -> DefaultHashMap<Cell2, T> {
    let mut grid: DefaultHashMap<Cell<2>, T> = DefaultHashMap::new(default);

    for (y, line) in s.lines().enumerate() {
        for (x, c) in line.chars().enumerate() {
            grid[c2(x as _, -(y as i64))] = f(c);
        }
    }

    grid
}

pub fn parse_hashset(s: &str, mut f: impl FnMut(char) -> bool) -> HashSet<Cell2> {
    let mut grid: HashSet<Cell<2>> = HashSet::new();

    for (y, line) in s.lines().enumerate() {
        for (x, c) in line.chars().enumerate() {
            if f(c) {
                grid.insert(c2(x as _, -(y as i64)));
            }
        }
    }

    grid
}

pub fn max<T: Ord>(items: impl IntoIterator<Item = T>) -> T {
    items.into_iter().max().unwrap()
}

pub fn min<T: Ord>(items: impl IntoIterator<Item = T>) -> T {
    items.into_iter().min().unwrap()
}

pub fn set<T: Hash + Eq>(items: impl IntoIterator<Item = T>) -> HashSet<T> {
    items.into_iter().collect()
}

pub fn defaultdict<K: Hash + Eq, V: Clone>(default: V) -> DefaultHashMap<K, V> {
    DefaultHashMap::new(default)
}

pub fn pause() {
    println!("Press enter to continue");
    io::stdin().read_line(&mut String::new()).unwrap();
}

pub trait BetterToOwned: ToOwned {
    fn to(&self) -> <Self as ToOwned>::Owned {
        self.to_owned()
    }
}

impl<T: ToOwned> BetterToOwned for T {}

pub fn ord(x: char) -> i64 {
    x as i64
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Robot<const N: usize> {
    pub pos: Cell<N>,
    pub vel: Vector<N>,
}

impl<const N: usize> Robot<N> {
    pub fn new(pos: Cell<N>, vel: Vector<N>) -> Self {
        Self { pos, vel }
    }

    pub fn tick(&mut self) {
        self.pos += self.vel;
    }
}

pub trait IsIn: PartialEq + Sized {
    fn is_in(&self, iter: impl IntoIterator<Item = Self>) -> bool {
        iter.into_iter().contains(self)
    }
}

impl<T: PartialEq> IsIn for T {}

pub fn bag<T: Eq + Hash>() -> DefaultHashMap<T, i64> {
    DefaultHashMap::new(0)
}

pub fn use_cycles<T, K: Eq + Hash>(
    mut state: T,
    mut next: impl FnMut(T) -> T,
    mut key_fn: impl FnMut(&T) -> K,
    count: usize,
) -> T {
    let mut cache: HashMap<K, usize> = HashMap::new();
    for i in 0..count {
        let key: K = key_fn(&state);
        if let Some(&v) = cache.get(&key) {
            let cycle_length: usize = i - v;
            let iters_left: usize = count - i;
            let iters_after_cycles: usize = iters_left % cycle_length;
            for _ in 0..iters_after_cycles {
                state = next(state);
            }
            return state;
        }
        cache.insert(key, i);
        state = next(state);
    }
    state
}

pub trait Serde: Serialize {
    fn serde<T: DeserializeOwned>(&self) -> T {
        serde_json::from_value(serde_json::to_value(&self).unwrap()).unwrap()
    }
}

impl<T: Serialize> Serde for T {}

pub trait DebugExt: Debug + Sized {
    fn dbg(self) -> Self {
        self.dbgr();
        self
    }

    fn dbgr(&self) {
        let bt: Backtrace = Backtrace::capture();
        let file: &BacktraceFrame = bt
            .frames()
            .iter()
            .find(|x: &&BacktraceFrame| format!("{x:?}").contains("/src/bin/"))
            .unwrap();
        let trace: String = format!("{:?}", file);
        let trace: Split<&str> = trace.split(", ");
        let [_, file, line] = trace.ca();
        let file: &str = file
            .split_once("/src/bin/")
            .unwrap()
            .1
            .strip_suffix("\"")
            .unwrap();
        let line: i64 = line.ints()[0];
        eprintln!("[src/bin/{}:{}] {:#?}", file, line, self);
    }

    fn cd(&self) -> Self
    where
        Self: Clone,
    {
        self.dbgr();
        self.clone()
    }

    fn to_debug_string(&self) -> String {
        format!("{self:?}")
    }
}

impl<T: Debug> DebugExt for T {}

pub fn bar() {
    let size: (Width, Height) = terminal_size::terminal_size().unwrap();
    eprintln!("{}", "⎯".repeat(size.0 .0 as usize))
}

pub fn unparse_grid(grid: &DefaultHashMap<Cell<2>, char>) -> String {
    let b: Cuboid<2> = bounds(grid.find(|x: char| x != grid.default));
    let mut s: String = String::new();
    let mut i: i64 = 0;
    for cell in b.cells() {
        s.push(grid[cell]);
        i += 1;
        if i == b.length(0) {
            i = 0;
            s.push('\n');
        }
    }
    s
}

pub fn zip<T, U>(a: impl IntoIterator<Item = T>, b: impl IntoIterator<Item = U>) -> Vec<(T, U)> {
    a.into_iter().zip(b.into_iter()).collect()
}
