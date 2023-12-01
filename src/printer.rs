use std::{collections::HashSet, fmt::Display};

use crate::defaultmap::DefaultHashMap;
use owo_colors::OwoColorize;

use crate::{
    bounds,
    cartesian::{c2, Cell2},
};

struct Layer {
    points: HashSet<Cell2>,
    c: char,
}

#[must_use = "won't print unless print() is called"]
pub struct Printer {
    background: DefaultHashMap<Cell2, char>,
    layers: Vec<Layer>,
}

impl Default for Printer {
    fn default() -> Self {
        Self {
            background: DefaultHashMap::new(' '),
            layers: Default::default(),
        }
    }
}

impl Printer {
    pub fn new(background: DefaultHashMap<Cell2, char>) -> Self {
        Self {
            background,
            layers: vec![],
        }
    }

    pub fn layer(mut self, c: impl Display, points: impl IntoIterator<Item = Cell2>) -> Self {
        self.layers.push(Layer {
            points: points.into_iter().collect(),
            c: c.to_string().chars().next().unwrap(),
        });
        self
    }

    pub fn point(self, c: impl Display, pos: Cell2) -> Self {
        self.layer(c, [pos])
    }

    pub fn print(self) {
        let mut all_points = HashSet::new();
        for layer in &self.layers {
            all_points.extend(layer.points.clone());
        }
        for &p in self.background.keys() {
            all_points.insert(p);
        }
        if all_points.is_empty() {
            println!("printer has nothing to print");
            return;
        }
        let b = bounds(all_points);
        for r in (b.min[1]..=b.max[1]).rev() {
            for c in b.min[0]..=b.max[0] {
                let pos = c2(c, r);
                let c = self
                    .layers
                    .iter()
                    .rev()
                    .find(|x| x.points.contains(&pos) && x.c != ' ')
                    .map(|x| x.c)
                    .unwrap_or(self.background[pos]);
                let display = self
                    .layers
                    .iter()
                    .rev()
                    .enumerate()
                    .find(|(_, x)| x.points.contains(&pos))
                    .map(|(i, _)| match i % 6 {
                        0 => c.blue().to_string(),
                        1 => c.red().to_string(),
                        2 => c.green().to_string(),
                        3 => c.yellow().to_string(),
                        4 => c.magenta().to_string(),
                        5 => c.purple().to_string(),
                        _ => unreachable!(),
                    })
                    .unwrap_or(self.background[pos].to_string());
                print!("{display}");
            }
            println!();
        }
        println!();
    }
}
