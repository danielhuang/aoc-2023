use std::cmp::Reverse;

use aoc_2023::*;

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut sum = 0;
        let mut v = vec![];

        let bests = if part2 {
            "AKQT98765432J".chars().cv()
        } else {
            "AKQJT98765432".chars().cv()
        };

        for line in input.lines() {
            let [cards, bid] = line.words().ca();
            v.push((cards, bid.int()));
        }

        v.sort_by_cached_key(|(cards, bid)| {
            let rank = if part2 {
                let mut ranks = vec![];
                for best in bests.clone() {
                    let cards = cards
                        .chars()
                        .map(|x| if x == 'J' { best } else { x })
                        .cstr();
                    ranks.push(calc_rank(cards.into_bytes()));
                }
                max(ranks)
            } else {
                calc_rank(cards.clone().into_bytes())
            };

            (
                rank,
                Reverse(
                    cards
                        .chars()
                        .map(|x| bests.iter().position(|&y| x == y))
                        .cv(),
                ),
                bid.int(),
            )
        });

        for (i, (_, bid)) in v.ii().enumerate() {
            sum += (i as i64 + 1) * bid;
        }

        cp(sum);
    }
}

fn calc_rank(mut cards: Vec<u8>) -> i64 {
    cards.sort();
    let mut label_count = 0;
    let mut min_count = cards.len();
    let mut max_count = 0;
    for (count, _) in cards.into_iter().dedup_with_count() {
        min_count = min_count.min(count);
        max_count = max_count.max(count);
        label_count += 1;
    }
    match (label_count, min_count, max_count) {
        (1, _, _) => -1,
        (2, _, 4) => -2,
        (2, 2, 3) => -3,
        (3, 1, 3) => -4,
        (3, 1, 2) => -5,
        (4, _, _) => -6,
        (5, _, _) => -7,
        _ => unreachable!(),
    }
}
