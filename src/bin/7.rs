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
                    let cards = cards.replace('J', &best.tos());
                    ranks.push(calc_rank(&cards));
                }
                max(ranks)
            } else {
                calc_rank(cards)
            };

            (
                rank,
                Reverse(
                    cards
                        .chars()
                        .map(|x| bests.iter().position(|y| x.tos() == y.tos()))
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

fn calc_rank(cards: &str) -> i64 {
    let freqs = freqs(cards.chars());
    let label_count = cards.chars().cset().len();
    let (min_count, max_count) = freqs.values().copied().minmax().into_option().unwrap();
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
