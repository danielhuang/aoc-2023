use aoc_2023::*;

fn main() {
    let input = load_input();
    let input = input.lines().cv();

    let mut sum = 0;
    let mut copies = vec![1; input.len()];

    for (i, line) in input.ii().enumerate() {
        let (_, card) = line.split_once(": ").unwrap();
        let (winning, mine) = card.split_once(" | ").unwrap();
        let winning = winning.ints().cset();
        let mine = mine.ints();
        let mut score = 0;
        let mut count = 0;
        for num in mine {
            if winning.contains(&num) {
                count += 1;
                if score == 0 {
                    score = 1;
                } else {
                    score *= 2;
                }
            }
        }
        for j in (i + 1)..(i + count + 1) {
            copies[j] += copies[i];
        }
        if score > 0 {
            sum += score;
        }
    }

    cp(sum);
    cp(copies.ii().sumi());
}
