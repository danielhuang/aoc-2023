use aoc_2023::*;

// [!] BAD CODE ALERT [!]

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut sum = 0;
        for line in input.lines() {
            let [mut unknown, known] = line.words().ca();
            let mut known = known.ints().cv();

            if part2 {
                unknown = format!("{}?", unknown)
                    .repeat(5)
                    .chars()
                    .cii()
                    .rev()
                    .skip(1)
                    .rev()
                    .cstr();
                known = known.repeat(5);
            }

            let cnt = count(unknown.chars().cv(), known);
            sum += cnt;
        }

        cp(sum);
    }
}

fn count(template: Vec<char>, counts: Vec<i64>) -> usize {
    count_paths(
        ("".tos(), template, counts),
        |(prev, remaining, counts)| {
            if remaining.is_empty() {
                return vec![];
            }
            let mut prev = prev.clone().trim_start_matches('.').tos();
            let mut counts = counts.clone();
            while prev.starts_with('#') && prev.contains('.') {
                prev = prev.trim_start_matches('.').tos();
                if !prev.is_empty() && prev.starts_with('#') {
                    let amount = prev.len() - prev.trim_start_matches('#').len();
                    if counts.is_empty() {
                        break;
                    }
                    if counts[0] != amount.int() {
                        break;
                    }
                    counts.remove(0);
                    prev = prev.trim_start_matches('#').tos();
                }
            }

            let mut results = vec![];
            let next = remaining[0];
            let a = format!("{}#", prev);
            let b = format!("{}.", prev);
            if next == '?' {
                if partial(&a, &counts) {
                    results.push((a, remaining[1..].to_vec(), counts.clone()));
                }
                if partial(&b, &counts) {
                    results.push((b, remaining[1..].to_vec(), counts.clone()));
                }
            } else if next == '#' {
                results.push((a, remaining[1..].to_vec(), counts.clone()));
            } else if next == '.' {
                results.push((b, remaining[1..].to_vec(), counts.clone()));
            }
            results.cset().cv()
        },
        |(prev, remaining, counts)| remaining.is_empty() && total(prev, counts),
    )
}

fn total(after: &str, known: &[i64]) -> bool {
    let after = after.split('.').filter(|x| !x.is_empty()).cv();
    if after.len() != known.len() {
        return false;
    }
    for i in 0..known.len() {
        if after[i].len() != known[i].uint() {
            return false;
        }
    }
    true
}

fn partial(after: &str, known: &[i64]) -> bool {
    assert!(!after.contains('?'));
    let mut known = known.to_vec();
    while !known.is_empty() {
        if total(after, &known) {
            return true;
        }
        *known.last_mut().unwrap() -= 1;
        while !known.is_empty() && known.last().copied().unwrap() == 0 {
            known.pop().unwrap();
        }
    }
    total(after, &known)
}
