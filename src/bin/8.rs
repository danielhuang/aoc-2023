use aoc_2023::*;

fn main() {
    let input = load_input();

    let [head, body] = input.paras().ca();

    let mut map = HashMap::new();
    for line in body.lines() {
        let [from, left, right] = line.alphanumeric_words().ca();
        map.insert(from, (left, right));
    }

    for part2 in [false, true] {
        let starts = if part2 {
            map.keys().filter(|&x| x.ends_with('A')).cloned().cv()
        } else {
            vec!["AAA".tos()]
        };

        let mut counts = vec![];
        for mut cur in starts.clone() {
            let mut count: Z = 0;
            for direction in head.chars().cycle() {
                if direction == 'L' {
                    cur = map[&cur].0.tos();
                } else {
                    cur = map[&cur].1.tos();
                }
                count += 1;
                if (cur == "ZZZ" && !part2) || (cur.ends_with('Z') && part2) {
                    break;
                }
            }
            counts.push(count);
        }

        cp(counts.ii().reduce(|a, b| a.lcm(&b)).unwrap());
    }
}
