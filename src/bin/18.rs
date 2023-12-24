use aoc_2023::*;

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut corners = vec![];
        let mut pos = c2(0, 0);

        let mut edge_length = 0;

        for line in input.lines() {
            let words = line.alphanumeric_words();

            let (amount, dir) = if part2 {
                (
                    from_hex(&words[2][0..5]),
                    match line.alphanumeric_words()[2].trim().chars().last().unwrap() {
                        '0' => 'R',
                        '1' => 'D',
                        '2' => 'L',
                        '3' => 'U',
                        _ => unreachable!(),
                    },
                )
            } else {
                (words[1].int(), words[0].chars().next().unwrap())
            };

            edge_length += amount;

            pos += charvel(dir) * amount;
            corners.push(pos);
        }

        let double_area = corners
            .ii()
            .circular_tuple_windows()
            .map(|(a, b)| Matrix::new([a.vector(), b.vector()]).det())
            .sumi()
            .abs();

        cp(double_area / 2 + edge_length / 2 + 1);
    }
}

fn from_hex(x: &str) -> Z {
    Z::from_str_radix(x, 16).unwrap()
}
