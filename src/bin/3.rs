use aoc_2023::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x, '.');

    for part2 in [false, true] {
        let mut sum = 0;

        let symbols = if part2 {
            grid.find(|x| x == '*')
        } else {
            grid.find(|x| !x.is_digit(10) && x != '.')
        };

        let all_digits = grid.find(|x| x.is_ascii_digit()).cset();
        let mut adjacent_numbers = defaultdict(vec![]);

        for pos in all_digits.clone() {
            let mut points = vec![];
            let mut x = pos;
            while grid[x].is_digit(10) {
                points.push(x);
                x = x.right(1);
            }

            if !all_digits.contains(&pos.left(1)) {
                for symbol in symbols.clone() {
                    if symbol.adj_diag().ii().any(|x| points.contains(&x)) {
                        adjacent_numbers[symbol].push(pos);
                    }
                }
            }
        }

        if part2 {
            for &gear in adjacent_numbers.keys() {
                let points = adjacent_numbers[gear].clone();
                if points.len() == 2 {
                    let mut prod = 1;
                    for mut x in points {
                        let mut digits = vec![];
                        while grid[x].is_digit(10) {
                            digits.push(grid[x]);
                            x = x.right(1);
                        }
                        prod *= digits.collect_string().int();
                    }
                    sum += prod;
                }
            }
        } else {
            let all_numbers = adjacent_numbers.values().flatten().copied().cset();
            for mut x in all_numbers {
                let mut digits = vec![];
                while grid[x].is_digit(10) {
                    digits.push(grid[x]);
                    x = x.right(1);
                }
                sum += digits.collect_string().int();
            }
        }

        cp(sum);
    }
}
