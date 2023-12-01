use aoc_2023::*;

fn main() {
    let input = load_input();

    let mut sum1 = 0;
    let mut sum2 = 0;

    let nums = [
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];

    for mut line in input.lines() {
        sum1 += line.digits()[0] * 10;
        sum1 += line.digits().last().unwrap();
        while !nums.cii().any(|x| line.starts_with(x)) && !line.chars().next().unwrap().is_numeric()
        {
            line = &line[1..];
        }
        while !nums.cii().any(|x| line.ends_with(x)) && !line.chars().last().unwrap().is_numeric() {
            line = &line[..line.len() - 1];
        }
        let mut a = 0;
        let mut b = 0;
        for (x, num) in nums.iter().enumerate() {
            if line.starts_with(num) {
                a = x + 1;
            }
            if line.ends_with(num) {
                b = x + 1;
            }
            if line.chars().next().unwrap().is_numeric() {
                a = line.digits()[0].uint();
            }
            if line.chars().last().unwrap().is_numeric() {
                b = line.digits()[line.digits().len() - 1].uint();
            }
        }
        sum2 += a * 10 + b;
    }

    cp(sum1);
    cp(sum2);
}
