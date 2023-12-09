use aoc_2023::*;

fn main() {
    let input = load_input();

    for part2 in [false, true] {
        let mut sum = 0;

        for line in input.lines() {
            let mut nums = line.ints();
            if part2 {
                nums.reverse();
            }
            let mut consts = vec![];
            while nums.cii().any(|x| x != 0) {
                nums = derivative(&nums);
                consts.push(nums.remove(0));
            }
            consts.push(0);
            for c in consts.cii().rev() {
                nums.insert(0, c);
                nums = integral(&nums);
            }
            sum += nums.last().copied().unwrap();
        }

        cp(sum);
    }
}
