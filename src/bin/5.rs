use aoc_2023::*;

fn main() {
    let input = load_input();

    let paras = input.paras();

    let mut maps = vec![];

    for para in paras.cii().skip(1) {
        let mut map = vec![];
        for line in para.lines().skip(1) {
            let [dest, src, len] = line.ints().ca();
            map.push([dest, src, len]);
        }
        maps.push(map);
    }

    let nums = input.lines().next().unwrap().ints();

    for part2 in [false, true] {
        let mut seeds = Intervals::default();
        if part2 {
            for (&start, &len) in nums.iter().tuples() {
                seeds.add(start, start + len);
            }
        } else {
            for &num in &nums {
                seeds.add_one(num);
            }
        };

        for map in maps.iter() {
            let mut next = Intervals::default();
            for &[dest, src, len] in map {
                let mut taken = seeds.take_range(src, src + len);
                taken.shift(dest - src);
                next.extend(&taken);
            }
            next.extend(&seeds);
            seeds = next;
        }

        cp(seeds.iter().next().unwrap());
    }
}
