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

    for part2 in [false, true] {
        let seeds = if part2 {
            let all_ints = input.uints2();
            let mut all_ints_extended = vec![];
            for &a in all_ints.iter() {
                for &b in all_ints.iter() {
                    all_ints_extended.push(a + b);
                    all_ints_extended.push(a - b);
                    all_ints_extended.push(a);
                    all_ints_extended.push(b);
                }
            }
            let all_ints: BTreeSet<_> = all_ints_extended.ii().collect();
            let mut seeds = vec![];
            for (a, b) in input.lines().next().unwrap().ints().ii().tuples() {
                let range = a..(a + b);
                seeds.extend(all_ints.range(range));
            }
            seeds
        } else {
            input.lines().next().unwrap().ints()
        };

        let mut cands = vec![];
        for mut seed in seeds {
            for map in maps.iter() {
                let mut new_seed = seed;
                for &[dest, src, len] in map {
                    if (src..(src + len)).contains(&seed) {
                        new_seed = seed - src + dest;
                    }
                }
                seed = new_seed;
            }
            cands.push(seed);
        }

        cp(min(cands));
    }
}
