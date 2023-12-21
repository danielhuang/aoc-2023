use aoc_2023::*;

fn main() {
    let input = load_input();

    let mut roots = vec![];
    let mut paths = FxHashMap::default();

    for line in input.lines() {
        if line.starts_with("broadcaster") {
            let right = line.split(" -> ").cv()[1];
            roots.extend(right.alphanumeric_words().cv());
        }
        if line.starts_with('%') || line.starts_with('&') {
            let (src, dest) = line.split_once(" -> ").unwrap();
            paths.insert(
                src.alphanumeric_words().one(),
                (src, dest.alphanumeric_words().cv()),
            );
        }
    }

    let mut flip_mem = defaultdict(false);
    let mut conj_mem = defaultdict(defaultdict(false));

    for (src, (_, dest)) in paths.clone() {
        let src = src.tos().alphanumeric_words().one();
        for dest in dest {
            conj_mem[dest][src.clone()] = false;
        }
    }

    let mut his = 0;
    let mut los = 0;

    let back1 = conj_mem["rx".tos()].keys().one().tos();
    let back2 = conj_mem[back1].keys().map(|x| x.tos()).cv();

    let mut cycles = defaultdict(vec![]);

    for i in 1i64.. {
        let mut queue = VecDeque::new();
        let mut history = vec![];

        los += 1;
        for root in roots.clone() {
            queue.push_back((root.tos(), false, "broadcaster".tos()));
        }

        while let Some((pulse_to, pulse, pulse_from)) = queue.pop_front() {
            history.push((pulse_to.tos(), pulse, pulse_from.tos()));
            if back2.contains(&pulse_to) && !pulse {
                cycles[pulse_to.tos()].push(i);
                if cycles.len() == 4 {
                    cp(cycles.values().map(|x| x[0]).fold(1, |a, b| a.lcm(&b)));
                    return;
                }
            }
            if pulse {
                his += 1;
            } else {
                los += 1;
            }
            if let Some((src, dest)) = paths.get(&pulse_to) {
                if src.starts_with('%') && !pulse {
                    flip_mem[pulse_to.tos()] = !flip_mem[pulse_to.tos()];
                    let to_send = flip_mem[pulse_to.tos()];
                    for dest in dest.clone() {
                        queue.push_back((dest.tos(), to_send, pulse_to.tos()));
                    }
                }
                if src.starts_with('&') {
                    conj_mem[pulse_to.tos()][pulse_from.tos()] = pulse;
                    let to_send = !conj_mem[pulse_to.tos()].values().all(|x| *x);
                    for dest in dest.clone() {
                        queue.push_back((dest.tos(), to_send, pulse_to.tos()));
                    }
                }
            }
        }

        if i == 1000 {
            cp(his * los);
        }
    }
}
