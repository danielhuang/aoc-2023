use aoc_2023::*;

fn main() {
    let input = load_input();

    let mut components = vec![].cset();

    let mut conns = defaultdict(vec![]);
    for line in input.lines().sorted_by_key(|x| x.len()) {
        for a in line.alphanumeric_words() {
            components.insert(a);
        }
        let words = line.alphanumeric_words();
        for i in 1..words.len() {
            conns[words[0].tos()].push(words[i].tos());
            conns[words[i].tos()].push(words[0].tos());
        }
    }

    let start = input.alphanumeric_words()[0].tos();

    'a: for end in components.iter() {
        if end == &start {
            continue;
        }
        let mut removed = FxHashSet::default();
        for i in 0..=3 {
            let path = bfs(
                [start.tos()],
                |from| {
                    conns[from]
                        .iter()
                        .filter(|to| !removed.contains(&(from.tos(), to.tos())))
                        .cloned()
                        .cv()
                },
                |x| x == end,
            );

            match path {
                Some(path) if i < 3 => {
                    for (a, b) in path.ii().tuple_windows() {
                        removed.insert((a.tos(), b.tos()));
                        removed.insert((b, a));
                    }
                }
                None if i == 3 => {}
                _ => {
                    continue 'a;
                }
            }
        }

        let [left, right] = [start.tos(), end.tos()].map(|x| {
            bfs_reach([x], |from| {
                conns[from]
                    .iter()
                    .filter(|to| !removed.contains(&(from.tos(), to.tos())))
                    .cloned()
                    .cv()
            })
            .map(|x| x.0)
            .cv()
        });

        if left.len() + right.len() != components.len() {
            continue;
        }

        cp(left.len() * right.len());

        return;
    }

    unreachable!()
}
