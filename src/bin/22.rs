use aoc_2023::*;

fn main() {
    let input = load_input();

    let mut bricks = vec![];

    for line in input.lines() {
        let [a, b, c, x, y, z] = line.ints().ca();
        bricks.push(bounds([c3(a, b, c), c3(x, y, z)]));
    }

    bricks.sort_by_key(|x| x.min[2]);

    let mut uses = defaultdict(vec![]);
    let mut used_by = defaultdict(vec![]);

    loop {
        let mut changed = false;
        for i in 0..bricks.len() {
            let others = bricks
                .cii()
                .enumerate()
                .filter(|x| x.0 != i)
                .map(|x| x.1)
                .cv();
            loop {
                let brick = bricks[i];
                let moved_brick = brick + v3(0, 0, -1);
                if moved_brick.all_corner_cells().cii().any(|x| x[2] < 0) {
                    break;
                }
                if others.iter().any(|&x| moved_brick.intersect_cells(x)) {
                    break;
                }
                bricks[i] = moved_brick;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (i, upper) in bricks.iter().enumerate() {
        for (j, lower) in bricks.iter().enumerate() {
            if i != j {
                for cell in upper.cells() {
                    let below = cell + v3(0, 0, -1);
                    if lower.contains(below) {
                        uses[i].push(j);
                        used_by[j].push(i);
                    }
                }
            }
        }
    }

    let need_to_use = uses.keys().copied().cset();
    let base = (0..bricks.len())
        .filter(|x| !need_to_use.contains(x))
        .cset();

    let mut count = 0;
    let mut sum = 0;

    for removed in 0..bricks.len() {
        let mut base = base.clone();
        base.retain(|&x| x != removed);
        let mut all = bfs_reach(base, |x| used_by[x].cii().filter(|&x| x != removed))
            .map(|x| x.0)
            .cv();
        all.retain(|&x| x != removed);
        let expected = bricks.len() - 1;
        let missing = expected - all.len();
        if missing == 0 {
            count += 1;
        }
        sum += missing;
    }

    cp(count);
    cp(sum);
}
