use aoc_2023::*;

fn calc(grid: &DefaultHashMap<Cell<2>, char>, pos: Cell<2>, vel: Vector<2>) -> usize {
    let all = bfs_reach([(pos, vel)], |prev| {
        let (mut pos, mut vel) = *prev;
        let mut results = vec![];
        if grid[pos] == ' ' {
            return vec![];
        }
        if grid[pos] == '/' {
            vel = vel.flip_over_fwslash();
        }
        if grid[pos] == '\\' {
            vel = vel.flip_over_bkslash();
        }
        if grid[pos] == '|' && vel[0] != 0 {
            results.push((pos, vel.turn_right()));
            vel = vel.turn_left();
        }
        if grid[pos] == '-' && vel[1] != 0 {
            results.push((pos, vel.turn_left()));
            vel = vel.turn_right();
        }
        pos += vel;
        results.push((pos, vel));
        results
    })
    .cv();

    all.cii()
        .map(|x| x.0)
        .map(|x| x.0)
        .filter(|x| grid[x] != ' ')
        .cset()
        .len()
}

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x, ' ');

    cp(calc(&grid, c2(0, 0), v2(1, 0)));

    let b = bounds(grid.keys().cloned());
    let width = b.length(0);
    let height = b.length(1);

    let mut starts = vec![];
    for top in 0..width {
        starts.push((c2(top, 0), v2(0, -1)));
        starts.push((c2(top, -height + 1), v2(0, 1)));
    }
    for y in 0..height {
        starts.push((c2(0, -y), v2(1, 0)));
        starts.push((c2(width - 1, -y), v2(-1, 0)));
    }

    let m = starts
        .ii()
        .map(|start| calc(&grid, start.0, start.1))
        .max()
        .unwrap();

    cp(m);
}
