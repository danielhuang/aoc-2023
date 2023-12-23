use aoc_2023::*;

fn adj(c: Cell2, grid: &DefaultHashMap<Cell<2>, char>, b: &Cuboid<2>, wrap: bool) -> Vec<Cell2> {
    c.adj()
        .ii()
        .filter(|x| grid[if wrap { b.wrap(*x) } else { *x }] == '.')
        .cv()
}

fn main() {
    let input = load_input();

    let mut grid: DefaultHashMap<Cell<2>, char> = parse_grid(&input, |x| x, '.');

    let b = bounds(grid.keys().cloned());
    let size = b.length(0) as usize;

    let start = grid.findv('S')[0];
    grid[start] = '.';

    cp(bfs_reach([start], |x| adj(*x, &grid, &b, false))
        .take_while(|x| x.1 <= 64)
        .filter(|x| x.1 % 2 == 0)
        .count());

    let mut amount_left = 26501365;
    let mut all = [start].cset();
    if amount_left % 2 == 1 {
        all = start.adj().ii().filter(|x| grid[*x] == '.').cset();
    }

    let mut init_iters = 0;
    while amount_left % size != 0 {
        init_iters += 1;
        amount_left -= 1;
    }

    let all = bfs_reach(all, |x| adj(*x, &grid, &b, true))
        .take_while(|x| x.1 <= init_iters)
        .filter(|x| x.1 % 2 == 0)
        .map(|x| x.0)
        .cset();

    let mut c = 0;
    let mut lens = vec![];

    for (len, (_, cost)) in bfs_reach(all, |x| adj(*x, &grid, &b, true))
        .take_while(|x| x.1 <= amount_left)
        .filter(|x| x.1 % 2 == 0)
        .enumerate()
    {
        if cost > c {
            lens.push(len as i64);
            c = cost;

            let lens_stepped = lens.cii().step_by(size).cv();
            let d = derivative(&lens_stepped);
            let mut dd = derivative(&d);
            if dd.len() > 2 && dd[dd.len() - 1] == dd[dd.len() - 2] {
                while dd.len() * size < amount_left / 2 + 1 {
                    dd.push(dd[dd.len() - 1]);
                }
                let i = integral(&dd);
                let ii = integral(&i);
                cp(ii[ii.len() - 1]);
                return;
            }
        }
    }
}
