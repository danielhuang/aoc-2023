use aoc_2023::*;

fn main() {
    let input = load_input();

    let grid = parse_grid(&input, |x| x, '.');
    let mut connects = DefaultHashMap::new(vec![]);

    let pos = grid.find(|x| x == 'S')[0];

    for pipe in grid.find(|x| x != '.' && x != 'S') {
        let val = grid[pipe];
        if val == '|' {
            connects[pipe].push(pipe.up(1));
            connects[pipe].push(pipe.down(1));
        }
        if val == '-' {
            connects[pipe].push(pipe.left(1));
            connects[pipe].push(pipe.right(1));
        }
        if val == 'L' {
            connects[pipe].push(pipe.up(1));
            connects[pipe].push(pipe.right(1));
        }
        if val == 'J' {
            connects[pipe].push(pipe.left(1));
            connects[pipe].push(pipe.up(1));
        }
        if val == '7' {
            connects[pipe].push(pipe.left(1));
            connects[pipe].push(pipe.down(1));
        }
        if val == 'F' {
            connects[pipe].push(pipe.down(1));
            connects[pipe].push(pipe.right(1));
        }
    }

    for adj in pos.adj() {
        if grid[adj] != '.' && connects[adj].contains(&pos) {
            connects[pos].push(adj);
        }
    }

    let all = bfs_reach([pos], |x| connects[x].clone()).cv();

    cp(all.iter().map(|x| x.1).max().unwrap());

    let all = all.ii().map(|x| x.0).cv();

    let mut pipe = vec![];
    pipe.push(all[0]);
    pipe.push(all[1]);
    let all = all.cset();
    let mut seen = pipe.clone().cset();
    loop {
        let next = connects[pipe[pipe.len() - 1]]
            .cii()
            .find(|x| all.contains(x) && !seen.contains(x));
        if let Some(next) = next {
            pipe.push(next);
            seen.insert(next);
        } else {
            break;
        }
    }

    let area = pipe
        .cii()
        .circular_tuple_windows()
        .map(|(a, b)| Matrix::new([a.vector(), b.vector()]))
        .map(|x| x.det())
        .sumi()
        .abs()
        / 2;

    let b = pipe.len() as i64;

    cp(area - (b / 2) + 1);
}
