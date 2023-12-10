use aoc_2023::*;

// [!] BAD CODE ALERT [!]

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

    let mut big_grid = HashSet::new();
    for &cell in grid.keys() {
        let tile = grid[cell];
        let bigtile = cell * 3;
        if tile != '.' {
            big_grid.insert(bigtile);
        }
        if connects[cell].contains(&cell.up(1)) {
            big_grid.insert(bigtile.up(1));
        }
        if connects[cell].contains(&cell.down(1)) {
            big_grid.insert(bigtile.down(1));
        }
        if connects[cell].contains(&cell.left(1)) {
            big_grid.insert(bigtile.left(1));
        }
        if connects[cell].contains(&cell.right(1)) {
            big_grid.insert(bigtile.right(1));
        }
    }
    let big_bounds = bounds(big_grid.clone()).grow();
    let corner = big_bounds.corner_cells()[0];
    let outside = bfs_reach([corner], |x| {
        x.adj()
            .ii()
            .filter(|y| !big_grid.contains(y) && big_bounds.contains(*y))
    })
    .map(|x| x.0)
    .cset();

    let mut sets = DisjointSet::using(big_bounds.cells());
    for big_cell in big_bounds.cells() {
        if outside.contains(&big_cell) {
            continue;
        }
        for bigadj in big_cell.adj() {
            if !big_grid.contains(&bigadj) && !big_grid.contains(&big_cell) {
                sets.join(big_cell, bigadj);
            }
        }
    }
    let mut big_inside = sets.sets().ii().max_by_key(|x| x.len()).unwrap().cset();

    let mut junk = DisjointSet::using(big_grid.clone());
    for big_cell in big_grid.clone() {
        for adj in big_cell.adj().ii().filter(|x| big_grid.contains(x)) {
            junk.join(big_cell, adj);
        }
    }
    let not_junk = junk.sets().ii().max_by_key(|x| x.len()).unwrap().cset();

    big_inside.extend(
        bfs_reach(big_inside.clone(), |x| {
            x.adj().ii().filter(|x| !not_junk.contains(x))
        })
        .map(|x| x.0),
    );

    let mut smalls = HashSet::new();
    for &small in grid.keys() {
        let bigtile = small * 3;
        if big_inside.contains(&bigtile) {
            smalls.insert(small);
        }
    }

    cp(smalls.len())
}
