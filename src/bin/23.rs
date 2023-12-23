use aoc_2023::*;

fn longest(
    grid: &DefaultHashMap<Cell2, char>,
    seen: &mut FxHashSet<Cell2>,
    start: Cell2,
    end: Cell2,
    paths: &FxHashMap<Cell2, Vec<(Cell2, usize)>>,
    part2: bool,
) -> i64 {
    if start == end {
        return 0;
    }

    let mut longest_dist = -999999;

    let tile = grid[start];
    assert!(tile != '#');

    let nexts = if part2 {
        paths[&start].clone()
    } else if tile == '.' {
        start
            .adj()
            .ii()
            .filter(|x| grid[x] != '#')
            .map(|x| (x, 1))
            .cv()
    } else {
        vec![(start + charvel(tile), 1)]
    };

    for (next, dist) in nexts {
        if !seen.insert(next) {
            continue;
        }
        let x = longest(grid, seen, next, end, paths, part2) + dist as i64;
        if x > longest_dist {
            longest_dist = x;
        }
        seen.remove(&next);
    }

    longest_dist
}

fn main() {
    let input = load_input();
    let grid = parse_grid(&input, |x| x, '#');

    let start = c2(1, 0);
    let end = grid.findv('.').ii().min_by_key(|x| x[1]).unwrap();

    let mut critical_points = HashSet::new();
    critical_points.insert(start);
    critical_points.insert(end);

    for place in grid.find(|x| x != '#') {
        let adj = place.adj().ii().filter(|x| grid[x] != '#').cv();
        if adj.len() > 2 {
            critical_points.insert(place);
        }
    }

    let mut paths = FxHashMap::default();
    for &place in critical_points.iter() {
        paths.insert(
            place,
            bfs_reach([place], |x| {
                if critical_points.contains(x) && *x != place {
                    vec![]
                } else {
                    x.adj().ii().filter(|x| grid[x] != '#').cv()
                }
            })
            .filter(|x| critical_points.contains(&x.0))
            .cv(),
        );
    }

    cp(longest(
        &grid,
        &mut Default::default(),
        start,
        end,
        &paths,
        false,
    ));

    cp(longest(
        &grid,
        &mut Default::default(),
        start,
        end,
        &paths,
        true,
    ));
}
