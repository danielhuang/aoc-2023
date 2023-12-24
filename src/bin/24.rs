use aoc_2023::*;
use z3::ast::{Ast, Int, Real};

fn line(pos: Point3, vel: Vector3) -> (Q, Q) {
    let Point([x1, y1, _]) = pos;
    let Point([x2, y2, _]) = pos + vel;
    let a = Q::new(y2 - y1, x2 - x1);
    let b = Q::new(y2, 1) - a * Q::new(x2, 1);
    (a, b)
}

fn main() {
    let input = load_input();

    let mut stones = vec![];

    for line in input.lines() {
        let [a, b, c, x, y, z] = line.ints().ca();
        stones.push((p3(a, b, c), v3(x, y, z)));
    }

    let (min, max) = if DEBUG {
        (7, 27)
    } else {
        (200000000000000, 400000000000000)
    };

    let mut count = 0;

    for (stone1, stone2) in stones.cii().tuple_combinations() {
        let startx1 = Q::new(stone1.0[0], 1);
        let dir1 = stone1.1[0];
        let startx2 = Q::new(stone2.0[0], 1);
        let dir2 = stone2.1[0];

        assert!(dir1 != 0);
        assert!(dir2 != 0);

        let (m1, b1) = line(stone1.0, stone1.1);
        let (m2, b2) = line(stone2.0, stone2.1);
        if m1 != m2 {
            let x = (b2 - b1) / (m1 - m2);
            let y = m1 * x + b1;
            if x >= Q::new(min, 1)
                && x <= Q::new(max, 1)
                && y >= Q::new(min, 1)
                && y <= Q::new(max, 1)
                && ((x > startx1 && dir1 > 0) || (x < startx1 && dir1 < 0))
                && ((x > startx2 && dir2 > 0) || (x < startx2 && dir2 < 0))
            {
                count += 1;
            }
        }
    }

    cp(count);

    let cfg = z3::Config::new();
    let ctx = z3::Context::new(&cfg);
    let solver = z3::Solver::new(&ctx);

    let start_x = Real::new_const(&ctx, "start_x");
    let start_y = Real::new_const(&ctx, "start_y");
    let start_z = Real::new_const(&ctx, "start_z");
    let vel_x = Real::new_const(&ctx, "vel_x");
    let vel_y = Real::new_const(&ctx, "vel_y");
    let vel_z = Real::new_const(&ctx, "vel_z");

    for (pos, vel) in stones {
        let time = Real::fresh_const(&ctx, "time");
        solver.assert(&(&start_x + &time * &vel_x)._eq(
            &(Real::from_int(&Int::from_i64(&ctx, pos[0] as i64))
                + Real::from_int(&Int::from_i64(&ctx, vel[0] as i64)) * &time),
        ));
        solver.assert(&(&start_y + &time * &vel_y)._eq(
            &(Real::from_int(&Int::from_i64(&ctx, pos[1] as i64))
                + Real::from_int(&Int::from_i64(&ctx, vel[1] as i64)) * &time),
        ));
        solver.assert(&(&start_z + &time * &vel_z)._eq(
            &(Real::from_int(&Int::from_i64(&ctx, pos[2] as i64))
                + Real::from_int(&Int::from_i64(&ctx, vel[2] as i64)) * &time),
        ));
    }

    solver.check();
    let model = solver.get_model().unwrap();

    let mut sum = 0;
    for dim in [&start_x, &start_y, &start_z] {
        let (a, b) = model.get_const_interp(dim).unwrap().as_real().unwrap();
        let num = Q::new(a as _, b as _);
        sum += num.int();
    }

    cp(sum);
}
