use aoc_2023::*;

fn main() {
    let input = load_input();

    let mut p1 = 0;
    let mut p2 = 0;

    for line in input.lines() {
        let (id, data) = line.split_once(": ").unwrap();
        let game = data.split("; ").cv();
        let mut blue = 0;
        let mut green = 0;
        let mut red = 0;
        for round in game {
            for cube in round.split(", ") {
                let amount = cube.ints()[0];
                let color = cube.split(" ").nth(1).unwrap();
                if color == "blue" {
                    blue = blue.max(amount);
                }
                if color == "red" {
                    red = red.max(amount);
                }
                if color == "green" {
                    green = green.max(amount);
                }
            }
        }
        if red <= 12 && green <= 13 && blue <= 14 {
            p1 += id.ints()[0];
        }
        p2 += green * red * blue;
    }

    cp(p1);
    cp(p2);
}
