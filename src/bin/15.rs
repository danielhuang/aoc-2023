use aoc_2023::*;

fn main() {
    let input = load_input();
    let mut p1 = 0;
    let mut p2 = 0;

    let mut boxes = vec![IndexMap::new(); 256];

    for step in input.trim().split(',') {
        let mut hash1 = 0;
        let mut hash2 = 0;

        for c in step.chars() {
            if c == '-' || c == '=' {
                hash2 = hash1;
            }
            hash1 += c as usize;
            hash1 *= 17;
            hash1 %= 256;
        }

        p1 += hash1;

        let label = step.chars().take_while(|x| x.is_alphabetic()).cstr();

        if step.contains('-') {
            boxes[hash2].shift_remove(&label);
        } else {
            let length = step.ints()[0];
            boxes[hash2].insert(label, length);
        }
    }

    for (num, inside) in boxes.ii().enumerate() {
        for (slot, (_, length)) in inside.ii().enumerate() {
            p2 += (num + 1) * (slot + 1) * (length as usize);
        }
    }

    cp(p1);
    cp(p2);
}
