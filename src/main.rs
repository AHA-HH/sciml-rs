use sciml_rs::neural_operators::scripts::burgers::run_burgers;

fn main() {
    let start = std::time::Instant::now();
    run_burgers();

    println!("total time: {:.2?}", start.elapsed());
}