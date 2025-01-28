use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

fn bench_dummy(c: &mut Criterion) {
    c.benchmark_group("bench_dummy")
        .throughput(Throughput::Elements(1))
        .bench_function("dummy", |bencher| {
            bencher.iter(|| {
                std::thread::sleep(std::time::Duration::from_millis(100));
            });
        });
}

criterion_group!(benches, bench_dummy);
criterion_main!(benches);
