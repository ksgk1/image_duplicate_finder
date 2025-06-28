use {
    criterion::{BenchmarkId, Criterion, criterion_group, criterion_main},
    image_duplicates::data::CorrelationEntry,
    std::hint::black_box,
};

const TILE_COUNT: usize = 16; // 4x4 tiles

fn generate_test_data() -> Vec<([f32; TILE_COUNT], [f32; TILE_COUNT])> {
    vec![
        // Test case 1: Identical data
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        // Test case 2: Horizontally mirrored (should have high correlation)
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 12.0, 11.0, 10.0, 9.0, 16.0, 15.0, 14.0, 13.0]),
        // Test case 3: Random-ish data (low correlation)
        ([1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0], [16.0, 12.0, 8.0, 4.0, 15.0, 11.0, 7.0, 3.0, 14.0, 10.0, 6.0, 2.0, 13.0, 9.0, 5.0, 1.0]),
        // Test case 4: All zeros (edge case)
        ([0.0; TILE_COUNT], [0.0; TILE_COUNT]),
        // Test case 5: Mixed values
        ([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5], [15.5, 14.5, 13.5, 12.5, 11.5, 10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5]),
    ]
}

fn bench_correlation_methods(c: &mut Criterion) {
    let test_data = generate_test_data();

    let mut group = c.benchmark_group("correlation_coefficient");

    for (i, (data_a, data_b)) in test_data.iter().enumerate() {
        let test_name = match i {
            0 => "identical",
            1 => "mirrored",
            2 => "random",
            3 => "zeros",
            4 => "mixed",
            _ => "other",
        };

        // Benchmark the non-SIMD implementation
        group.bench_with_input(BenchmarkId::new("non_simd", test_name), &(data_a, data_b), |b, (a, b_data)| {
            b.iter(|| CorrelationEntry::correlation_coefficient(black_box(a), black_box(b_data)));
        });

        // Benchmark the SIMD implementation (only if feature is enabled)
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", test_name), &(data_a, data_b), |b, (a, b_data)| {
            b.iter(|| CorrelationEntry::correlation_coefficient_simd(black_box(a), black_box(b_data)));
        });

        // Benchmark the dispatch method
        group.bench_with_input(BenchmarkId::new("dispatch", test_name), &(data_a, data_b), |b, (a, b_data)| {
            b.iter(|| CorrelationEntry::correlation_coefficient(black_box(a), black_box(b_data)));
        });
    }

    group.finish();
}

fn bench_bulk_operations(c: &mut Criterion) {
    let test_data = generate_test_data();

    let mut group = c.benchmark_group("bulk_correlation");

    // Test processing many correlation calculations (simulates real usage)
    let iterations = [100, 1000, 10000];

    for &iter_count in &iterations {
        group.bench_with_input(BenchmarkId::new("non_simd", iter_count), &iter_count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let data_idx = i % test_data.len();
                    let (a, b_data) = &test_data[data_idx];
                    black_box(CorrelationEntry::correlation_coefficient(black_box(a), black_box(b_data)));
                }
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", iter_count), &iter_count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let data_idx = i % test_data.len();
                    let (a, b_data) = &test_data[data_idx];
                    black_box(CorrelationEntry::correlation_coefficient_simd(black_box(a), black_box(b_data)));
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_correlation_methods, bench_bulk_operations);
criterion_main!(benches);
