use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rusty_kernel_tools::*;
use rusty_green_kernel::*;
use ndarray;
use num::complex::Complex;
use rand::Rng;

fn benchmark_laplace_assemble_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f64>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f64>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f64>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f64>());
    targets.map_inplace(|item| *item = rng.gen::<f64>());

    c.bench_function("laplace assemble double precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view())
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_laplace_assemble_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f32>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f32>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f32>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f32>());
    targets.map_inplace(|item| *item = rng.gen::<f32>());

    c.bench_function("laplace assemble single precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view())
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_laplace_evaluate_values_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("laplace evaluate values double precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view()).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_laplace_evalaute_values_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f32;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("laplace evaluate values single precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view()).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_laplace_evaluate_values_and_derivs_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("laplace evaluate values and derivs double precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view()).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_laplace_evaluate_values_and_derivs_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f32;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("laplace evaluate values and derivs single precision", |b| {
        b.iter(|| {
            make_laplace_evaluator(sources.view(), targets.view()).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_helmholtz_assemble_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let wavenumber = Complex::new(2.5, 0.0);

    type MyType = f32;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<Complex<MyType>>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("helmholtz assemble single precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber)
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_helmholtz_assemble_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let wavenumber = Complex::new(2.5, 0.0);

    type MyType = f64;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<Complex<MyType>>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("helmholtz assemble double precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber)
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_helmholtz_evaluate_values_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let wavenumber = Complex::new(2.5, 0.0);

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<Complex<MyType>>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<Complex<MyType>>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| {
        item.re = rng.gen::<MyType>();
        item.im = rng.gen::<MyType>();
    });

    c.bench_function("helmholtz evaluate values double precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_helmholtz_evaluate_values_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f32;

    let wavenumber = Complex::new(2.5, 0.0);

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<Complex<MyType>>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<Complex<MyType>>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| {
        item.re = rng.gen::<MyType>();
        item.im = rng.gen::<MyType>();
    });

    c.bench_function("helmholtz evaluate values single precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_helmholtz_evaluate_values_and_derivs_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let wavenumber = Complex::new(2.5, 0.0);

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<Complex<MyType>>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<Complex<MyType>>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| {
        item.re = rng.gen::<MyType>();
        item.im = rng.gen::<MyType>();
    });

    c.bench_function("helmholtz evaluate values and derivs double precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}


fn benchmark_helmholtz_evaluate_values_and_derivs_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f32;

    let wavenumber = Complex::new(2.5, 0.0);

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<Complex<MyType>>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<Complex<MyType>>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| {
        item.re = rng.gen::<MyType>();
        item.im = rng.gen::<MyType>();
    });

    c.bench_function("helmholtz evaluate values and derivs single precision", |b| {
        b.iter(|| {
            make_helmholtz_evaluator(sources.view(), targets.view(), wavenumber).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}


fn benchmark_modified_helmholtz_assemble_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let omega = 2.5;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f64>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f64>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f64>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f64>());
    targets.map_inplace(|item| *item = rng.gen::<f64>());

    c.bench_function("modified helmholtz assemble double precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega)
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_modified_helmholtz_assemble_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;

    let omega = 2.5;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<f32>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<f32>::zeros((3, ntargets));
    let mut result = ndarray::Array2::<f32>::zeros((ntargets, nsources));

    sources.map_inplace(|item| *item = rng.gen::<f32>());
    targets.map_inplace(|item| *item = rng.gen::<f32>());

    c.bench_function("modified helmholtz assemble single precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega)
                .assemble_in_place(black_box(result.view_mut()), ThreadingType::Parallel);
        })
    });
}

fn benchmark_modified_helmholtz_evaluate_values_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let omega = 2.5;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("modified helmholtz evaluate values double precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_modified_helmholtz_evaluate_values_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    let omega = 2.5;

    type MyType = f32;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 1));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("modified helmholtz evaluate values single precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::Value,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_modified_helmholtz_evaluate_values_and_derivs_double_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f64;

    let omega = 2.5;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("modified helmholtz evaluate values and derivs double precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}

fn benchmark_modified_helmholtz_evaluate_values_and_derivs_single_precision(c: &mut Criterion) {
    let nsources = 20000;
    let ntargets = 20000;
    let ncharge_vecs = 2;

    type MyType = f32;

    let omega = 2.5;

    let mut rng = rand::thread_rng();

    let mut sources = ndarray::Array2::<MyType>::zeros((3, nsources));
    let mut targets = ndarray::Array2::<MyType>::zeros((3, ntargets));
    let mut charges = ndarray::Array2::<MyType>::zeros((ncharge_vecs, nsources));
    let mut result = ndarray::Array3::<MyType>::zeros((ncharge_vecs, ntargets, 4));

    sources.map_inplace(|item| *item = rng.gen::<MyType>());
    targets.map_inplace(|item| *item = rng.gen::<MyType>());
    charges.map_inplace(|item| *item = rng.gen::<MyType>());

    c.bench_function("modified helmholtz evaluate values and derivs single precision", |b| {
        b.iter(|| {
            make_modified_helmholtz_evaluator(sources.view(), targets.view(), omega).evaluate_in_place(
                charges.view(),
                black_box(result.view_mut()),
                &EvalMode::ValueGrad,
                ThreadingType::Parallel,
            );
        })
    });
}



criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(30).measurement_time(std::time::Duration::from_secs(10));
    targets = benchmark_laplace_assemble_single_precision,
              benchmark_laplace_assemble_double_precision,
              benchmark_laplace_evalaute_values_single_precision,
              benchmark_laplace_evaluate_values_double_precision,
              benchmark_laplace_evaluate_values_and_derivs_single_precision,
              benchmark_laplace_evaluate_values_and_derivs_double_precision,
              benchmark_helmholtz_assemble_single_precision,
              benchmark_helmholtz_assemble_double_precision,
              benchmark_helmholtz_evaluate_values_double_precision,
              benchmark_helmholtz_evaluate_values_single_precision,
              benchmark_helmholtz_evaluate_values_and_derivs_double_precision,
              benchmark_helmholtz_evaluate_values_and_derivs_single_precision,
              benchmark_modified_helmholtz_assemble_single_precision,
              benchmark_modified_helmholtz_assemble_double_precision,
              benchmark_modified_helmholtz_evaluate_values_double_precision,
              benchmark_modified_helmholtz_evaluate_values_single_precision,
              benchmark_modified_helmholtz_evaluate_values_and_derivs_double_precision,
              benchmark_modified_helmholtz_evaluate_values_and_derivs_single_precision,


            }
criterion_main!(benches);
