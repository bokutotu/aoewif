#![allow(clippy::needless_raw_string_hashes)]

use aoewif::{
    CodegenError, ComparePredicate, ComputeFunctionBuilder, ComputeOpId, CpuSchedule, CudaBinding,
    CudaSchedule, CudaTarget, Dim, IndexExpr, IteratorId, ScalarLiteral, TensorType,
    VerifiedComputeFunction, generate_c, generate_cuda,
};

struct ElementwiseFixture {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    column: IteratorId,
}

fn elementwise_with_index_select() -> ElementwiseFixture {
    let mut builder = ComputeFunctionBuilder::new("choose_by_column").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let left = builder
        .input(
            "left",
            TensorType::f32(vec![Dim::Static(2), Dim::Symbol(columns)]),
        )
        .unwrap();
    let right = builder
        .input(
            "right",
            TensorType::f32(vec![Dim::Static(2), Dim::Symbol(columns)]),
        )
        .unwrap();

    let (column, output) = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Static(2)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let left_value = operation
            .read(
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let right_value = operation
            .read(
                right,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let column_index = operation.index(column).unwrap();
        let threshold = operation.constant(ScalarLiteral::Index(4));
        let condition = operation
            .compare(ComparePredicate::GreaterEqual, column_index, threshold)
            .unwrap();
        let value = operation
            .select(condition, left_value, right_value)
            .unwrap();
        (column, operation.finish(value).unwrap())
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    ElementwiseFixture {
        operation: function.operations()[0].id(),
        function,
        column,
    }
}

fn gemm() -> (VerifiedComputeFunction, ComputeOpId) {
    let mut builder = ComputeFunctionBuilder::new("gemm").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let reduction = builder.symbol("reduction").unwrap();
    let left = builder
        .input(
            "left",
            TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(reduction)]),
        )
        .unwrap();
    let right = builder
        .input(
            "right",
            TensorType::f32(vec![Dim::Symbol(reduction), Dim::Symbol(columns)]),
        )
        .unwrap();

    let output = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let inner = operation
            .reduction("inner", Dim::Symbol(reduction))
            .unwrap();
        let left_value = operation
            .read(
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(inner)],
            )
            .unwrap();
        let right_value = operation
            .read(
                right,
                vec![IndexExpr::Iterator(inner), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        let value = operation.fma(left_value, right_value, accumulator).unwrap();
        operation.finish(value).unwrap()
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = function.operations()[0].id();
    (function, operation)
}

struct CudaFixture {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    row: IteratorId,
    column: IteratorId,
}

fn cuda_elementwise() -> CudaFixture {
    let mut builder = ComputeFunctionBuilder::new("cuda_add").unwrap();
    let left = builder
        .input(
            "left",
            TensorType::f32(vec![Dim::Static(5), Dim::Static(10)]),
        )
        .unwrap();
    let right = builder
        .input(
            "right",
            TensorType::f32(vec![Dim::Static(5), Dim::Static(10)]),
        )
        .unwrap();

    let (row, column, output) = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Static(5)).unwrap();
        let column = operation.parallel("column", Dim::Static(10)).unwrap();
        let left_value = operation
            .read(
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let right_value = operation
            .read(
                right,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let value = operation.add(left_value, right_value).unwrap();
        (row, column, operation.finish(value).unwrap())
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    CudaFixture {
        operation: function.operations()[0].id(),
        function,
        row,
        column,
    }
}

fn two_operation_function() -> (VerifiedComputeFunction, ComputeOpId) {
    let mut builder = ComputeFunctionBuilder::new("two_operations").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();

    let intermediate = {
        let mut operation = builder.compute("intermediate").unwrap();
        let element = operation.parallel("element", Dim::Static(4)).unwrap();
        let value = operation
            .read(input, vec![IndexExpr::Iterator(element)])
            .unwrap();
        operation.finish(value).unwrap()
    };
    let output = {
        let mut operation = builder.compute("output").unwrap();
        let element = operation.parallel("element", Dim::Static(4)).unwrap();
        let value = operation
            .read(intermediate, vec![IndexExpr::Iterator(element)])
            .unwrap();
        operation.finish(value).unwrap()
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = function.operations()[1].id();
    (function, operation)
}

#[test]
fn generates_complete_cpu_elementwise_source_with_split_tail_and_abi() {
    let fixture = elementwise_with_index_select();
    let mut schedule = CpuSchedule::new(fixture.function, fixture.operation).unwrap();
    let column = schedule.loop_for(fixture.column).unwrap();
    schedule.split(column, 4).unwrap();
    let source = generate_c(&schedule.verify().unwrap()).unwrap();

    assert_eq!(source.function_name(), "choose_by_column");
    assert_eq!(source.source(), EXPECTED_CPU_ELEMENTWISE);
}

#[test]
fn generates_complete_cpu_strict_gemm_source() {
    let (function, operation) = gemm();
    let schedule = CpuSchedule::new(function, operation)
        .unwrap()
        .verify()
        .unwrap();
    let source = generate_c(&schedule).unwrap();

    assert_eq!(source.function_name(), "gemm");
    assert_eq!(source.source(), EXPECTED_CPU_GEMM);
}

#[test]
fn generates_complete_cuda_source_with_block_and_thread_bindings() {
    let fixture = cuda_elementwise();
    let mut schedule =
        CudaSchedule::new(fixture.function, fixture.operation, CudaTarget::default()).unwrap();
    let row = schedule.loop_for(fixture.row).unwrap();
    let column = schedule.loop_for(fixture.column).unwrap();
    let (row_outer, row_inner) = schedule.split(row, 2).unwrap();
    let (column_outer, column_inner) = schedule.split(column, 4).unwrap();
    schedule
        .reorder(&[row_outer, column_outer, row_inner, column_inner])
        .unwrap();
    schedule.bind(row_outer, CudaBinding::BlockY).unwrap();
    schedule.bind(column_outer, CudaBinding::BlockX).unwrap();
    schedule.bind(row_inner, CudaBinding::ThreadY).unwrap();
    schedule.bind(column_inner, CudaBinding::ThreadX).unwrap();
    let source = generate_cuda(&schedule.verify().unwrap()).unwrap();

    assert_eq!(source.kernel_name(), "cuda_add");
    assert_eq!(source.source(), EXPECTED_CUDA_ELEMENTWISE);
}

#[test]
fn rejects_multiple_operations_for_both_backends() {
    let (function, operation) = two_operation_function();
    let cpu_schedule = CpuSchedule::new(function.clone(), operation)
        .unwrap()
        .verify()
        .unwrap();
    let cuda_schedule = CudaSchedule::new(function, operation, CudaTarget::default())
        .unwrap()
        .verify()
        .unwrap();
    assert_eq!(
        generate_c(&cpu_schedule),
        Err(CodegenError::ExactlyOneOperationRequired { actual: 2 })
    );
    assert_eq!(
        generate_cuda(&cuda_schedule),
        Err(CodegenError::ExactlyOneOperationRequired { actual: 2 })
    );
}

const EXPECTED_CPU_ELEMENTWISE: &str = r#"#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#pragma STDC FP_CONTRACT OFF

void choose_by_column(const float* input0, const float* input1, float* output, size_t symbol0) {
    for (size_t loop0 = 0; loop0 < 2; ++loop0) {
        for (size_t loop2 = 0; loop2 < ((symbol0) + 3u) / 4u; ++loop2) {
            for (size_t loop3 = 0; loop3 < 4; ++loop3) {
                if (((loop2 * 4u) + loop3) < symbol0) {
                    float scalar0 = input0[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];
                    float scalar1 = input1[((loop0) * symbol0 + ((loop2 * 4u) + loop3))];
                    size_t scalar2 = ((loop2 * 4u) + loop3);
                    size_t scalar3 = 4u;
                    bool scalar4 = (scalar2 >= scalar3);
                    float scalar5 = (scalar4 ? scalar0 : scalar1);
                    output[((loop0) * symbol0 + ((loop2 * 4u) + loop3))] = scalar5;
                }
            }
        }
    }
}
"#;

const EXPECTED_CPU_GEMM: &str = r#"#include <math.h>
#include <stdbool.h>
#include <stddef.h>

#pragma STDC FP_CONTRACT OFF

void gemm(const float* input0, const float* input1, float* output, size_t symbol0, size_t symbol1, size_t symbol2) {
    for (size_t loop0 = 0; loop0 < symbol0; ++loop0) {
        for (size_t loop1 = 0; loop1 < symbol1; ++loop1) {
            float accumulator = 0.0f;
            for (size_t loop2 = 0; loop2 < symbol2; ++loop2) {
                float scalar0 = input0[((loop0) * symbol2 + loop2)];
                float scalar1 = input1[((loop2) * symbol1 + loop1)];
                float scalar3 = fmaf(scalar0, scalar1, accumulator);
                accumulator = scalar3;
            }
            output[((loop0) * symbol1 + loop1)] = accumulator;
        }
    }
}
"#;

const EXPECTED_CUDA_ELEMENTWISE: &str = r#"#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>

__global__ void cuda_add(const float* input0, const float* input1, float* output) {
    if (((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)) < 10 && ((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y)) < 5) {
        float scalar0 = input0[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];
        float scalar1 = input1[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))];
        float scalar2 = __fadd_rn(scalar0, scalar1);
        output[((((((size_t)blockIdx.y) * 2u) + ((size_t)threadIdx.y))) * 10 + ((((size_t)blockIdx.x) * 4u) + ((size_t)threadIdx.x)))] = scalar2;
    }
}
"#;
