#![allow(clippy::too_many_lines)]

use aoewif::{
    ComputeFunctionBuilder, ComputeOpId, CpuSchedule, CudaBinding, CudaDim3, CudaSchedule,
    CudaTarget, Dim, IndexExpr, IteratorId, IteratorKind, LoopExtent, LoopId, LoopIndexExpr,
    ScalarLiteral, ScheduleError, SymbolId, TensorType, VerifiedComputeFunction,
};

struct ParallelFixture {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    first: IteratorId,
    second: IteratorId,
}

struct DynamicFixture {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    iterator: IteratorId,
    extent: SymbolId,
}

struct ReductionFixture {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    row: IteratorId,
    first_reduction: IteratorId,
    second_reduction: IteratorId,
}

fn parallel_2d(first_extent: u64, second_extent: u64) -> ParallelFixture {
    let mut builder = ComputeFunctionBuilder::new("parallel_2d").unwrap();
    let input = builder
        .input(
            "input",
            TensorType::f32(vec![Dim::Static(first_extent), Dim::Static(second_extent)]),
        )
        .unwrap();

    let (first, second, output) = {
        let mut operation = builder.compute("output").unwrap();
        let first = operation
            .parallel("first", Dim::Static(first_extent))
            .unwrap();
        let second = operation
            .parallel("second", Dim::Static(second_extent))
            .unwrap();
        let value = operation
            .read(
                input,
                vec![IndexExpr::Iterator(first), IndexExpr::Iterator(second)],
            )
            .unwrap();
        let output = operation.finish(value).unwrap();
        (first, second, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    ParallelFixture {
        operation: function.operations()[0].id(),
        function,
        first,
        second,
    }
}

fn parallel_1d(extent: u64) -> ParallelFixture {
    let mut builder = ComputeFunctionBuilder::new("parallel_1d").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Static(extent)]))
        .unwrap();

    let (iterator, output) = {
        let mut operation = builder.compute("output").unwrap();
        let iterator = operation.parallel("element", Dim::Static(extent)).unwrap();
        let value = operation
            .read(input, vec![IndexExpr::Iterator(iterator)])
            .unwrap();
        let output = operation.finish(value).unwrap();
        (iterator, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    ParallelFixture {
        operation: function.operations()[0].id(),
        function,
        first: iterator,
        second: iterator,
    }
}

fn dynamic_parallel_1d() -> DynamicFixture {
    let mut builder = ComputeFunctionBuilder::new("dynamic_parallel_1d").unwrap();
    let extent = builder.symbol("extent").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Symbol(extent)]))
        .unwrap();

    let (iterator, output) = {
        let mut operation = builder.compute("output").unwrap();
        let iterator = operation.parallel("element", Dim::Symbol(extent)).unwrap();
        let value = operation
            .read(input, vec![IndexExpr::Iterator(iterator)])
            .unwrap();
        let output = operation.finish(value).unwrap();
        (iterator, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    DynamicFixture {
        operation: function.operations()[0].id(),
        function,
        iterator,
        extent,
    }
}

fn mixed_reduction() -> ReductionFixture {
    let mut builder = ComputeFunctionBuilder::new("mixed_reduction").unwrap();
    let input = builder
        .input(
            "input",
            TensorType::f32(vec![Dim::Static(4), Dim::Static(8), Dim::Static(6)]),
        )
        .unwrap();

    let (row, first_reduction, second_reduction, output) = {
        let mut operation = builder.compute("output").unwrap();
        let first_reduction = operation
            .reduction("first_reduction", Dim::Static(4))
            .unwrap();
        let row = operation.parallel("row", Dim::Static(8)).unwrap();
        let second_reduction = operation
            .reduction("second_reduction", Dim::Static(6))
            .unwrap();
        let value = operation
            .read(
                input,
                vec![
                    IndexExpr::Iterator(first_reduction),
                    IndexExpr::Iterator(row),
                    IndexExpr::Iterator(second_reduction),
                ],
            )
            .unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        let sum = operation.add(accumulator, value).unwrap();
        let output = operation.finish(sum).unwrap();
        (row, first_reduction, second_reduction, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    ReductionFixture {
        operation: function.operations()[0].id(),
        function,
        row,
        first_reduction,
        second_reduction,
    }
}

fn split_index(outer: LoopId, inner: LoopId, factor: u64) -> LoopIndexExpr {
    LoopIndexExpr::Add {
        lhs: Box::new(LoopIndexExpr::Mul {
            lhs: Box::new(LoopIndexExpr::Loop(outer)),
            rhs: Box::new(LoopIndexExpr::Constant(factor)),
        }),
        rhs: Box::new(LoopIndexExpr::Loop(inner)),
    }
}

#[test]
fn cpu_schedule_normalizes_parallel_loops_before_reductions() {
    let fixture = mixed_reduction();
    let schedule = CpuSchedule::new(fixture.function, fixture.operation)
        .unwrap()
        .verify()
        .unwrap();
    let row_loop = schedule.plan().loop_for(fixture.row).unwrap();
    let first_reduction_loop = schedule.plan().loop_for(fixture.first_reduction).unwrap();
    let second_reduction_loop = schedule.plan().loop_for(fixture.second_reduction).unwrap();

    assert_eq!(
        schedule
            .plan()
            .loops()
            .iter()
            .map(|loop_axis| (
                loop_axis.id(),
                loop_axis.source_iterator(),
                loop_axis.name(),
                loop_axis.extent().clone(),
                loop_axis.kind(),
                loop_axis.binding(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                row_loop,
                fixture.row,
                "row",
                LoopExtent::Static(8),
                IteratorKind::Parallel,
                None,
            ),
            (
                first_reduction_loop,
                fixture.first_reduction,
                "first_reduction",
                LoopExtent::Static(4),
                IteratorKind::Reduction,
                None,
            ),
            (
                second_reduction_loop,
                fixture.second_reduction,
                "second_reduction",
                LoopExtent::Static(6),
                IteratorKind::Reduction,
                None,
            ),
        ]
    );
    assert_eq!(
        schedule
            .plan()
            .logical_index(fixture.row)
            .unwrap()
            .expression(),
        &LoopIndexExpr::Loop(row_loop)
    );
    assert_eq!(schedule.operation().id(), fixture.operation);
}

#[test]
fn cpu_split_rewrites_the_logical_index_without_a_divisible_tail() {
    let fixture = parallel_2d(16, 7);
    let mut schedule = CpuSchedule::new(fixture.function, fixture.operation).unwrap();
    let first_loop = schedule.loop_for(fixture.first).unwrap();
    let second_loop = schedule.loop_for(fixture.second).unwrap();
    let (outer, inner) = schedule.split(first_loop, 4).unwrap();
    let schedule = schedule.verify().unwrap();

    assert_eq!(
        schedule
            .plan()
            .loops()
            .iter()
            .map(|loop_axis| (loop_axis.id(), loop_axis.name(), loop_axis.extent().clone(),))
            .collect::<Vec<_>>(),
        vec![
            (outer, "first_outer", LoopExtent::Static(4)),
            (inner, "first_inner", LoopExtent::Static(4)),
            (second_loop, "second", LoopExtent::Static(7)),
        ]
    );
    let logical_index = schedule.plan().logical_index(fixture.first).unwrap();
    assert_eq!(logical_index.expression(), &split_index(outer, inner, 4));
    assert!(logical_index.tail_predicates().is_empty());
    assert_eq!(schedule.plan().loop_for(fixture.first), None);
}

#[test]
fn cpu_split_adds_a_tail_predicate_for_a_partial_tile() {
    let fixture = parallel_2d(10, 7);
    let mut schedule = CpuSchedule::new(fixture.function, fixture.operation).unwrap();
    let first_loop = schedule.loop_for(fixture.first).unwrap();
    let (outer, inner) = schedule.split(first_loop, 4).unwrap();
    let schedule = schedule.verify().unwrap();
    let logical_index = schedule.plan().logical_index(fixture.first).unwrap();
    let predicate = &logical_index.tail_predicates()[0];

    assert_eq!(
        schedule.plan().loop_axis(outer).unwrap().extent(),
        &LoopExtent::Static(3)
    );
    assert_eq!(
        schedule.plan().loop_axis(inner).unwrap().extent(),
        &LoopExtent::Static(4)
    );
    assert_eq!(logical_index.expression(), &split_index(outer, inner, 4));
    assert_eq!(logical_index.tail_predicates().len(), 1);
    assert_eq!(predicate.index(), &split_index(outer, inner, 4));
    assert_eq!(predicate.extent(), &LoopExtent::Static(10));
}

#[test]
fn cpu_reorder_changes_the_physical_loop_order() {
    let fixture = parallel_2d(8, 16);
    let mut schedule = CpuSchedule::new(fixture.function, fixture.operation).unwrap();
    let first_loop = schedule.loop_for(fixture.first).unwrap();
    let second_loop = schedule.loop_for(fixture.second).unwrap();
    schedule.reorder(&[second_loop, first_loop]).unwrap();
    let schedule = schedule.verify().unwrap();

    assert_eq!(
        schedule
            .plan()
            .loops()
            .iter()
            .map(aoewif::LoopAxis::id)
            .collect::<Vec<_>>(),
        vec![second_loop, first_loop]
    );
    assert_eq!(
        schedule
            .plan()
            .logical_index(fixture.first)
            .unwrap()
            .expression(),
        &LoopIndexExpr::Loop(first_loop)
    );
}

#[test]
fn cuda_schedule_builds_block_and_thread_loop_plan() {
    let fixture = parallel_2d(128, 70);
    let target = CudaTarget::default();
    let mut schedule = CudaSchedule::new(fixture.function, fixture.operation, target).unwrap();
    let first = schedule.loop_for(fixture.first).unwrap();
    let second = schedule.loop_for(fixture.second).unwrap();
    let (first_outer, first_inner) = schedule.split(first, 16).unwrap();
    let (second_outer, second_inner) = schedule.split(second, 32).unwrap();
    schedule
        .reorder(&[first_outer, second_outer, first_inner, second_inner])
        .unwrap();
    schedule.bind(first_outer, CudaBinding::BlockY).unwrap();
    schedule.bind(second_outer, CudaBinding::BlockX).unwrap();
    schedule.bind(first_inner, CudaBinding::ThreadY).unwrap();
    schedule.bind(second_inner, CudaBinding::ThreadX).unwrap();
    let schedule = schedule.verify().unwrap();

    assert_eq!(schedule.target(), target);
    assert_eq!(
        schedule
            .plan()
            .loops()
            .iter()
            .map(|loop_axis| (
                loop_axis.id(),
                loop_axis.extent().clone(),
                loop_axis.binding(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                first_outer,
                LoopExtent::Static(8),
                Some(CudaBinding::BlockY),
            ),
            (
                second_outer,
                LoopExtent::Static(3),
                Some(CudaBinding::BlockX),
            ),
            (
                first_inner,
                LoopExtent::Static(16),
                Some(CudaBinding::ThreadY),
            ),
            (
                second_inner,
                LoopExtent::Static(32),
                Some(CudaBinding::ThreadX),
            ),
        ]
    );
    let second_index = schedule.plan().logical_index(fixture.second).unwrap();
    assert_eq!(
        second_index.expression(),
        &split_index(second_outer, second_inner, 32)
    );
    assert_eq!(second_index.tail_predicates().len(), 1);
    assert_eq!(
        second_index.tail_predicates()[0].extent(),
        &LoopExtent::Static(70)
    );
}

#[test]
fn cuda_schedule_allows_a_dynamic_grid_with_static_threads() {
    let fixture = dynamic_parallel_1d();
    let mut schedule =
        CudaSchedule::new(fixture.function, fixture.operation, CudaTarget::default()).unwrap();
    let element = schedule.loop_for(fixture.iterator).unwrap();
    let (outer, inner) = schedule.split(element, 32).unwrap();
    schedule.bind(outer, CudaBinding::BlockX).unwrap();
    schedule.bind(inner, CudaBinding::ThreadX).unwrap();
    let schedule = schedule.verify().unwrap();

    assert_eq!(
        schedule.plan().loop_axis(outer).unwrap().extent(),
        &LoopExtent::CeilDiv {
            dividend: Box::new(LoopExtent::Symbol(fixture.extent)),
            divisor: 32,
        }
    );
    assert_eq!(
        schedule.plan().loop_axis(inner).unwrap().extent(),
        &LoopExtent::Static(32)
    );
    assert_eq!(
        schedule
            .plan()
            .logical_index(fixture.iterator)
            .unwrap()
            .tail_predicates()[0]
            .extent(),
        &LoopExtent::Symbol(fixture.extent)
    );
}

#[test]
fn cuda_schedule_rejects_zero_target_limits() {
    let fixture = parallel_1d(1);
    let block = CudaDim3::new(1024, 1024, 64);
    let grid = CudaDim3::new(2_147_483_647, 65_535, 65_535);

    for (target, field) in [
        (CudaTarget::new(0, block, grid), "max threads per block"),
        (
            CudaTarget::new(1024, CudaDim3::new(0, 1024, 64), grid),
            "max block dimension x",
        ),
        (
            CudaTarget::new(1024, CudaDim3::new(1024, 0, 64), grid),
            "max block dimension y",
        ),
        (
            CudaTarget::new(1024, CudaDim3::new(1024, 1024, 0), grid),
            "max block dimension z",
        ),
        (
            CudaTarget::new(1024, block, CudaDim3::new(0, 65_535, 65_535)),
            "max grid dimension x",
        ),
        (
            CudaTarget::new(1024, block, CudaDim3::new(2_147_483_647, 0, 65_535)),
            "max grid dimension y",
        ),
        (
            CudaTarget::new(1024, block, CudaDim3::new(2_147_483_647, 65_535, 0)),
            "max grid dimension z",
        ),
    ] {
        assert_eq!(
            CudaSchedule::new(fixture.function.clone(), fixture.operation, target),
            Err(ScheduleError::InvalidCudaTargetLimit { field })
        );
    }
}

#[test]
fn split_rejects_zero_stale_and_foreign_loops() {
    let fixture = parallel_2d(16, 8);
    let mut first = CpuSchedule::new(fixture.function.clone(), fixture.operation).unwrap();
    let second = CpuSchedule::new(fixture.function, fixture.operation).unwrap();
    let original = first.loop_for(fixture.first).unwrap();
    let foreign = second.loop_for(fixture.first).unwrap();

    assert_eq!(
        first.split(original, 0),
        Err(ScheduleError::ZeroSplitFactor)
    );
    first.split(original, 4).unwrap();
    assert_eq!(
        first.split(original, 2),
        Err(ScheduleError::UnknownLoop { loop_id: original })
    );
    assert_eq!(
        first.split(foreign, 2),
        Err(ScheduleError::ForeignLoop { loop_id: foreign })
    );
}

#[test]
fn strict_reduction_loops_cannot_be_split_reordered_or_bound() {
    let fixture = mixed_reduction();

    let mut split_schedule = CpuSchedule::new(fixture.function.clone(), fixture.operation).unwrap();
    let first_reduction = split_schedule.loop_for(fixture.first_reduction).unwrap();
    assert_eq!(
        split_schedule.split(first_reduction, 2),
        Err(ScheduleError::ReductionSplitUnsupported {
            loop_name: "first_reduction".to_owned(),
        })
    );

    let mut reorder_schedule =
        CpuSchedule::new(fixture.function.clone(), fixture.operation).unwrap();
    let row = reorder_schedule.loop_for(fixture.row).unwrap();
    let first_reduction = reorder_schedule.loop_for(fixture.first_reduction).unwrap();
    let second_reduction = reorder_schedule.loop_for(fixture.second_reduction).unwrap();
    assert_eq!(
        reorder_schedule.reorder(&[row, second_reduction, first_reduction]),
        Err(ScheduleError::ReductionReorderUnsupported)
    );
    assert_eq!(
        reorder_schedule.reorder(&[first_reduction, row, second_reduction]),
        Err(ScheduleError::ParallelLoopInsideReduction)
    );

    let mut cuda_schedule =
        CudaSchedule::new(fixture.function, fixture.operation, CudaTarget::default()).unwrap();
    let first_reduction = cuda_schedule.loop_for(fixture.first_reduction).unwrap();
    assert_eq!(
        cuda_schedule.bind(first_reduction, CudaBinding::ThreadX),
        Err(ScheduleError::ReductionBindUnsupported {
            loop_name: "first_reduction".to_owned(),
        })
    );
}

#[test]
fn cuda_schedule_rejects_duplicate_bindings() {
    let fixture = parallel_2d(8, 16);
    let mut schedule =
        CudaSchedule::new(fixture.function, fixture.operation, CudaTarget::default()).unwrap();
    let first = schedule.loop_for(fixture.first).unwrap();
    let second = schedule.loop_for(fixture.second).unwrap();
    schedule.bind(first, CudaBinding::ThreadX).unwrap();

    assert_eq!(
        schedule.bind(first, CudaBinding::ThreadY),
        Err(ScheduleError::LoopAlreadyBound {
            loop_name: "first".to_owned(),
            binding: CudaBinding::ThreadX,
        })
    );
    assert_eq!(
        schedule.bind(second, CudaBinding::ThreadX),
        Err(ScheduleError::BindingAlreadyUsed {
            binding: CudaBinding::ThreadX,
            loop_name: "first".to_owned(),
        })
    );
}

#[test]
fn cuda_schedule_rejects_a_dynamic_thread_extent() {
    let fixture = dynamic_parallel_1d();
    let mut schedule =
        CudaSchedule::new(fixture.function, fixture.operation, CudaTarget::default()).unwrap();
    let element = schedule.loop_for(fixture.iterator).unwrap();
    schedule.bind(element, CudaBinding::ThreadX).unwrap();

    assert_eq!(
        schedule.verify(),
        Err(ScheduleError::DynamicCudaThreadExtent {
            loop_name: "element".to_owned(),
        })
    );
}

#[test]
fn cuda_schedule_enforces_block_and_grid_dimension_limits() {
    let fixture = parallel_1d(33);

    let mut thread_schedule = CudaSchedule::new(
        fixture.function.clone(),
        fixture.operation,
        CudaTarget::new(
            1024,
            CudaDim3::new(32, 1024, 64),
            CudaDim3::new(1024, 1024, 1024),
        ),
    )
    .unwrap();
    let element = thread_schedule.loop_for(fixture.first).unwrap();
    thread_schedule.bind(element, CudaBinding::ThreadX).unwrap();
    assert_eq!(
        thread_schedule.verify(),
        Err(ScheduleError::CudaDimensionExceeded {
            binding: CudaBinding::ThreadX,
            requested: 33,
            maximum: 32,
        })
    );

    let mut block_schedule = CudaSchedule::new(
        fixture.function,
        fixture.operation,
        CudaTarget::new(
            1024,
            CudaDim3::new(1024, 1024, 64),
            CudaDim3::new(32, 1024, 1024),
        ),
    )
    .unwrap();
    let element = block_schedule.loop_for(fixture.first).unwrap();
    block_schedule.bind(element, CudaBinding::BlockX).unwrap();
    assert_eq!(
        block_schedule.verify(),
        Err(ScheduleError::CudaDimensionExceeded {
            binding: CudaBinding::BlockX,
            requested: 33,
            maximum: 32,
        })
    );
}

#[test]
fn cuda_schedule_enforces_total_threads_per_block() {
    let fixture = parallel_2d(32, 16);
    let mut schedule = CudaSchedule::new(
        fixture.function,
        fixture.operation,
        CudaTarget::new(
            256,
            CudaDim3::new(1024, 1024, 64),
            CudaDim3::new(1024, 1024, 1024),
        ),
    )
    .unwrap();
    let first = schedule.loop_for(fixture.first).unwrap();
    let second = schedule.loop_for(fixture.second).unwrap();
    schedule.bind(first, CudaBinding::ThreadX).unwrap();
    schedule.bind(second, CudaBinding::ThreadY).unwrap();

    assert_eq!(
        schedule.verify(),
        Err(ScheduleError::CudaThreadsPerBlockExceeded {
            requested: 512,
            maximum: 256,
        })
    );
}
