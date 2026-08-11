use aoewif::{
    ComparePredicate, ComputeFunctionBuilder, Dim, IndexExpr, IrError, IteratorKind,
    ReductionPolicy, ScalarArgumentKind, ScalarLiteral, ScalarOperationKind, ScalarType,
    TensorDefinition, TensorType,
};

#[test]
fn builds_elementwise_add() {
    let mut builder = ComputeFunctionBuilder::new("elementwise_add").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let shape = vec![Dim::Symbol(rows), Dim::Symbol(columns)];
    let left = builder
        .input("left", TensorType::f32(shape.clone()))
        .unwrap();
    let right = builder
        .input("right", TensorType::f32(shape.clone()))
        .unwrap();

    let (row, column, left_element, right_element, sum, output) = {
        let mut operation = builder.compute("sum").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let projection = vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)];
        let left_element = operation.read(left, projection.clone()).unwrap();
        let right_element = operation.read(right, projection).unwrap();
        let sum = operation.add(left_element, right_element).unwrap();
        let output = operation.finish(sum).unwrap();
        (row, column, left_element, right_element, sum, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(function.name(), "elementwise_add");
    assert_eq!(
        function
            .symbols()
            .iter()
            .map(|symbol| (symbol.id(), symbol.name()))
            .collect::<Vec<_>>(),
        vec![(rows, "rows"), (columns, "columns")]
    );
    assert_eq!(
        function
            .tensors()
            .iter()
            .map(|tensor| (
                tensor.name().to_owned(),
                tensor.tensor_type().clone(),
                tensor.definition().clone(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                "left".to_owned(),
                TensorType::f32(shape.clone()),
                TensorDefinition::Input { input_index: 0 },
            ),
            (
                "right".to_owned(),
                TensorType::f32(shape.clone()),
                TensorDefinition::Input { input_index: 1 },
            ),
            (
                "sum".to_owned(),
                TensorType::f32(shape),
                TensorDefinition::ComputeResult {
                    operation: operation.id(),
                },
            ),
        ]
    );
    assert_eq!(
        operation
            .iterators()
            .iter()
            .map(|iterator| (
                iterator.id(),
                iterator.name(),
                iterator.extent().clone(),
                iterator.kind(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (row, "row", Dim::Symbol(rows), IteratorKind::Parallel),
            (
                column,
                "column",
                Dim::Symbol(columns),
                IteratorKind::Parallel,
            ),
        ]
    );
    assert_eq!(
        operation
            .accesses()
            .iter()
            .map(|access| (access.tensor(), access.indices().to_vec(), access.scalar()))
            .collect::<Vec<_>>(),
        vec![
            (
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
                left_element,
            ),
            (
                right,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
                right_element,
            ),
        ]
    );
    assert_eq!(
        operation.body().operations()[0].kind(),
        &ScalarOperationKind::Add {
            lhs: left_element,
            rhs: right_element,
        }
    );
    assert_eq!(operation.body().result(), sum);
    assert_eq!(function.outputs(), &[output]);
}

#[test]
fn builds_vector_broadcast() {
    let mut builder = ComputeFunctionBuilder::new("broadcast_add").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let matrix = builder
        .input(
            "matrix",
            TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(columns)]),
        )
        .unwrap();
    let bias = builder
        .input("bias", TensorType::f32(vec![Dim::Symbol(columns)]))
        .unwrap();

    let output = {
        let mut operation = builder.compute("biased").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let matrix_element = operation
            .read(
                matrix,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let bias_element = operation
            .read(bias, vec![IndexExpr::Iterator(column)])
            .unwrap();
        let biased = operation.add(matrix_element, bias_element).unwrap();
        operation.finish(biased).unwrap()
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(
        operation
            .accesses()
            .iter()
            .map(|access| (access.tensor(), access.indices().to_vec()))
            .collect::<Vec<_>>(),
        vec![
            (
                matrix,
                vec![
                    IndexExpr::Iterator(operation.iterators()[0].id()),
                    IndexExpr::Iterator(operation.iterators()[1].id()),
                ],
            ),
            (
                bias,
                vec![IndexExpr::Iterator(operation.iterators()[1].id())],
            ),
        ]
    );
    assert_eq!(
        function.tensor(output).unwrap().tensor_type(),
        &TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(columns)])
    );
}

#[test]
fn builds_transpose_projection() {
    let mut builder = ComputeFunctionBuilder::new("transpose").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let input = builder
        .input(
            "input",
            TensorType::f32(vec![Dim::Symbol(columns), Dim::Symbol(rows)]),
        )
        .unwrap();

    let (row, column, output) = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let element = operation
            .read(
                input,
                vec![IndexExpr::Iterator(column), IndexExpr::Iterator(row)],
            )
            .unwrap();
        let output = operation.finish(element).unwrap();
        (row, column, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(
        operation.accesses()[0].indices(),
        &[IndexExpr::Iterator(column), IndexExpr::Iterator(row)]
    );
    assert_eq!(
        function.tensor(output).unwrap().tensor_type(),
        &TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(columns)])
    );
}

#[test]
fn builds_strict_reduction() {
    let mut builder = ComputeFunctionBuilder::new("row_sum").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let input = builder
        .input(
            "input",
            TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(columns)]),
        )
        .unwrap();

    let (row, column, element, accumulator, sum, output) = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.reduction("column", Dim::Symbol(columns)).unwrap();
        let element = operation
            .read(
                input,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        let sum = operation.add(accumulator, element).unwrap();
        let output = operation.finish(sum).unwrap();
        (row, column, element, accumulator, sum, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(
        operation
            .iterators()
            .iter()
            .map(|iterator| (iterator.id(), iterator.kind()))
            .collect::<Vec<_>>(),
        vec![
            (row, IteratorKind::Parallel),
            (column, IteratorKind::Reduction),
        ]
    );
    assert!(operation.has_reduction());
    assert_eq!(operation.init(), Some(ScalarLiteral::F32(0.0)));
    assert_eq!(operation.reduction_policy(), ReductionPolicy::Strict);
    assert_eq!(
        operation
            .body()
            .arguments()
            .iter()
            .map(|argument| (
                argument.value(),
                argument.scalar_type(),
                argument.kind().clone(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (
                element,
                ScalarType::F32,
                ScalarArgumentKind::InputElement { access_index: 0 },
            ),
            (
                accumulator,
                ScalarType::F32,
                ScalarArgumentKind::Accumulator,
            ),
        ]
    );
    assert_eq!(
        operation.body().operations()[0].kind(),
        &ScalarOperationKind::Add {
            lhs: accumulator,
            rhs: element,
        }
    );
    assert_eq!(operation.body().result(), sum);
    assert_eq!(
        function.tensor(output).unwrap().tensor_type(),
        &TensorType::f32(vec![Dim::Symbol(rows)])
    );
}

#[test]
fn builds_gemm() {
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

    let (row, column, inner, left_element, right_element, accumulator, result, output) = {
        let mut operation = builder.compute("output").unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let inner = operation
            .reduction("inner", Dim::Symbol(reduction))
            .unwrap();
        let left_element = operation
            .read(
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(inner)],
            )
            .unwrap();
        let right_element = operation
            .read(
                right,
                vec![IndexExpr::Iterator(inner), IndexExpr::Iterator(column)],
            )
            .unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        let result = operation
            .fma(left_element, right_element, accumulator)
            .unwrap();
        let output = operation.finish(result).unwrap();
        (
            row,
            column,
            inner,
            left_element,
            right_element,
            accumulator,
            result,
            output,
        )
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(
        operation
            .accesses()
            .iter()
            .map(|access| (access.tensor(), access.indices().to_vec(), access.scalar()))
            .collect::<Vec<_>>(),
        vec![
            (
                left,
                vec![IndexExpr::Iterator(row), IndexExpr::Iterator(inner)],
                left_element,
            ),
            (
                right,
                vec![IndexExpr::Iterator(inner), IndexExpr::Iterator(column)],
                right_element,
            ),
        ]
    );
    assert_eq!(
        operation.body().operations()[0].kind(),
        &ScalarOperationKind::Fma {
            lhs: left_element,
            rhs: right_element,
            accumulator,
        }
    );
    assert_eq!(operation.body().result(), result);
    assert_eq!(
        function.tensor(output).unwrap().tensor_type(),
        &TensorType::f32(vec![Dim::Symbol(rows), Dim::Symbol(columns)])
    );
}

#[test]
fn builds_batched_gemm() {
    let mut builder = ComputeFunctionBuilder::new("batched_gemm").unwrap();
    let batches = builder.symbol("batches").unwrap();
    let rows = builder.symbol("rows").unwrap();
    let columns = builder.symbol("columns").unwrap();
    let reduction = builder.symbol("reduction").unwrap();
    let left = builder
        .input(
            "left",
            TensorType::f32(vec![
                Dim::Symbol(batches),
                Dim::Symbol(rows),
                Dim::Symbol(reduction),
            ]),
        )
        .unwrap();
    let right = builder
        .input(
            "right",
            TensorType::f32(vec![
                Dim::Symbol(batches),
                Dim::Symbol(reduction),
                Dim::Symbol(columns),
            ]),
        )
        .unwrap();

    let (batch, row, column, inner, output) = {
        let mut operation = builder.compute("output").unwrap();
        let batch = operation.parallel("batch", Dim::Symbol(batches)).unwrap();
        let row = operation.parallel("row", Dim::Symbol(rows)).unwrap();
        let column = operation.parallel("column", Dim::Symbol(columns)).unwrap();
        let inner = operation
            .reduction("inner", Dim::Symbol(reduction))
            .unwrap();
        let left_element = operation
            .read(
                left,
                vec![
                    IndexExpr::Iterator(batch),
                    IndexExpr::Iterator(row),
                    IndexExpr::Iterator(inner),
                ],
            )
            .unwrap();
        let right_element = operation
            .read(
                right,
                vec![
                    IndexExpr::Iterator(batch),
                    IndexExpr::Iterator(inner),
                    IndexExpr::Iterator(column),
                ],
            )
            .unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        let result = operation
            .fma(left_element, right_element, accumulator)
            .unwrap();
        let output = operation.finish(result).unwrap();
        (batch, row, column, inner, output)
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operation = &function.operations()[0];

    assert_eq!(
        operation
            .iterators()
            .iter()
            .map(|iterator| iterator.kind())
            .collect::<Vec<_>>(),
        vec![
            IteratorKind::Parallel,
            IteratorKind::Parallel,
            IteratorKind::Parallel,
            IteratorKind::Reduction,
        ]
    );
    assert_eq!(
        operation
            .accesses()
            .iter()
            .map(|access| access.indices().to_vec())
            .collect::<Vec<_>>(),
        vec![
            vec![
                IndexExpr::Iterator(batch),
                IndexExpr::Iterator(row),
                IndexExpr::Iterator(inner),
            ],
            vec![
                IndexExpr::Iterator(batch),
                IndexExpr::Iterator(inner),
                IndexExpr::Iterator(column),
            ],
        ]
    );
    assert_eq!(
        function.tensor(output).unwrap().tensor_type(),
        &TensorType::f32(vec![
            Dim::Symbol(batches),
            Dim::Symbol(rows),
            Dim::Symbol(columns),
        ])
    );
}

#[test]
fn builds_index_compare_and_select() {
    let mut builder = ComputeFunctionBuilder::new("masked_values").unwrap();
    let length = builder.symbol("length").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Symbol(length)]))
        .unwrap();

    let (iterator, element, index, limit, condition, zero, selected, output) = {
        let mut operation = builder.compute("output").unwrap();
        let iterator = operation.parallel("index", Dim::Symbol(length)).unwrap();
        let element = operation
            .read(input, vec![IndexExpr::Iterator(iterator)])
            .unwrap();
        let index = operation.index(iterator).unwrap();
        let limit = operation.constant(ScalarLiteral::Index(4));
        let condition = operation
            .compare(ComparePredicate::Less, index, limit)
            .unwrap();
        let zero = operation.constant(ScalarLiteral::F32(0.0));
        let selected = operation.select(condition, element, zero).unwrap();
        let output = operation.finish(selected).unwrap();
        (
            iterator, element, index, limit, condition, zero, selected, output,
        )
    };
    builder.mark_output(output).unwrap();

    let function = builder.finish().unwrap();
    let operations = function.operations()[0]
        .body()
        .operations()
        .iter()
        .map(|operation| {
            (
                operation.result(),
                operation.result_type(),
                operation.kind().clone(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(
        operations,
        vec![
            (
                index,
                ScalarType::Index,
                ScalarOperationKind::Index(iterator),
            ),
            (
                limit,
                ScalarType::Index,
                ScalarOperationKind::Constant(ScalarLiteral::Index(4)),
            ),
            (
                condition,
                ScalarType::Bool,
                ScalarOperationKind::Compare {
                    predicate: ComparePredicate::Less,
                    lhs: index,
                    rhs: limit,
                },
            ),
            (
                zero,
                ScalarType::F32,
                ScalarOperationKind::Constant(ScalarLiteral::F32(0.0)),
            ),
            (
                selected,
                ScalarType::F32,
                ScalarOperationKind::Select {
                    condition,
                    true_value: element,
                    false_value: zero,
                },
            ),
        ]
    );
}

#[test]
fn chains_compute_results_through_tensor_ssa() {
    let mut builder = ComputeFunctionBuilder::new("chain").unwrap();
    let length = builder.symbol("length").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Symbol(length)]))
        .unwrap();

    let squared = {
        let mut operation = builder.compute("squared").unwrap();
        let iterator = operation.parallel("index", Dim::Symbol(length)).unwrap();
        let element = operation
            .read(input, vec![IndexExpr::Iterator(iterator)])
            .unwrap();
        let result = operation.mul(element, element).unwrap();
        operation.finish(result).unwrap()
    };
    let shifted = {
        let mut operation = builder.compute("shifted").unwrap();
        let iterator = operation.parallel("index", Dim::Symbol(length)).unwrap();
        let element = operation
            .read(squared, vec![IndexExpr::Iterator(iterator)])
            .unwrap();
        let one = operation.constant(ScalarLiteral::F32(1.0));
        let result = operation.add(element, one).unwrap();
        operation.finish(result).unwrap()
    };
    builder.mark_output(shifted).unwrap();

    let function = builder.finish().unwrap();
    let square_operation = &function.operations()[0];
    let shift_operation = &function.operations()[1];

    assert_eq!(
        function.tensor(squared).unwrap().definition(),
        &TensorDefinition::ComputeResult {
            operation: square_operation.id(),
        }
    );
    assert_eq!(
        function.tensor(shifted).unwrap().definition(),
        &TensorDefinition::ComputeResult {
            operation: shift_operation.id(),
        }
    );
    assert_eq!(shift_operation.accesses()[0].tensor(), squared);
    assert_eq!(
        function
            .input_tensors()
            .map(|tensor| tensor.id())
            .collect::<Vec<_>>(),
        vec![input]
    );
    assert_eq!(function.outputs(), &[shifted]);
}

#[test]
fn rejects_ids_owned_by_another_function() {
    let mut foreign = ComputeFunctionBuilder::new("foreign").unwrap();
    let foreign_symbol = foreign.symbol("foreign_extent").unwrap();
    let foreign_tensor = foreign
        .input("foreign_input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    let (foreign_iterator, foreign_scalar) = {
        let mut operation = foreign.compute("foreign_operation").unwrap();
        let iterator = operation.parallel("index", Dim::Static(4)).unwrap();
        let scalar = operation.constant(ScalarLiteral::F32(1.0));
        (iterator, scalar)
    };

    let mut local = ComputeFunctionBuilder::new("local").unwrap();
    assert_eq!(
        local.input(
            "foreign_shaped",
            TensorType::f32(vec![Dim::Symbol(foreign_symbol)]),
        ),
        Err(IrError::ForeignId { kind: "symbol" })
    );
    assert_eq!(
        local.mark_output(foreign_tensor),
        Err(IrError::ForeignId { kind: "tensor" })
    );
    let local_tensor = local
        .input("local_input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();

    let mut operation = local.compute("local_operation").unwrap();
    let local_iterator = operation.parallel("index", Dim::Static(4)).unwrap();
    assert_eq!(
        operation.read(foreign_tensor, vec![IndexExpr::Iterator(local_iterator)]),
        Err(IrError::ForeignId { kind: "tensor" })
    );
    assert_eq!(
        operation.read(local_tensor, vec![IndexExpr::Iterator(foreign_iterator)]),
        Err(IrError::ForeignId { kind: "iterator" })
    );
    assert_eq!(
        operation.index(foreign_iterator),
        Err(IrError::ForeignId { kind: "iterator" })
    );
    let local_scalar = operation.constant(ScalarLiteral::F32(2.0));
    assert_eq!(
        operation.add(local_scalar, foreign_scalar),
        Err(IrError::ForeignId { kind: "scalar" })
    );
}

#[test]
fn rejects_access_rank_mismatch() {
    let mut builder = ComputeFunctionBuilder::new("bad_rank").unwrap();
    let input = builder
        .input(
            "input",
            TensorType::f32(vec![Dim::Static(4), Dim::Static(8)]),
        )
        .unwrap();
    let mut operation = builder.compute("output").unwrap();
    let row = operation.parallel("row", Dim::Static(4)).unwrap();

    assert_eq!(
        operation.read(input, vec![IndexExpr::Iterator(row)]),
        Err(IrError::TensorRankMismatch {
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn rejects_access_dimension_mismatch() {
    let mut builder = ComputeFunctionBuilder::new("bad_dimension").unwrap();
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    let mut operation = builder.compute("output").unwrap();
    let index = operation.parallel("index", Dim::Static(8)).unwrap();

    assert_eq!(
        operation.read(input, vec![IndexExpr::Iterator(index)]),
        Err(IrError::DimensionMismatch)
    );
}

#[test]
fn rejects_invalid_constant_indices() {
    let mut builder = ComputeFunctionBuilder::new("bad_constant_index").unwrap();
    let length = builder.symbol("length").unwrap();
    let dynamic = builder
        .input("dynamic", TensorType::f32(vec![Dim::Symbol(length)]))
        .unwrap();
    let fixed = builder
        .input("fixed", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    let mut operation = builder.compute("output").unwrap();

    assert_eq!(
        operation.read(dynamic, vec![IndexExpr::Constant(0)]),
        Err(IrError::ConstantIndexRequiresStaticDimension)
    );
    assert_eq!(
        operation.read(fixed, vec![IndexExpr::Constant(4)]),
        Err(IrError::ConstantIndexOutOfBounds {
            index: 4,
            extent: 4,
        })
    );
}

#[test]
fn rejects_unsupported_and_mismatched_scalar_types() {
    let mut builder = ComputeFunctionBuilder::new("bad_types").unwrap();
    for (name, scalar_type) in [("bools", ScalarType::Bool), ("indices", ScalarType::Index)] {
        assert_eq!(
            builder.input(name, TensorType::new(vec![Dim::Static(4)], scalar_type)),
            Err(IrError::TensorElementTypeUnsupported { scalar_type })
        );
    }
    let input = builder
        .input("input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    let mut operation = builder.compute("output").unwrap();
    let iterator = operation.parallel("index", Dim::Static(4)).unwrap();
    let element = operation
        .read(input, vec![IndexExpr::Iterator(iterator)])
        .unwrap();
    let index = operation.index(iterator).unwrap();
    let other_index = operation.constant(ScalarLiteral::Index(0));

    assert_eq!(
        operation.add(index, element),
        Err(IrError::ScalarTypeMismatch {
            expected: ScalarType::F32,
            actual: ScalarType::Index,
        })
    );
    assert_eq!(
        operation.compare(ComparePredicate::Equal, index, element),
        Err(IrError::ScalarOperandsMustMatch)
    );
    assert_eq!(
        operation.select(element, element, element),
        Err(IrError::ScalarTypeMismatch {
            expected: ScalarType::Bool,
            actual: ScalarType::F32,
        })
    );
    let condition = operation
        .compare(ComparePredicate::Equal, index, other_index)
        .unwrap();
    assert_eq!(
        operation.select(condition, element, index),
        Err(IrError::ScalarOperandsMustMatch)
    );
    assert_eq!(
        operation.finish(condition),
        Err(IrError::TensorElementTypeUnsupported {
            scalar_type: ScalarType::Bool,
        })
    );
}

#[test]
fn rejects_invalid_reduction_contracts() {
    let mut builder = ComputeFunctionBuilder::new("bad_reductions").unwrap();

    {
        let mut operation = builder.compute("missing_init").unwrap();
        operation.reduction("index", Dim::Static(4)).unwrap();
        let value = operation.constant(ScalarLiteral::F32(1.0));
        assert_eq!(operation.finish(value), Err(IrError::ReductionInitRequired));
    }

    {
        let mut operation = builder.compute("init_without_reduction").unwrap();
        operation.parallel("index", Dim::Static(4)).unwrap();
        let accumulator = operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        assert_eq!(
            operation.finish(accumulator),
            Err(IrError::ReductionInitWithoutReduction)
        );
    }

    {
        let mut operation = builder.compute("duplicate_init").unwrap();
        operation.reduction("index", Dim::Static(4)).unwrap();
        operation.reduction_init(ScalarLiteral::F32(0.0)).unwrap();
        assert_eq!(
            operation.reduction_init(ScalarLiteral::F32(1.0)),
            Err(IrError::DuplicateAccumulator)
        );
    }

    {
        let mut operation = builder.compute("init_result_type_mismatch").unwrap();
        operation.reduction("index", Dim::Static(4)).unwrap();
        operation.reduction_init(ScalarLiteral::Index(0)).unwrap();
        let result = operation.constant(ScalarLiteral::F32(0.0));
        assert_eq!(
            operation.finish(result),
            Err(IrError::ScalarResultMustMatchInit)
        );
    }
}

#[test]
fn rejects_missing_duplicate_and_foreign_outputs() {
    let mut missing = ComputeFunctionBuilder::new("missing_output").unwrap();
    let input = missing
        .input("input", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    {
        let mut operation = missing.compute("computed").unwrap();
        let index = operation.parallel("index", Dim::Static(4)).unwrap();
        let element = operation
            .read(input, vec![IndexExpr::Iterator(index)])
            .unwrap();
        operation.finish(element).unwrap();
    }
    assert_eq!(missing.finish(), Err(IrError::OutputRequired));

    let mut duplicate = ComputeFunctionBuilder::new("duplicate_output").unwrap();
    let output = duplicate
        .input("output", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    duplicate.mark_output(output).unwrap();
    assert_eq!(
        duplicate.mark_output(output),
        Err(IrError::OutputAlreadyMarked {
            index: output.index(),
        })
    );

    let mut foreign = ComputeFunctionBuilder::new("foreign_output").unwrap();
    let foreign_output = foreign
        .input("output", TensorType::f32(vec![Dim::Static(4)]))
        .unwrap();
    assert_eq!(
        duplicate.mark_output(foreign_output),
        Err(IrError::ForeignId { kind: "tensor" })
    );
}
