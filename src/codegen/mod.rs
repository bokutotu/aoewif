mod c;
mod cuda;

use std::error::Error;
use std::fmt;

use crate::ir::{
    ComparePredicate, ComputeOp, Dim, ScalarLiteral, ScalarType, ScalarValueId, TensorDefinition,
    TensorType, TensorValueId, VerifiedComputeFunction,
};
use crate::schedule::{LoopExtent, LoopId, LoopIndexExpr, LoopPlan};

pub use c::generate_c;
pub use cuda::generate_cuda;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CSource {
    source: String,
    function_name: String,
}

impl CSource {
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSource {
    source: String,
    kernel_name: String,
}

impl CudaSource {
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodegenError {
    ExactlyOneOperationRequired { actual: usize },
    ExactlyOneOutputRequired { actual: usize },
    ScheduledOperationIsNotOutput,
    ScheduledOperationMismatch,
    NonInputTensorAccess { tensor_index: usize },
    UnsupportedTensorElement,
    UnsupportedScalarType,
    MissingScalarValue { scalar_index: usize },
    MissingLogicalIndex { iterator_index: usize },
    InvalidLoopPlan,
}

impl fmt::Display for CodegenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExactlyOneOperationRequired { actual } => {
                write!(
                    formatter,
                    "code generation requires exactly one compute operation, got {actual}"
                )
            }
            Self::ExactlyOneOutputRequired { actual } => {
                write!(
                    formatter,
                    "code generation requires exactly one output, got {actual}"
                )
            }
            Self::ScheduledOperationIsNotOutput => {
                write!(
                    formatter,
                    "the scheduled operation result is not the function output"
                )
            }
            Self::ScheduledOperationMismatch => {
                write!(
                    formatter,
                    "the schedule does not refer to the function operation"
                )
            }
            Self::NonInputTensorAccess { tensor_index } => {
                write!(
                    formatter,
                    "tensor value {tensor_index} is not a function input"
                )
            }
            Self::UnsupportedTensorElement => write!(formatter, "only f32 tensors are supported"),
            Self::UnsupportedScalarType => write!(formatter, "unsupported scalar type"),
            Self::MissingScalarValue { scalar_index } => {
                write!(formatter, "missing scalar value {scalar_index}")
            }
            Self::MissingLogicalIndex { iterator_index } => {
                write!(
                    formatter,
                    "missing logical index for iterator {iterator_index}"
                )
            }
            Self::InvalidLoopPlan => write!(formatter, "invalid scheduled loop plan"),
        }
    }
}

impl Error for CodegenError {}

pub(super) struct SourceBuilder {
    source: String,
    indentation: usize,
}

impl SourceBuilder {
    pub(super) fn new() -> Self {
        Self {
            source: String::new(),
            indentation: 0,
        }
    }

    pub(super) fn line(&mut self, line: impl AsRef<str>) {
        for _ in 0..self.indentation {
            self.source.push_str("    ");
        }
        self.source.push_str(line.as_ref());
        self.source.push('\n');
    }

    pub(super) fn blank_line(&mut self) {
        self.source.push('\n');
    }

    pub(super) fn open(&mut self, line: impl AsRef<str>) {
        self.line(line);
        self.indentation += 1;
    }

    pub(super) fn close(&mut self, line: impl AsRef<str>) {
        self.indentation -= 1;
        self.line(line);
    }

    pub(super) fn finish(self) -> String {
        self.source
    }
}

pub(super) fn scalar_name(value: ScalarValueId) -> String {
    format!("scalar{}", value.index())
}

pub(super) fn scalar_type_name(scalar_type: ScalarType) -> Result<&'static str, CodegenError> {
    match scalar_type {
        ScalarType::F32 => Ok("float"),
        ScalarType::Bool => Ok("bool"),
        ScalarType::Index => Ok("size_t"),
    }
}

pub(super) fn scalar_literal(literal: ScalarLiteral) -> String {
    match literal {
        ScalarLiteral::F32(value) if value.is_nan() => "NAN".to_owned(),
        ScalarLiteral::F32(value) if value == f32::INFINITY => "INFINITY".to_owned(),
        ScalarLiteral::F32(value) if value == f32::NEG_INFINITY => "-INFINITY".to_owned(),
        ScalarLiteral::F32(value) => format!("{value:?}f"),
        ScalarLiteral::Bool(value) => value.to_string(),
        ScalarLiteral::Index(value) => format!("{value}u"),
    }
}

pub(super) fn compare_operator(predicate: ComparePredicate) -> &'static str {
    match predicate {
        ComparePredicate::Equal => "==",
        ComparePredicate::NotEqual => "!=",
        ComparePredicate::Less => "<",
        ComparePredicate::LessEqual => "<=",
        ComparePredicate::Greater => ">",
        ComparePredicate::GreaterEqual => ">=",
    }
}

pub(super) fn dim_expression(dim: &Dim) -> String {
    match dim {
        Dim::Static(extent) => extent.to_string(),
        Dim::Symbol(symbol) => format!("symbol{}", symbol.index()),
    }
}

pub(super) fn loop_extent_expression(extent: &LoopExtent) -> String {
    match extent {
        LoopExtent::Static(value) => value.to_string(),
        LoopExtent::Symbol(symbol) => format!("symbol{}", symbol.index()),
        LoopExtent::CeilDiv { dividend, divisor } => {
            let dividend = loop_extent_expression(dividend);
            format!("(({dividend}) + {}u) / {divisor}u", divisor - 1)
        }
    }
}

pub(super) fn loop_index_expression(
    expression: &LoopIndexExpr,
    loop_value: &impl Fn(LoopId) -> Result<String, CodegenError>,
) -> Result<String, CodegenError> {
    match expression {
        LoopIndexExpr::Loop(loop_id) => loop_value(*loop_id),
        LoopIndexExpr::Constant(value) => Ok(format!("{value}u")),
        LoopIndexExpr::Add { lhs, rhs } => Ok(format!(
            "({} + {})",
            loop_index_expression(lhs, loop_value)?,
            loop_index_expression(rhs, loop_value)?
        )),
        LoopIndexExpr::Mul { lhs, rhs } => Ok(format!(
            "({} * {})",
            loop_index_expression(lhs, loop_value)?,
            loop_index_expression(rhs, loop_value)?
        )),
    }
}

pub(super) fn logical_index_expression(
    plan: &LoopPlan,
    iterator: crate::ir::IteratorId,
    loop_value: &impl Fn(LoopId) -> Result<String, CodegenError>,
) -> Result<String, CodegenError> {
    let logical_index = plan
        .logical_index(iterator)
        .ok_or(CodegenError::MissingLogicalIndex {
            iterator_index: iterator.index(),
        })?;
    loop_index_expression(logical_index.expression(), loop_value)
}

pub(super) fn tail_condition(
    plan: &LoopPlan,
    loop_value: &impl Fn(LoopId) -> Result<String, CodegenError>,
) -> Result<Option<String>, CodegenError> {
    let mut predicates = Vec::new();
    for logical_index in plan.logical_indices() {
        for predicate in logical_index.tail_predicates() {
            predicates.push(format!(
                "{} < {}",
                loop_index_expression(predicate.index(), loop_value)?,
                loop_extent_expression(predicate.extent())
            ));
        }
    }
    predicates.sort();
    predicates.dedup();
    Ok((!predicates.is_empty()).then(|| predicates.join(" && ")))
}

pub(super) fn flattened_index(tensor_type: &TensorType, indices: &[String]) -> String {
    if indices.is_empty() {
        return "0u".to_owned();
    }
    let mut expression = indices[0].clone();
    for (dimension, index) in tensor_type
        .shape()
        .iter()
        .skip(1)
        .zip(indices.iter().skip(1))
    {
        expression = format!("(({expression}) * {} + {index})", dim_expression(dimension));
    }
    expression
}

pub(super) fn validate_codegen_function(
    function: &VerifiedComputeFunction,
    operation: &ComputeOp,
) -> Result<(), CodegenError> {
    if function.operations().len() != 1 {
        return Err(CodegenError::ExactlyOneOperationRequired {
            actual: function.operations().len(),
        });
    }
    if function.operations()[0].id() != operation.id() {
        return Err(CodegenError::ScheduledOperationMismatch);
    }
    if function.outputs().len() != 1 {
        return Err(CodegenError::ExactlyOneOutputRequired {
            actual: function.outputs().len(),
        });
    }
    if function.outputs()[0] != operation.result() {
        return Err(CodegenError::ScheduledOperationIsNotOutput);
    }
    for tensor in function.tensors() {
        if tensor.tensor_type().element() != ScalarType::F32 {
            return Err(CodegenError::UnsupportedTensorElement);
        }
    }
    Ok(())
}

pub(super) fn input_name(
    function: &VerifiedComputeFunction,
    tensor: TensorValueId,
) -> Result<String, CodegenError> {
    let tensor_value = function
        .tensor(tensor)
        .ok_or(CodegenError::NonInputTensorAccess {
            tensor_index: tensor.index(),
        })?;
    match tensor_value.definition() {
        TensorDefinition::Input { input_index } => Ok(format!("input{input_index}")),
        TensorDefinition::ComputeResult { .. } => Err(CodegenError::NonInputTensorAccess {
            tensor_index: tensor.index(),
        }),
    }
}
