use std::collections::HashMap;

use crate::ir::{IndexExpr, IteratorKind, ScalarArgumentKind, ScalarOperationKind, ScalarValueId};
use crate::schedule::{LoopId, VerifiedCpuSchedule};

use super::{
    CSource, CodegenError, SourceBuilder, compare_operator, flattened_index, input_name,
    logical_index_expression, loop_extent_expression, scalar_literal, scalar_name,
    scalar_type_name, tail_condition, validate_codegen_function,
};

pub fn generate_c(schedule: &VerifiedCpuSchedule) -> Result<CSource, CodegenError> {
    let function = schedule.function();
    let operation = schedule.operation();
    let plan = schedule.plan();
    validate_codegen_function(function, operation)?;

    let mut source = SourceBuilder::new();
    source.line("#include <math.h>");
    source.line("#include <stdbool.h>");
    source.line("#include <stddef.h>");
    source.blank_line();
    source.line("#pragma STDC FP_CONTRACT OFF");
    source.blank_line();

    let mut parameters = function
        .input_tensors()
        .map(|tensor| match tensor.definition() {
            crate::ir::TensorDefinition::Input { input_index } => {
                format!("const float* input{input_index}")
            }
            crate::ir::TensorDefinition::ComputeResult { .. } => unreachable!(),
        })
        .collect::<Vec<_>>();
    parameters.push("float* output".to_owned());
    parameters.extend(
        function
            .symbols()
            .iter()
            .map(|symbol| format!("size_t symbol{}", symbol.id().index())),
    );
    source.open(format!(
        "void {}({}) {{",
        function.name(),
        parameters.join(", ")
    ));

    let loop_value = |loop_id: LoopId| -> Result<String, CodegenError> {
        plan.loop_axis(loop_id)
            .map(|_| format!("loop{}", loop_id.index()))
            .ok_or(CodegenError::InvalidLoopPlan)
    };

    let parallel_loops = plan
        .loops()
        .iter()
        .filter(|loop_axis| loop_axis.kind() == IteratorKind::Parallel)
        .collect::<Vec<_>>();
    let reduction_loops = plan
        .loops()
        .iter()
        .filter(|loop_axis| loop_axis.kind() == IteratorKind::Reduction)
        .collect::<Vec<_>>();

    for loop_axis in &parallel_loops {
        let loop_name = loop_value(loop_axis.id())?;
        source.open(format!(
            "for (size_t {loop_name} = 0; {loop_name} < {}; ++{loop_name}) {{",
            loop_extent_expression(loop_axis.extent())
        ));
    }

    let tail = tail_condition(plan, &loop_value)?;
    if let Some(condition) = &tail {
        source.open(format!("if ({condition}) {{"));
    }

    let body_result = if let Some(init) = operation.init() {
        source.line(format!("float accumulator = {};", scalar_literal(init)));
        for loop_axis in &reduction_loops {
            let loop_name = loop_value(loop_axis.id())?;
            source.open(format!(
                "for (size_t {loop_name} = 0; {loop_name} < {}; ++{loop_name}) {{",
                loop_extent_expression(loop_axis.extent())
            ));
        }
        let result = emit_scalar_body(&mut source, schedule, &loop_value)?;
        source.line(format!("accumulator = {result};"));
        for _ in reduction_loops.iter().rev() {
            source.close("}");
        }
        "accumulator".to_owned()
    } else {
        emit_scalar_body(&mut source, schedule, &loop_value)?
    };

    let output_indices = operation
        .iterators()
        .iter()
        .filter(|iterator| iterator.kind() == IteratorKind::Parallel)
        .map(|iterator| logical_index_expression(plan, iterator.id(), &loop_value))
        .collect::<Result<Vec<_>, _>>()?;
    let output_type = function
        .tensor(operation.result())
        .ok_or(CodegenError::ScheduledOperationMismatch)?
        .tensor_type();
    source.line(format!(
        "output[{}] = {body_result};",
        flattened_index(output_type, &output_indices)
    ));

    if tail.is_some() {
        source.close("}");
    }
    for _ in parallel_loops.iter().rev() {
        source.close("}");
    }
    source.close("}");

    Ok(CSource {
        source: source.finish(),
        function_name: function.name().to_owned(),
    })
}

fn emit_scalar_body(
    source: &mut SourceBuilder,
    schedule: &VerifiedCpuSchedule,
    loop_value: &impl Fn(LoopId) -> Result<String, CodegenError>,
) -> Result<String, CodegenError> {
    let function = schedule.function();
    let operation = schedule.operation();
    let plan = schedule.plan();
    let mut values = HashMap::<ScalarValueId, String>::new();

    for argument in operation.body().arguments() {
        match argument.kind() {
            ScalarArgumentKind::InputElement { access_index } => {
                let access = &operation.accesses()[*access_index];
                let tensor =
                    function
                        .tensor(access.tensor())
                        .ok_or(CodegenError::NonInputTensorAccess {
                            tensor_index: access.tensor().index(),
                        })?;
                let indices = access
                    .indices()
                    .iter()
                    .map(|index| match index {
                        IndexExpr::Iterator(iterator) => {
                            logical_index_expression(plan, *iterator, loop_value)
                        }
                        IndexExpr::Constant(index) => Ok(format!("{index}u")),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let name = scalar_name(argument.value());
                source.line(format!(
                    "float {name} = {}[{}];",
                    input_name(function, access.tensor())?,
                    flattened_index(tensor.tensor_type(), &indices)
                ));
                values.insert(argument.value(), name);
            }
            ScalarArgumentKind::Accumulator => {
                values.insert(argument.value(), "accumulator".to_owned());
            }
        }
    }

    for operation in operation.body().operations() {
        let expression = scalar_operation_expression(operation.kind(), plan, loop_value, &values)?;
        let name = scalar_name(operation.result());
        source.line(format!(
            "{} {name} = {expression};",
            scalar_type_name(operation.result_type())?
        ));
        values.insert(operation.result(), name);
    }

    values
        .get(&operation.body().result())
        .cloned()
        .ok_or(CodegenError::MissingScalarValue {
            scalar_index: operation.body().result().index(),
        })
}

fn scalar_operation_expression(
    operation: &ScalarOperationKind,
    plan: &crate::schedule::LoopPlan,
    loop_value: &impl Fn(LoopId) -> Result<String, CodegenError>,
    values: &HashMap<ScalarValueId, String>,
) -> Result<String, CodegenError> {
    let value = |id: ScalarValueId| {
        values
            .get(&id)
            .cloned()
            .ok_or(CodegenError::MissingScalarValue {
                scalar_index: id.index(),
            })
    };
    match operation {
        ScalarOperationKind::Constant(literal) => Ok(scalar_literal(*literal)),
        ScalarOperationKind::Index(iterator) => {
            logical_index_expression(plan, *iterator, loop_value)
        }
        ScalarOperationKind::Add { lhs, rhs } => {
            Ok(format!("({} + {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Sub { lhs, rhs } => {
            Ok(format!("({} - {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Mul { lhs, rhs } => {
            Ok(format!("({} * {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Div { lhs, rhs } => {
            Ok(format!("({} / {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Fma {
            lhs,
            rhs,
            accumulator,
        } => Ok(format!(
            "fmaf({}, {}, {})",
            value(*lhs)?,
            value(*rhs)?,
            value(*accumulator)?
        )),
        ScalarOperationKind::Min { lhs, rhs } => {
            Ok(format!("fminf({}, {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Max { lhs, rhs } => {
            Ok(format!("fmaxf({}, {})", value(*lhs)?, value(*rhs)?))
        }
        ScalarOperationKind::Exp { input } => Ok(format!("expf({})", value(*input)?)),
        ScalarOperationKind::Log { input } => Ok(format!("logf({})", value(*input)?)),
        ScalarOperationKind::Compare {
            predicate,
            lhs,
            rhs,
        } => Ok(format!(
            "({} {} {})",
            value(*lhs)?,
            compare_operator(*predicate),
            value(*rhs)?
        )),
        ScalarOperationKind::Select {
            condition,
            true_value,
            false_value,
        } => Ok(format!(
            "({} ? {} : {})",
            value(*condition)?,
            value(*true_value)?,
            value(*false_value)?
        )),
    }
}
