mod cpu;
mod cuda;

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ir::{
    ComputeOp, ComputeOpId, Dim, IteratorId, IteratorKind, SymbolId, VerifiedComputeFunction,
};

pub use cpu::{CpuSchedule, VerifiedCpuSchedule};
pub use cuda::{CudaDim3, CudaSchedule, CudaTarget, VerifiedCudaSchedule};

static NEXT_LOOP_PLAN_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopId {
    owner: u64,
    index: usize,
}

impl LoopId {
    fn new(owner: u64, index: usize) -> Self {
        Self { owner, index }
    }

    pub fn index(self) -> usize {
        self.index
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopExtent {
    Static(u64),
    Symbol(SymbolId),
    CeilDiv {
        dividend: Box<LoopExtent>,
        divisor: u64,
    },
}

impl LoopExtent {
    pub fn static_value(&self) -> Option<u64> {
        match self {
            Self::Static(value) => Some(*value),
            Self::Symbol(_) => None,
            Self::CeilDiv { dividend, divisor } => dividend
                .static_value()
                .map(|value| ceil_div(value, *divisor)),
        }
    }

    fn from_dim(dimension: &Dim) -> Self {
        match dimension {
            Dim::Static(value) => Self::Static(*value),
            Dim::Symbol(symbol) => Self::Symbol(*symbol),
        }
    }

    fn ceil_div(self, divisor: u64) -> Result<Self, ScheduleError> {
        match self {
            Self::Static(value) => Ok(Self::Static(ceil_div(value, divisor))),
            Self::CeilDiv {
                dividend,
                divisor: inner_divisor,
            } => {
                let divisor = inner_divisor.checked_mul(divisor).ok_or(
                    ScheduleError::ArithmeticOverflow {
                        context: "split loop divisor",
                    },
                )?;
                Ok(Self::CeilDiv { dividend, divisor })
            }
            dividend => Ok(Self::CeilDiv {
                dividend: Box::new(dividend),
                divisor,
            }),
        }
    }

    fn is_divisible_by(&self, divisor: u64) -> bool {
        self.static_value()
            .is_some_and(|extent| extent.is_multiple_of(divisor))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopIndexExpr {
    Loop(LoopId),
    Constant(u64),
    Add {
        lhs: Box<LoopIndexExpr>,
        rhs: Box<LoopIndexExpr>,
    },
    Mul {
        lhs: Box<LoopIndexExpr>,
        rhs: Box<LoopIndexExpr>,
    },
}

impl LoopIndexExpr {
    fn split_index(outer: LoopId, inner: LoopId, factor: u64) -> Self {
        Self::Add {
            lhs: Box::new(Self::Mul {
                lhs: Box::new(Self::Loop(outer)),
                rhs: Box::new(Self::Constant(factor)),
            }),
            rhs: Box::new(Self::Loop(inner)),
        }
    }

    fn replace_loop(&self, target: LoopId, replacement: &Self) -> Self {
        match self {
            Self::Loop(loop_id) if *loop_id == target => replacement.clone(),
            Self::Loop(loop_id) => Self::Loop(*loop_id),
            Self::Constant(value) => Self::Constant(*value),
            Self::Add { lhs, rhs } => Self::Add {
                lhs: Box::new(lhs.replace_loop(target, replacement)),
                rhs: Box::new(rhs.replace_loop(target, replacement)),
            },
            Self::Mul { lhs, rhs } => Self::Mul {
                lhs: Box::new(lhs.replace_loop(target, replacement)),
                rhs: Box::new(rhs.replace_loop(target, replacement)),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailPredicate {
    index: LoopIndexExpr,
    extent: LoopExtent,
}

impl TailPredicate {
    pub fn index(&self) -> &LoopIndexExpr {
        &self.index
    }

    pub fn extent(&self) -> &LoopExtent {
        &self.extent
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalIndex {
    iterator: IteratorId,
    expression: LoopIndexExpr,
    tail_predicates: Vec<TailPredicate>,
}

impl LogicalIndex {
    pub fn iterator(&self) -> IteratorId {
        self.iterator
    }

    pub fn expression(&self) -> &LoopIndexExpr {
        &self.expression
    }

    pub fn tail_predicates(&self) -> &[TailPredicate] {
        &self.tail_predicates
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudaBinding {
    BlockX,
    BlockY,
    BlockZ,
    ThreadX,
    ThreadY,
    ThreadZ,
}

impl CudaBinding {
    pub const fn is_block(self) -> bool {
        matches!(self, Self::BlockX | Self::BlockY | Self::BlockZ)
    }

    pub const fn is_thread(self) -> bool {
        matches!(self, Self::ThreadX | Self::ThreadY | Self::ThreadZ)
    }

    pub const fn dimension(self) -> CudaDimension {
        match self {
            Self::BlockX | Self::ThreadX => CudaDimension::X,
            Self::BlockY | Self::ThreadY => CudaDimension::Y,
            Self::BlockZ | Self::ThreadZ => CudaDimension::Z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudaDimension {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopAxis {
    id: LoopId,
    source_iterator: IteratorId,
    name: String,
    extent: LoopExtent,
    kind: IteratorKind,
    binding: Option<CudaBinding>,
}

impl LoopAxis {
    pub fn id(&self) -> LoopId {
        self.id
    }

    pub fn source_iterator(&self) -> IteratorId {
        self.source_iterator
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn extent(&self) -> &LoopExtent {
        &self.extent
    }

    pub fn kind(&self) -> IteratorKind {
        self.kind
    }

    pub fn binding(&self) -> Option<CudaBinding> {
        self.binding
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopPlan {
    owner: u64,
    next_loop_index: usize,
    loops: Vec<LoopAxis>,
    logical_indices: Vec<LogicalIndex>,
}

impl LoopPlan {
    pub fn loops(&self) -> &[LoopAxis] {
        &self.loops
    }

    pub fn logical_indices(&self) -> &[LogicalIndex] {
        &self.logical_indices
    }

    pub fn logical_index(&self, iterator: IteratorId) -> Option<&LogicalIndex> {
        self.logical_indices
            .iter()
            .find(|logical_index| logical_index.iterator == iterator)
    }

    pub fn loop_axis(&self, loop_id: LoopId) -> Option<&LoopAxis> {
        (loop_id.owner == self.owner)
            .then(|| self.loops.iter().find(|loop_axis| loop_axis.id == loop_id))
            .flatten()
    }

    pub fn loop_for(&self, iterator: IteratorId) -> Option<LoopId> {
        let mut matching = self
            .loops
            .iter()
            .filter(|loop_axis| loop_axis.source_iterator == iterator);
        let loop_id = matching.next()?.id;
        matching.next().is_none().then_some(loop_id)
    }

    fn new(operation: &ComputeOp) -> Self {
        let owner = NEXT_LOOP_PLAN_OWNER.fetch_add(1, Ordering::Relaxed);
        let mut plan = Self {
            owner,
            next_loop_index: 0,
            loops: Vec::with_capacity(operation.iterators().len()),
            logical_indices: Vec::with_capacity(operation.iterators().len()),
        };

        for kind in [IteratorKind::Parallel, IteratorKind::Reduction] {
            for iterator in operation
                .iterators()
                .iter()
                .filter(|iterator| iterator.kind() == kind)
            {
                let loop_id = plan.allocate_loop_id();
                plan.loops.push(LoopAxis {
                    id: loop_id,
                    source_iterator: iterator.id(),
                    name: iterator.name().to_owned(),
                    extent: LoopExtent::from_dim(iterator.extent()),
                    kind,
                    binding: None,
                });
            }
        }

        for iterator in operation.iterators() {
            let loop_id = plan
                .loops
                .iter()
                .find(|loop_axis| loop_axis.source_iterator == iterator.id())
                .expect("every compute iterator must have a normalized loop")
                .id;
            plan.logical_indices.push(LogicalIndex {
                iterator: iterator.id(),
                expression: LoopIndexExpr::Loop(loop_id),
                tail_predicates: Vec::new(),
            });
        }

        plan
    }

    fn split(&mut self, loop_id: LoopId, factor: u64) -> Result<(LoopId, LoopId), ScheduleError> {
        if factor == 0 {
            return Err(ScheduleError::ZeroSplitFactor);
        }
        let position = self.loop_position(loop_id)?;
        let original = self.loops[position].clone();
        if original.kind == IteratorKind::Reduction {
            return Err(ScheduleError::ReductionSplitUnsupported {
                loop_name: original.name,
            });
        }
        if original.binding.is_some() {
            return Err(ScheduleError::BoundLoopSplitUnsupported {
                loop_name: original.name,
            });
        }

        let outer_id = self.allocate_loop_id();
        let inner_id = self.allocate_loop_id();
        let outer_extent = original.extent.clone().ceil_div(factor)?;
        let outer = LoopAxis {
            id: outer_id,
            source_iterator: original.source_iterator,
            name: format!("{}_outer", original.name),
            extent: outer_extent,
            kind: original.kind,
            binding: None,
        };
        let inner = LoopAxis {
            id: inner_id,
            source_iterator: original.source_iterator,
            name: format!("{}_inner", original.name),
            extent: LoopExtent::Static(factor),
            kind: original.kind,
            binding: None,
        };
        self.loops.splice(position..=position, [outer, inner]);

        let replacement = LoopIndexExpr::split_index(outer_id, inner_id, factor);
        let logical_index = self
            .logical_indices
            .iter_mut()
            .find(|logical_index| logical_index.iterator == original.source_iterator)
            .expect("every loop must have a logical index");
        logical_index.expression = logical_index.expression.replace_loop(loop_id, &replacement);
        for predicate in &mut logical_index.tail_predicates {
            predicate.index = predicate.index.replace_loop(loop_id, &replacement);
        }
        if !original.extent.is_divisible_by(factor) {
            logical_index.tail_predicates.push(TailPredicate {
                index: replacement,
                extent: original.extent,
            });
        }

        Ok((outer_id, inner_id))
    }

    fn reorder(&mut self, order: &[LoopId]) -> Result<(), ScheduleError> {
        if order.len() != self.loops.len() {
            return Err(ScheduleError::IncompleteLoopOrder {
                expected: self.loops.len(),
                actual: order.len(),
            });
        }

        let mut reordered = Vec::with_capacity(order.len());
        for (position, loop_id) in order.iter().copied().enumerate() {
            if order[..position].contains(&loop_id) {
                return Err(ScheduleError::DuplicateLoop { loop_id });
            }
            reordered.push(self.loop_axis_checked(loop_id)?.clone());
        }

        let current_reductions = self
            .loops
            .iter()
            .filter(|loop_axis| loop_axis.kind == IteratorKind::Reduction)
            .map(|loop_axis| loop_axis.id);
        let reordered_reductions = reordered
            .iter()
            .filter(|loop_axis| loop_axis.kind == IteratorKind::Reduction)
            .map(|loop_axis| loop_axis.id);
        if !current_reductions.eq(reordered_reductions) {
            return Err(ScheduleError::ReductionReorderUnsupported);
        }

        if has_parallel_inside_reduction(&reordered) {
            return Err(ScheduleError::ParallelLoopInsideReduction);
        }

        self.loops = reordered;
        Ok(())
    }

    fn bind(&mut self, loop_id: LoopId, binding: CudaBinding) -> Result<(), ScheduleError> {
        let position = self.loop_position(loop_id)?;
        if self.loops[position].kind == IteratorKind::Reduction {
            return Err(ScheduleError::ReductionBindUnsupported {
                loop_name: self.loops[position].name.clone(),
            });
        }
        if let Some(existing) = self.loops[position].binding {
            return Err(ScheduleError::LoopAlreadyBound {
                loop_name: self.loops[position].name.clone(),
                binding: existing,
            });
        }
        if let Some(existing) = self
            .loops
            .iter()
            .find(|loop_axis| loop_axis.binding == Some(binding))
        {
            return Err(ScheduleError::BindingAlreadyUsed {
                binding,
                loop_name: existing.name.clone(),
            });
        }
        self.loops[position].binding = Some(binding);
        Ok(())
    }

    fn allocate_loop_id(&mut self) -> LoopId {
        let id = LoopId::new(self.owner, self.next_loop_index);
        self.next_loop_index += 1;
        id
    }

    fn loop_position(&self, loop_id: LoopId) -> Result<usize, ScheduleError> {
        if loop_id.owner != self.owner {
            return Err(ScheduleError::ForeignLoop { loop_id });
        }
        self.loops
            .iter()
            .position(|loop_axis| loop_axis.id == loop_id)
            .ok_or(ScheduleError::UnknownLoop { loop_id })
    }

    fn loop_axis_checked(&self, loop_id: LoopId) -> Result<&LoopAxis, ScheduleError> {
        let position = self.loop_position(loop_id)?;
        Ok(&self.loops[position])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    UnknownOperation {
        index: usize,
    },
    ForeignLoop {
        loop_id: LoopId,
    },
    UnknownLoop {
        loop_id: LoopId,
    },
    ZeroSplitFactor,
    ReductionSplitUnsupported {
        loop_name: String,
    },
    BoundLoopSplitUnsupported {
        loop_name: String,
    },
    IncompleteLoopOrder {
        expected: usize,
        actual: usize,
    },
    DuplicateLoop {
        loop_id: LoopId,
    },
    ReductionReorderUnsupported,
    ParallelLoopInsideReduction,
    ReductionBindUnsupported {
        loop_name: String,
    },
    LoopAlreadyBound {
        loop_name: String,
        binding: CudaBinding,
    },
    BindingAlreadyUsed {
        binding: CudaBinding,
        loop_name: String,
    },
    CpuBindingUnsupported {
        loop_name: String,
        binding: CudaBinding,
    },
    InvalidCudaTargetLimit {
        field: &'static str,
    },
    DynamicCudaThreadExtent {
        loop_name: String,
    },
    ZeroCudaLaunchDimension {
        binding: CudaBinding,
    },
    CudaDimensionExceeded {
        binding: CudaBinding,
        requested: u64,
        maximum: u64,
    },
    CudaThreadsPerBlockExceeded {
        requested: u64,
        maximum: u64,
    },
    ArithmeticOverflow {
        context: &'static str,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOperation { index } => {
                write!(formatter, "unknown compute operation {index}")
            }
            Self::ForeignLoop { loop_id } => write!(formatter, "foreign loop {}", loop_id.index()),
            Self::UnknownLoop { loop_id } => write!(formatter, "unknown loop {}", loop_id.index()),
            Self::ZeroSplitFactor => {
                write!(formatter, "loop split factor must be greater than zero")
            }
            Self::ReductionSplitUnsupported { loop_name } => write!(
                formatter,
                "splitting reduction loop `{loop_name}` is unsupported"
            ),
            Self::BoundLoopSplitUnsupported { loop_name } => write!(
                formatter,
                "bound loop `{loop_name}` must be split before CUDA binding"
            ),
            Self::IncompleteLoopOrder { expected, actual } => write!(
                formatter,
                "loop order must contain all {expected} loops exactly once, got {actual}"
            ),
            Self::DuplicateLoop { loop_id } => {
                write!(formatter, "loop {} appears more than once", loop_id.index())
            }
            Self::ReductionReorderUnsupported => {
                write!(formatter, "reordering reduction loops is unsupported")
            }
            Self::ParallelLoopInsideReduction => {
                write!(
                    formatter,
                    "strict reductions require all parallel loops outside reduction loops"
                )
            }
            Self::ReductionBindUnsupported { loop_name } => write!(
                formatter,
                "binding reduction loop `{loop_name}` to CUDA is unsupported"
            ),
            Self::LoopAlreadyBound { loop_name, binding } => write!(
                formatter,
                "loop `{loop_name}` is already bound to {binding:?}"
            ),
            Self::BindingAlreadyUsed { binding, loop_name } => write!(
                formatter,
                "CUDA binding {binding:?} is already used by loop `{loop_name}`"
            ),
            Self::CpuBindingUnsupported { loop_name, binding } => write!(
                formatter,
                "CPU loop `{loop_name}` cannot have CUDA binding {binding:?}"
            ),
            Self::InvalidCudaTargetLimit { field } => {
                write!(
                    formatter,
                    "CUDA target limit `{field}` must be greater than zero"
                )
            }
            Self::DynamicCudaThreadExtent { loop_name } => write!(
                formatter,
                "CUDA thread loop `{loop_name}` must have a static extent"
            ),
            Self::ZeroCudaLaunchDimension { binding } => {
                write!(formatter, "CUDA binding {binding:?} has zero extent")
            }
            Self::CudaDimensionExceeded {
                binding,
                requested,
                maximum,
            } => write!(
                formatter,
                "CUDA binding {binding:?} requests extent {requested}, exceeding target limit {maximum}"
            ),
            Self::CudaThreadsPerBlockExceeded { requested, maximum } => write!(
                formatter,
                "CUDA schedule requests {requested} threads per block, exceeding target limit {maximum}"
            ),
            Self::ArithmeticOverflow { context } => {
                write!(formatter, "arithmetic overflow while computing {context}")
            }
        }
    }
}

impl Error for ScheduleError {}

pub(crate) fn operation_for(
    function: &VerifiedComputeFunction,
    operation: ComputeOpId,
) -> Result<&ComputeOp, ScheduleError> {
    function
        .operation(operation)
        .ok_or(ScheduleError::UnknownOperation {
            index: operation.index(),
        })
}

pub(crate) fn create_loop_plan(
    function: &VerifiedComputeFunction,
    operation: ComputeOpId,
) -> Result<LoopPlan, ScheduleError> {
    Ok(LoopPlan::new(operation_for(function, operation)?))
}

pub(crate) fn verify_loop_plan(plan: &LoopPlan) -> Result<(), ScheduleError> {
    if has_parallel_inside_reduction(&plan.loops) {
        return Err(ScheduleError::ParallelLoopInsideReduction);
    }
    Ok(())
}

fn has_parallel_inside_reduction(loops: &[LoopAxis]) -> bool {
    let mut saw_reduction = false;
    for loop_axis in loops {
        match loop_axis.kind {
            IteratorKind::Parallel if saw_reduction => return true,
            IteratorKind::Reduction => saw_reduction = true,
            IteratorKind::Parallel => {}
        }
    }
    false
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    value / divisor + u64::from(value % divisor != 0)
}
