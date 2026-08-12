#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::too_many_lines)]

pub mod codegen;
pub mod ir;
pub mod schedule;

pub use codegen::{CSource, CodegenError, CudaSource, generate_c, generate_cuda};

pub use ir::{
    ComparePredicate, ComputeFunction, ComputeFunctionBuilder, ComputeOp, ComputeOpBuilder,
    ComputeOpId, Dim, IndexExpr, IrError, Iterator, IteratorId, IteratorKind, ReductionPolicy,
    ScalarArgument, ScalarArgumentKind, ScalarLiteral, ScalarOperation, ScalarOperationKind,
    ScalarRegion, ScalarType, ScalarValueId, Symbol, SymbolId, TensorAccess, TensorDefinition,
    TensorType, TensorValue, TensorValueId, VerifiedComputeFunction,
};
pub use schedule::{
    CpuSchedule, CudaBinding, CudaDim3, CudaDimension, CudaSchedule, CudaTarget, LogicalIndex,
    LoopAxis, LoopExtent, LoopId, LoopIndexExpr, LoopPlan, ScheduleError, TailPredicate,
    VerifiedCpuSchedule, VerifiedCudaSchedule,
};
