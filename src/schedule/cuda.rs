use crate::ir::{ComputeOp, ComputeOpId, IteratorId, VerifiedComputeFunction};

use super::{
    CudaBinding, CudaDimension, LoopId, LoopPlan, ScheduleError, create_loop_plan, operation_for,
    verify_loop_plan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaDim3 {
    x: u64,
    y: u64,
    z: u64,
}

impl CudaDim3 {
    pub const fn new(x: u64, y: u64, z: u64) -> Self {
        Self { x, y, z }
    }

    pub const fn x(self) -> u64 {
        self.x
    }

    pub const fn y(self) -> u64 {
        self.y
    }

    pub const fn z(self) -> u64 {
        self.z
    }

    pub const fn get(self, dimension: CudaDimension) -> u64 {
        match dimension {
            CudaDimension::X => self.x,
            CudaDimension::Y => self.y,
            CudaDimension::Z => self.z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaTarget {
    max_threads_per_block: u64,
    max_block_dimensions: CudaDim3,
    max_grid_dimensions: CudaDim3,
}

impl CudaTarget {
    pub const fn new(
        max_threads_per_block: u64,
        max_block_dimensions: CudaDim3,
        max_grid_dimensions: CudaDim3,
    ) -> Self {
        Self {
            max_threads_per_block,
            max_block_dimensions,
            max_grid_dimensions,
        }
    }

    pub const fn max_threads_per_block(self) -> u64 {
        self.max_threads_per_block
    }

    pub const fn max_block_dimensions(self) -> CudaDim3 {
        self.max_block_dimensions
    }

    pub const fn max_grid_dimensions(self) -> CudaDim3 {
        self.max_grid_dimensions
    }
}

impl Default for CudaTarget {
    fn default() -> Self {
        Self::new(
            1024,
            CudaDim3::new(1024, 1024, 64),
            CudaDim3::new(2_147_483_647, 65_535, 65_535),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaSchedule {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    plan: LoopPlan,
    target: CudaTarget,
}

impl CudaSchedule {
    pub fn new(
        function: VerifiedComputeFunction,
        operation: ComputeOpId,
        target: CudaTarget,
    ) -> Result<Self, ScheduleError> {
        validate_target(target)?;
        let plan = create_loop_plan(&function, operation)?;
        Ok(Self {
            function,
            operation,
            plan,
            target,
        })
    }

    pub fn function(&self) -> &VerifiedComputeFunction {
        &self.function
    }

    pub fn operation(&self) -> &ComputeOp {
        operation_for(&self.function, self.operation)
            .expect("a schedule must retain its compute operation")
    }

    pub fn plan(&self) -> &LoopPlan {
        &self.plan
    }

    pub const fn target(&self) -> CudaTarget {
        self.target
    }

    pub fn loop_for(&self, iterator: IteratorId) -> Option<LoopId> {
        self.plan.loop_for(iterator)
    }

    pub fn split(
        &mut self,
        loop_id: LoopId,
        factor: u64,
    ) -> Result<(LoopId, LoopId), ScheduleError> {
        self.plan.split(loop_id, factor)
    }

    pub fn reorder(&mut self, order: &[LoopId]) -> Result<(), ScheduleError> {
        self.plan.reorder(order)
    }

    pub fn bind(&mut self, loop_id: LoopId, binding: CudaBinding) -> Result<(), ScheduleError> {
        self.plan.bind(loop_id, binding)
    }

    pub fn verify(self) -> Result<VerifiedCudaSchedule, ScheduleError> {
        verify_loop_plan(&self.plan)?;
        validate_target(self.target)?;
        validate_bindings(&self.plan, self.target)?;
        Ok(VerifiedCudaSchedule(self))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedCudaSchedule(CudaSchedule);

impl VerifiedCudaSchedule {
    pub fn function(&self) -> &VerifiedComputeFunction {
        self.0.function()
    }

    pub fn operation(&self) -> &ComputeOp {
        self.0.operation()
    }

    pub fn plan(&self) -> &LoopPlan {
        self.0.plan()
    }

    pub const fn target(&self) -> CudaTarget {
        self.0.target()
    }
}

fn validate_target(target: CudaTarget) -> Result<(), ScheduleError> {
    for (field, value) in [
        ("max threads per block", target.max_threads_per_block()),
        ("max block dimension x", target.max_block_dimensions().x()),
        ("max block dimension y", target.max_block_dimensions().y()),
        ("max block dimension z", target.max_block_dimensions().z()),
        ("max grid dimension x", target.max_grid_dimensions().x()),
        ("max grid dimension y", target.max_grid_dimensions().y()),
        ("max grid dimension z", target.max_grid_dimensions().z()),
    ] {
        if value == 0 {
            return Err(ScheduleError::InvalidCudaTargetLimit { field });
        }
    }
    Ok(())
}

fn validate_bindings(plan: &LoopPlan, target: CudaTarget) -> Result<(), ScheduleError> {
    let mut threads_per_block = 1_u64;
    for loop_axis in plan.loops() {
        let Some(binding) = loop_axis.binding() else {
            continue;
        };
        let static_extent = loop_axis.extent().static_value();
        if binding.is_thread() && static_extent.is_none() {
            return Err(ScheduleError::DynamicCudaThreadExtent {
                loop_name: loop_axis.name().to_owned(),
            });
        }
        let Some(extent) = static_extent else {
            continue;
        };
        if extent == 0 {
            return Err(ScheduleError::ZeroCudaLaunchDimension { binding });
        }
        let maximum = if binding.is_thread() {
            target.max_block_dimensions().get(binding.dimension())
        } else {
            target.max_grid_dimensions().get(binding.dimension())
        };
        if extent > maximum {
            return Err(ScheduleError::CudaDimensionExceeded {
                binding,
                requested: extent,
                maximum,
            });
        }
        if binding.is_thread() {
            threads_per_block =
                threads_per_block
                    .checked_mul(extent)
                    .ok_or(ScheduleError::ArithmeticOverflow {
                        context: "CUDA threads per block",
                    })?;
        }
    }

    if threads_per_block > target.max_threads_per_block() {
        return Err(ScheduleError::CudaThreadsPerBlockExceeded {
            requested: threads_per_block,
            maximum: target.max_threads_per_block(),
        });
    }
    Ok(())
}
