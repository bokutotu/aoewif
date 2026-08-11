use crate::ir::{ComputeOp, ComputeOpId, IteratorId, VerifiedComputeFunction};

use super::{LoopId, LoopPlan, ScheduleError, create_loop_plan, operation_for, verify_loop_plan};

#[derive(Clone, Debug, PartialEq)]
pub struct CpuSchedule {
    function: VerifiedComputeFunction,
    operation: ComputeOpId,
    plan: LoopPlan,
}

impl CpuSchedule {
    pub fn new(
        function: VerifiedComputeFunction,
        operation: ComputeOpId,
    ) -> Result<Self, ScheduleError> {
        let plan = create_loop_plan(&function, operation)?;
        Ok(Self {
            function,
            operation,
            plan,
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

    pub fn verify(self) -> Result<VerifiedCpuSchedule, ScheduleError> {
        verify_loop_plan(&self.plan)?;
        for loop_axis in self.plan.loops() {
            if let Some(binding) = loop_axis.binding() {
                return Err(ScheduleError::CpuBindingUnsupported {
                    loop_name: loop_axis.name().to_owned(),
                    binding,
                });
            }
        }
        Ok(VerifiedCpuSchedule(self))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedCpuSchedule(CpuSchedule);

impl VerifiedCpuSchedule {
    pub fn function(&self) -> &VerifiedComputeFunction {
        self.0.function()
    }

    pub fn operation(&self) -> &ComputeOp {
        self.0.operation()
    }

    pub fn plan(&self) -> &LoopPlan {
        self.0.plan()
    }
}
