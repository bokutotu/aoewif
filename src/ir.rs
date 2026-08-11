use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            owner: u64,
            index: usize,
        }

        impl $name {
            fn new(owner: u64, index: usize) -> Self {
                Self { owner, index }
            }

            pub fn index(self) -> usize {
                self.index
            }
        }
    };
}

define_id!(SymbolId);
define_id!(TensorValueId);
define_id!(ComputeOpId);
define_id!(IteratorId);
define_id!(ScalarValueId);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    Static(u64),
    Symbol(SymbolId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    F32,
    Bool,
    Index,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorType {
    shape: Vec<Dim>,
    element: ScalarType,
}

impl TensorType {
    pub fn f32(shape: impl Into<Vec<Dim>>) -> Self {
        Self {
            shape: shape.into(),
            element: ScalarType::F32,
        }
    }

    pub fn new(shape: impl Into<Vec<Dim>>, element: ScalarType) -> Self {
        Self {
            shape: shape.into(),
            element,
        }
    }

    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn element(&self) -> ScalarType {
        self.element
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    id: SymbolId,
    name: String,
}

impl Symbol {
    pub fn id(&self) -> SymbolId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorDefinition {
    Input { input_index: usize },
    ComputeResult { operation: ComputeOpId },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorValue {
    id: TensorValueId,
    name: String,
    tensor_type: TensorType,
    definition: TensorDefinition,
}

impl TensorValue {
    pub fn id(&self) -> TensorValueId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tensor_type(&self) -> &TensorType {
        &self.tensor_type
    }

    pub fn definition(&self) -> &TensorDefinition {
        &self.definition
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IteratorKind {
    Parallel,
    Reduction,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iterator {
    id: IteratorId,
    name: String,
    extent: Dim,
    kind: IteratorKind,
}

impl Iterator {
    pub fn id(&self) -> IteratorId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn extent(&self) -> &Dim {
        &self.extent
    }

    pub fn kind(&self) -> IteratorKind {
        self.kind
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexExpr {
    Iterator(IteratorId),
    Constant(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAccess {
    tensor: TensorValueId,
    indices: Vec<IndexExpr>,
    scalar: ScalarValueId,
}

impl TensorAccess {
    pub fn tensor(&self) -> TensorValueId {
        self.tensor
    }

    pub fn indices(&self) -> &[IndexExpr] {
        &self.indices
    }

    pub fn scalar(&self) -> ScalarValueId {
        self.scalar
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarLiteral {
    F32(f32),
    Bool(bool),
    Index(u64),
}

impl ScalarLiteral {
    pub fn scalar_type(self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::Bool(_) => ScalarType::Bool,
            Self::Index(_) => ScalarType::Index,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparePredicate {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarOperationKind {
    Constant(ScalarLiteral),
    Index(IteratorId),
    Add {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Sub {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Mul {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Div {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Fma {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
        accumulator: ScalarValueId,
    },
    Min {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Max {
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Exp {
        input: ScalarValueId,
    },
    Log {
        input: ScalarValueId,
    },
    Compare {
        predicate: ComparePredicate,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    },
    Select {
        condition: ScalarValueId,
        true_value: ScalarValueId,
        false_value: ScalarValueId,
    },
}

impl ScalarOperationKind {
    fn operands(&self) -> Vec<ScalarValueId> {
        match self {
            Self::Constant(_) | Self::Index(_) => Vec::new(),
            Self::Add { lhs, rhs }
            | Self::Sub { lhs, rhs }
            | Self::Mul { lhs, rhs }
            | Self::Div { lhs, rhs }
            | Self::Min { lhs, rhs }
            | Self::Max { lhs, rhs }
            | Self::Compare { lhs, rhs, .. } => vec![*lhs, *rhs],
            Self::Fma {
                lhs,
                rhs,
                accumulator,
            } => vec![*lhs, *rhs, *accumulator],
            Self::Exp { input } | Self::Log { input } => vec![*input],
            Self::Select {
                condition,
                true_value,
                false_value,
            } => vec![*condition, *true_value, *false_value],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarOperation {
    result: ScalarValueId,
    result_type: ScalarType,
    kind: ScalarOperationKind,
}

impl ScalarOperation {
    pub fn result(&self) -> ScalarValueId {
        self.result
    }

    pub fn result_type(&self) -> ScalarType {
        self.result_type
    }

    pub fn kind(&self) -> &ScalarOperationKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarArgumentKind {
    InputElement { access_index: usize },
    Accumulator,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarArgument {
    value: ScalarValueId,
    scalar_type: ScalarType,
    kind: ScalarArgumentKind,
}

impl ScalarArgument {
    pub fn value(&self) -> ScalarValueId {
        self.value
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.scalar_type
    }

    pub fn kind(&self) -> &ScalarArgumentKind {
        &self.kind
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarRegion {
    arguments: Vec<ScalarArgument>,
    operations: Vec<ScalarOperation>,
    result: ScalarValueId,
}

impl ScalarRegion {
    pub fn arguments(&self) -> &[ScalarArgument] {
        &self.arguments
    }

    pub fn operations(&self) -> &[ScalarOperation] {
        &self.operations
    }

    pub fn result(&self) -> ScalarValueId {
        self.result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionPolicy {
    Strict,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputeOp {
    id: ComputeOpId,
    name: String,
    iterators: Vec<Iterator>,
    accesses: Vec<TensorAccess>,
    init: Option<ScalarLiteral>,
    reduction_policy: ReductionPolicy,
    body: ScalarRegion,
    result: TensorValueId,
}

impl ComputeOp {
    pub fn id(&self) -> ComputeOpId {
        self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn iterators(&self) -> &[Iterator] {
        &self.iterators
    }

    pub fn accesses(&self) -> &[TensorAccess] {
        &self.accesses
    }

    pub fn init(&self) -> Option<ScalarLiteral> {
        self.init
    }

    pub fn reduction_policy(&self) -> ReductionPolicy {
        self.reduction_policy
    }

    pub fn body(&self) -> &ScalarRegion {
        &self.body
    }

    pub fn result(&self) -> TensorValueId {
        self.result
    }

    pub fn has_reduction(&self) -> bool {
        self.iterators
            .iter()
            .any(|iterator| iterator.kind == IteratorKind::Reduction)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputeFunction {
    owner: u64,
    name: String,
    symbols: Vec<Symbol>,
    tensors: Vec<TensorValue>,
    operations: Vec<ComputeOp>,
    outputs: Vec<TensorValueId>,
}

impl ComputeFunction {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn symbols(&self) -> &[Symbol] {
        &self.symbols
    }

    pub fn tensors(&self) -> &[TensorValue] {
        &self.tensors
    }

    pub fn operations(&self) -> &[ComputeOp] {
        &self.operations
    }

    pub fn outputs(&self) -> &[TensorValueId] {
        &self.outputs
    }

    pub fn tensor(&self, id: TensorValueId) -> Option<&TensorValue> {
        (id.owner == self.owner)
            .then(|| self.tensors.get(id.index))
            .flatten()
    }

    pub fn operation(&self, id: ComputeOpId) -> Option<&ComputeOp> {
        (id.owner == self.owner)
            .then(|| self.operations.get(id.index))
            .flatten()
    }

    pub fn input_tensors(&self) -> impl std::iter::Iterator<Item = &TensorValue> {
        self.tensors
            .iter()
            .filter(|tensor| matches!(tensor.definition, TensorDefinition::Input { .. }))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedComputeFunction(ComputeFunction);

impl VerifiedComputeFunction {
    pub fn function(&self) -> &ComputeFunction {
        &self.0
    }
}

impl std::ops::Deref for VerifiedComputeFunction {
    type Target = ComputeFunction;

    fn deref(&self) -> &Self::Target {
        self.function()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    InvalidIdentifier {
        value: String,
    },
    DuplicateName {
        value: String,
    },
    ForeignId {
        kind: &'static str,
    },
    UnknownTensor {
        index: usize,
    },
    UnknownIterator {
        index: usize,
    },
    UnknownScalar {
        index: usize,
    },
    TensorRankMismatch {
        expected: usize,
        actual: usize,
    },
    TensorElementTypeUnsupported {
        scalar_type: ScalarType,
    },
    DimensionMismatch,
    ConstantIndexRequiresStaticDimension,
    ConstantIndexOutOfBounds {
        index: u64,
        extent: u64,
    },
    ReductionInitRequired,
    ReductionInitWithoutReduction,
    ReductionAccumulatorRequired,
    AccumulatorWithoutReduction,
    DuplicateAccumulator,
    ScalarTypeMismatch {
        expected: ScalarType,
        actual: ScalarType,
    },
    ScalarOperandsMustMatch,
    ScalarResultMustMatchInit,
    InvalidScalarResult,
    OutputRequired,
    OutputAlreadyMarked {
        index: usize,
    },
}

impl fmt::Display for IrError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { value } => write!(formatter, "invalid identifier `{value}`"),
            Self::DuplicateName { value } => write!(formatter, "duplicate name `{value}`"),
            Self::ForeignId { kind } => write!(formatter, "foreign {kind} ID"),
            Self::UnknownTensor { index } => write!(formatter, "unknown tensor value {index}"),
            Self::UnknownIterator { index } => write!(formatter, "unknown iterator {index}"),
            Self::UnknownScalar { index } => write!(formatter, "unknown scalar value {index}"),
            Self::TensorRankMismatch { expected, actual } => {
                write!(
                    formatter,
                    "tensor rank mismatch: expected {expected}, got {actual}"
                )
            }
            Self::TensorElementTypeUnsupported { scalar_type } => {
                write!(formatter, "unsupported tensor element type {scalar_type:?}")
            }
            Self::DimensionMismatch => {
                write!(formatter, "iterator extent does not match tensor dimension")
            }
            Self::ConstantIndexRequiresStaticDimension => {
                write!(
                    formatter,
                    "constant indices require a static tensor dimension"
                )
            }
            Self::ConstantIndexOutOfBounds { index, extent } => {
                write!(
                    formatter,
                    "constant index {index} is outside extent {extent}"
                )
            }
            Self::ReductionInitRequired => write!(formatter, "reduction requires an initial value"),
            Self::ReductionInitWithoutReduction => {
                write!(formatter, "initial value is only valid for a reduction")
            }
            Self::ReductionAccumulatorRequired => {
                write!(formatter, "reduction body requires an accumulator argument")
            }
            Self::AccumulatorWithoutReduction => {
                write!(formatter, "accumulator is only valid for a reduction")
            }
            Self::DuplicateAccumulator => write!(formatter, "reduction already has an accumulator"),
            Self::ScalarTypeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "scalar type mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::ScalarOperandsMustMatch => write!(formatter, "scalar operand types must match"),
            Self::ScalarResultMustMatchInit => {
                write!(
                    formatter,
                    "reduction result type must match its initial value"
                )
            }
            Self::InvalidScalarResult => write!(formatter, "invalid scalar region result"),
            Self::OutputRequired => {
                write!(formatter, "compute function requires at least one output")
            }
            Self::OutputAlreadyMarked { index } => {
                write!(
                    formatter,
                    "tensor value {index} is already marked as an output"
                )
            }
        }
    }
}

impl Error for IrError {}

pub struct ComputeFunctionBuilder {
    function: ComputeFunction,
    names: HashSet<String>,
    next_iterator: usize,
    next_scalar: usize,
}

impl ComputeFunctionBuilder {
    pub fn new(name: impl Into<String>) -> Result<Self, IrError> {
        let name = name.into();
        validate_identifier(&name)?;
        Ok(Self {
            function: ComputeFunction {
                owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
                name,
                symbols: Vec::new(),
                tensors: Vec::new(),
                operations: Vec::new(),
                outputs: Vec::new(),
            },
            names: HashSet::new(),
            next_iterator: 0,
            next_scalar: 0,
        })
    }

    pub fn symbol(&mut self, name: impl Into<String>) -> Result<SymbolId, IrError> {
        let name = name.into();
        self.reserve_name(&name)?;
        let id = SymbolId::new(self.function.owner, self.function.symbols.len());
        self.function.symbols.push(Symbol { id, name });
        Ok(id)
    }

    pub fn input(
        &mut self,
        name: impl Into<String>,
        tensor_type: TensorType,
    ) -> Result<TensorValueId, IrError> {
        let name = name.into();
        self.reserve_name(&name)?;
        self.verify_tensor_type(&tensor_type)?;
        let id = TensorValueId::new(self.function.owner, self.function.tensors.len());
        let input_index = self.function.input_tensors().count();
        self.function.tensors.push(TensorValue {
            id,
            name,
            tensor_type,
            definition: TensorDefinition::Input { input_index },
        });
        Ok(id)
    }

    pub fn compute(&mut self, name: impl Into<String>) -> Result<ComputeOpBuilder<'_>, IrError> {
        let name = name.into();
        self.reserve_name(&name)?;
        Ok(ComputeOpBuilder {
            function: self,
            name,
            iterators: Vec::new(),
            accesses: Vec::new(),
            init: None,
            accumulator: None,
            scalar_arguments: Vec::new(),
            scalar_operations: Vec::new(),
            scalar_types: HashMap::new(),
        })
    }

    pub fn mark_output(&mut self, tensor: TensorValueId) -> Result<(), IrError> {
        self.tensor(tensor)?;
        if self.function.outputs.contains(&tensor) {
            return Err(IrError::OutputAlreadyMarked {
                index: tensor.index,
            });
        }
        self.function.outputs.push(tensor);
        Ok(())
    }

    pub fn finish(self) -> Result<VerifiedComputeFunction, IrError> {
        verify_function(&self.function)?;
        Ok(VerifiedComputeFunction(self.function))
    }

    fn reserve_name(&mut self, name: &str) -> Result<(), IrError> {
        validate_identifier(name)?;
        if !self.names.insert(name.to_owned()) {
            return Err(IrError::DuplicateName {
                value: name.to_owned(),
            });
        }
        Ok(())
    }

    fn verify_tensor_type(&self, tensor_type: &TensorType) -> Result<(), IrError> {
        if tensor_type.element != ScalarType::F32 {
            return Err(IrError::TensorElementTypeUnsupported {
                scalar_type: tensor_type.element,
            });
        }
        for dim in &tensor_type.shape {
            if let Dim::Symbol(symbol) = dim {
                self.symbol_by_id(*symbol)?;
            }
        }
        Ok(())
    }

    fn symbol_by_id(&self, id: SymbolId) -> Result<&Symbol, IrError> {
        if id.owner != self.function.owner {
            return Err(IrError::ForeignId { kind: "symbol" });
        }
        self.function
            .symbols
            .get(id.index)
            .ok_or(IrError::ForeignId { kind: "symbol" })
    }

    fn tensor(&self, id: TensorValueId) -> Result<&TensorValue, IrError> {
        if id.owner != self.function.owner {
            return Err(IrError::ForeignId { kind: "tensor" });
        }
        self.function
            .tensors
            .get(id.index)
            .ok_or(IrError::UnknownTensor { index: id.index })
    }

    fn allocate_iterator(&mut self) -> IteratorId {
        let id = IteratorId::new(self.function.owner, self.next_iterator);
        self.next_iterator += 1;
        id
    }

    fn allocate_scalar(&mut self) -> ScalarValueId {
        let id = ScalarValueId::new(self.function.owner, self.next_scalar);
        self.next_scalar += 1;
        id
    }
}

pub struct ComputeOpBuilder<'a> {
    function: &'a mut ComputeFunctionBuilder,
    name: String,
    iterators: Vec<Iterator>,
    accesses: Vec<TensorAccess>,
    init: Option<ScalarLiteral>,
    accumulator: Option<ScalarValueId>,
    scalar_arguments: Vec<ScalarArgument>,
    scalar_operations: Vec<ScalarOperation>,
    scalar_types: HashMap<ScalarValueId, ScalarType>,
}

impl ComputeOpBuilder<'_> {
    pub fn parallel(
        &mut self,
        name: impl Into<String>,
        extent: Dim,
    ) -> Result<IteratorId, IrError> {
        self.iterator(name, extent, IteratorKind::Parallel)
    }

    pub fn reduction(
        &mut self,
        name: impl Into<String>,
        extent: Dim,
    ) -> Result<IteratorId, IrError> {
        self.iterator(name, extent, IteratorKind::Reduction)
    }

    pub fn read(
        &mut self,
        tensor: TensorValueId,
        indices: impl Into<Vec<IndexExpr>>,
    ) -> Result<ScalarValueId, IrError> {
        let indices = indices.into();
        let tensor_type = self.function.tensor(tensor)?.tensor_type.clone();
        self.verify_access(&tensor_type, &indices)?;
        let scalar = self.function.allocate_scalar();
        let access_index = self.accesses.len();
        self.accesses.push(TensorAccess {
            tensor,
            indices,
            scalar,
        });
        self.scalar_arguments.push(ScalarArgument {
            value: scalar,
            scalar_type: tensor_type.element,
            kind: ScalarArgumentKind::InputElement { access_index },
        });
        self.scalar_types.insert(scalar, tensor_type.element);
        Ok(scalar)
    }

    pub fn reduction_init(&mut self, init: ScalarLiteral) -> Result<ScalarValueId, IrError> {
        if self.accumulator.is_some() {
            return Err(IrError::DuplicateAccumulator);
        }
        let accumulator = self.function.allocate_scalar();
        self.init = Some(init);
        self.accumulator = Some(accumulator);
        self.scalar_arguments.push(ScalarArgument {
            value: accumulator,
            scalar_type: init.scalar_type(),
            kind: ScalarArgumentKind::Accumulator,
        });
        self.scalar_types.insert(accumulator, init.scalar_type());
        Ok(accumulator)
    }

    pub fn constant(&mut self, value: ScalarLiteral) -> ScalarValueId {
        self.push_operation(ScalarOperationKind::Constant(value), value.scalar_type())
    }

    pub fn index(&mut self, iterator: IteratorId) -> Result<ScalarValueId, IrError> {
        self.iterator_by_id(iterator)?;
        Ok(self.push_operation(ScalarOperationKind::Index(iterator), ScalarType::Index))
    }

    pub fn add(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Add { lhs, rhs })
    }

    pub fn sub(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Sub { lhs, rhs })
    }

    pub fn mul(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Mul { lhs, rhs })
    }

    pub fn div(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Div { lhs, rhs })
    }

    pub fn fma(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
        accumulator: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.expect_type(lhs, ScalarType::F32)?;
        self.expect_type(rhs, ScalarType::F32)?;
        self.expect_type(accumulator, ScalarType::F32)?;
        Ok(self.push_operation(
            ScalarOperationKind::Fma {
                lhs,
                rhs,
                accumulator,
            },
            ScalarType::F32,
        ))
    }

    pub fn min(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Min { lhs, rhs })
    }

    pub fn max(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.binary_f32(lhs, rhs, |lhs, rhs| ScalarOperationKind::Max { lhs, rhs })
    }

    pub fn exp(&mut self, input: ScalarValueId) -> Result<ScalarValueId, IrError> {
        self.expect_type(input, ScalarType::F32)?;
        Ok(self.push_operation(ScalarOperationKind::Exp { input }, ScalarType::F32))
    }

    pub fn log(&mut self, input: ScalarValueId) -> Result<ScalarValueId, IrError> {
        self.expect_type(input, ScalarType::F32)?;
        Ok(self.push_operation(ScalarOperationKind::Log { input }, ScalarType::F32))
    }

    pub fn compare(
        &mut self,
        predicate: ComparePredicate,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        let lhs_type = self.scalar_type(lhs)?;
        let rhs_type = self.scalar_type(rhs)?;
        if lhs_type != rhs_type {
            return Err(IrError::ScalarOperandsMustMatch);
        }
        Ok(self.push_operation(
            ScalarOperationKind::Compare {
                predicate,
                lhs,
                rhs,
            },
            ScalarType::Bool,
        ))
    }

    pub fn select(
        &mut self,
        condition: ScalarValueId,
        true_value: ScalarValueId,
        false_value: ScalarValueId,
    ) -> Result<ScalarValueId, IrError> {
        self.expect_type(condition, ScalarType::Bool)?;
        let true_type = self.scalar_type(true_value)?;
        let false_type = self.scalar_type(false_value)?;
        if true_type != false_type {
            return Err(IrError::ScalarOperandsMustMatch);
        }
        Ok(self.push_operation(
            ScalarOperationKind::Select {
                condition,
                true_value,
                false_value,
            },
            true_type,
        ))
    }

    pub fn finish(self, result: ScalarValueId) -> Result<TensorValueId, IrError> {
        let result_type = self.scalar_type(result)?;
        let has_reduction = self
            .iterators
            .iter()
            .any(|iterator| iterator.kind == IteratorKind::Reduction);
        match (has_reduction, self.init, self.accumulator) {
            (true, None, _) => return Err(IrError::ReductionInitRequired),
            (true, Some(_), None) => return Err(IrError::ReductionAccumulatorRequired),
            (false, Some(_), _) => return Err(IrError::ReductionInitWithoutReduction),
            (false, None, Some(_)) => return Err(IrError::AccumulatorWithoutReduction),
            _ => {}
        }
        if let Some(init) = self.init {
            if result_type != init.scalar_type() {
                return Err(IrError::ScalarResultMustMatchInit);
            }
        }
        if result_type != ScalarType::F32 {
            return Err(IrError::TensorElementTypeUnsupported {
                scalar_type: result_type,
            });
        }

        let owner = self.function.function.owner;
        let operation_id = ComputeOpId::new(owner, self.function.function.operations.len());
        let tensor_id = TensorValueId::new(owner, self.function.function.tensors.len());
        let tensor_type = TensorType::f32(
            self.iterators
                .iter()
                .filter(|iterator| iterator.kind == IteratorKind::Parallel)
                .map(|iterator| iterator.extent.clone())
                .collect::<Vec<_>>(),
        );
        self.function.function.tensors.push(TensorValue {
            id: tensor_id,
            name: self.name.clone(),
            tensor_type,
            definition: TensorDefinition::ComputeResult {
                operation: operation_id,
            },
        });
        self.function.function.operations.push(ComputeOp {
            id: operation_id,
            name: self.name,
            iterators: self.iterators,
            accesses: self.accesses,
            init: self.init,
            reduction_policy: ReductionPolicy::Strict,
            body: ScalarRegion {
                arguments: self.scalar_arguments,
                operations: self.scalar_operations,
                result,
            },
            result: tensor_id,
        });
        Ok(tensor_id)
    }

    fn iterator(
        &mut self,
        name: impl Into<String>,
        extent: Dim,
        kind: IteratorKind,
    ) -> Result<IteratorId, IrError> {
        let name = name.into();
        validate_identifier(&name)?;
        if self.iterators.iter().any(|iterator| iterator.name == name) {
            return Err(IrError::DuplicateName { value: name });
        }
        if let Dim::Symbol(symbol) = &extent {
            self.function.symbol_by_id(*symbol)?;
        }
        let id = self.function.allocate_iterator();
        self.iterators.push(Iterator {
            id,
            name,
            extent,
            kind,
        });
        Ok(id)
    }

    fn verify_access(
        &self,
        tensor_type: &TensorType,
        indices: &[IndexExpr],
    ) -> Result<(), IrError> {
        if tensor_type.rank() != indices.len() {
            return Err(IrError::TensorRankMismatch {
                expected: tensor_type.rank(),
                actual: indices.len(),
            });
        }
        for (dimension, index) in tensor_type.shape.iter().zip(indices) {
            match index {
                IndexExpr::Iterator(iterator) => {
                    let iterator = self.iterator_by_id(*iterator)?;
                    if &iterator.extent != dimension {
                        return Err(IrError::DimensionMismatch);
                    }
                }
                IndexExpr::Constant(index) => match dimension {
                    Dim::Static(extent) if index < extent => {}
                    Dim::Static(extent) => {
                        return Err(IrError::ConstantIndexOutOfBounds {
                            index: *index,
                            extent: *extent,
                        });
                    }
                    Dim::Symbol(_) => return Err(IrError::ConstantIndexRequiresStaticDimension),
                },
            }
        }
        Ok(())
    }

    fn iterator_by_id(&self, id: IteratorId) -> Result<&Iterator, IrError> {
        if id.owner != self.function.function.owner {
            return Err(IrError::ForeignId { kind: "iterator" });
        }
        self.iterators
            .iter()
            .find(|iterator| iterator.id == id)
            .ok_or(IrError::UnknownIterator { index: id.index })
    }

    fn binary_f32(
        &mut self,
        lhs: ScalarValueId,
        rhs: ScalarValueId,
        make_kind: impl FnOnce(ScalarValueId, ScalarValueId) -> ScalarOperationKind,
    ) -> Result<ScalarValueId, IrError> {
        self.expect_type(lhs, ScalarType::F32)?;
        self.expect_type(rhs, ScalarType::F32)?;
        Ok(self.push_operation(make_kind(lhs, rhs), ScalarType::F32))
    }

    fn expect_type(&self, value: ScalarValueId, expected: ScalarType) -> Result<(), IrError> {
        let actual = self.scalar_type(value)?;
        if actual != expected {
            return Err(IrError::ScalarTypeMismatch { expected, actual });
        }
        Ok(())
    }

    fn scalar_type(&self, value: ScalarValueId) -> Result<ScalarType, IrError> {
        if value.owner != self.function.function.owner {
            return Err(IrError::ForeignId { kind: "scalar" });
        }
        self.scalar_types
            .get(&value)
            .copied()
            .ok_or(IrError::UnknownScalar { index: value.index })
    }

    fn push_operation(
        &mut self,
        kind: ScalarOperationKind,
        result_type: ScalarType,
    ) -> ScalarValueId {
        let result = self.function.allocate_scalar();
        self.scalar_operations.push(ScalarOperation {
            result,
            result_type,
            kind,
        });
        self.scalar_types.insert(result, result_type);
        result
    }
}

fn validate_identifier(value: &str) -> Result<(), IrError> {
    let mut characters = value.chars();
    let Some(first) = characters.next() else {
        return Err(IrError::InvalidIdentifier {
            value: value.to_owned(),
        });
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return Err(IrError::InvalidIdentifier {
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn verify_function(function: &ComputeFunction) -> Result<(), IrError> {
    if function.outputs.is_empty() {
        return Err(IrError::OutputRequired);
    }
    for symbol in &function.symbols {
        if symbol.id.owner != function.owner {
            return Err(IrError::ForeignId { kind: "symbol" });
        }
    }
    for tensor in &function.tensors {
        if tensor.id.owner != function.owner {
            return Err(IrError::ForeignId { kind: "tensor" });
        }
        for dim in &tensor.tensor_type.shape {
            if let Dim::Symbol(symbol) = dim {
                if symbol.owner != function.owner || function.symbols.get(symbol.index).is_none() {
                    return Err(IrError::ForeignId { kind: "symbol" });
                }
            }
        }
    }
    for operation in &function.operations {
        verify_operation(function, operation)?;
    }
    for output in &function.outputs {
        if output.owner != function.owner || function.tensors.get(output.index).is_none() {
            return Err(IrError::ForeignId {
                kind: "output tensor",
            });
        }
    }
    Ok(())
}

fn verify_operation(function: &ComputeFunction, operation: &ComputeOp) -> Result<(), IrError> {
    let iterator_ids: HashSet<_> = operation
        .iterators
        .iter()
        .map(|iterator| iterator.id)
        .collect();
    for iterator in &operation.iterators {
        if iterator.id.owner != function.owner {
            return Err(IrError::ForeignId { kind: "iterator" });
        }
    }
    for access in &operation.accesses {
        let tensor = function
            .tensor(access.tensor)
            .ok_or(IrError::UnknownTensor {
                index: access.tensor.index,
            })?;
        if tensor.tensor_type.rank() != access.indices.len() {
            return Err(IrError::TensorRankMismatch {
                expected: tensor.tensor_type.rank(),
                actual: access.indices.len(),
            });
        }
        for index in &access.indices {
            if let IndexExpr::Iterator(iterator) = index {
                if !iterator_ids.contains(iterator) {
                    return Err(IrError::UnknownIterator {
                        index: iterator.index,
                    });
                }
            }
        }
    }

    let mut scalar_types = HashMap::new();
    let mut accumulator_count = 0;
    for argument in &operation.body.arguments {
        if argument.value.owner != function.owner {
            return Err(IrError::ForeignId { kind: "scalar" });
        }
        if matches!(argument.kind, ScalarArgumentKind::Accumulator) {
            accumulator_count += 1;
        }
        scalar_types.insert(argument.value, argument.scalar_type);
    }
    for scalar_operation in &operation.body.operations {
        for operand in scalar_operation.kind.operands() {
            if !scalar_types.contains_key(&operand) {
                return Err(IrError::UnknownScalar {
                    index: operand.index,
                });
            }
        }
        scalar_types.insert(scalar_operation.result, scalar_operation.result_type);
    }
    let result_type = scalar_types
        .get(&operation.body.result)
        .copied()
        .ok_or(IrError::InvalidScalarResult)?;
    let has_reduction = operation.has_reduction();
    match (has_reduction, operation.init, accumulator_count) {
        (true, None, _) => return Err(IrError::ReductionInitRequired),
        (true, Some(_), 0) => return Err(IrError::ReductionAccumulatorRequired),
        (true, Some(_), 1) => {}
        (true, Some(_), _) => return Err(IrError::DuplicateAccumulator),
        (false, Some(_), _) => return Err(IrError::ReductionInitWithoutReduction),
        (false, None, 0) => {}
        (false, None, _) => return Err(IrError::AccumulatorWithoutReduction),
    }
    if let Some(init) = operation.init {
        if init.scalar_type() != result_type {
            return Err(IrError::ScalarResultMustMatchInit);
        }
    }
    let result_tensor = function
        .tensor(operation.result)
        .ok_or(IrError::UnknownTensor {
            index: operation.result.index,
        })?;
    if result_tensor.tensor_type.element != result_type {
        return Err(IrError::ScalarTypeMismatch {
            expected: result_tensor.tensor_type.element,
            actual: result_type,
        });
    }
    Ok(())
}
