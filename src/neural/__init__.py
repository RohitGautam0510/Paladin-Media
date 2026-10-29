"""Phase 3 Neural Wavelet Steganography Core Components.

Implements the isolated neural modules for the ACS-Sub-INN v3 architecture:
- ContextPredictorCNN: Predicts context coefficients from preserved wavelet subbands.
- AdaptiveGainCNN: Predicts spatially varying strictly positive modulation gain.
- RoundSTE, SpatialQuantizeSTE: Straight-Through Estimator differentiable quantizers.
- ACSSubINN: Orchestrates forward coefficient embedding and inverse extraction.
- NeuralSpatialBridge: Bridges spatial RGB images to wavelet domain with DWT/IDWT.
- losses: Candidate loss functions and diagnostic metrics.
- training_scaffold: Training loop scaffold, dataset split protection, and gate registry.
"""

from src.neural.adaptive_gain import AdaptiveGainCNN
from src.neural.context_predictor import ContextPredictorCNN
from src.neural.losses import (
    AdaptiveGainRegularization,
    BitRecoveryLoss,
    ClippingDiagnosticReport,
    CompositeLoss,
    ContextPredictionLoss,
    ImageFidelityLoss,
    LossBreakdown,
    evaluate_clipping_diagnostic,
)
from src.neural.onnx_export import (
    ACSSubINNEmbeddingGraph,
    ACSSubINNExtractionGraph,
    NeuralSpatialEmbeddingGraph,
    NeuralSpatialExtractionGraph,
    check_onnx_dependencies,
    compute_numerical_parity,
)
from src.neural.quantization import (
    RoundSTE,
    RoundSTEFunction,
    SpatialQuantizeSTE,
    SpatialQuantizeSTEFunction,
    round_ste,
    spatial_quantize_ste,
)
from src.neural.acs_sub_inn import ACSSubINN
from src.neural.spatial_bridge import (
    HaarDWT2DTorch,
    NeuralSpatialBridge,
    SpatialEmbeddingResult,
    SpatialExtractionResult,
    SpatialRoundtripMetrics,
    haar_dwt2d,
    haar_idwt2d,
)
from src.neural.training_scaffold import (
    DatasetSplitManager,
    GateDefinition,
    GateRegistry,
    TrainingConfig,
    TrainingStepResult,
    build_model_and_optimizer,
    evaluate_partition_verified_step,
    execute_partition_verified_step,
    execute_training_step,
    load_and_validate_checkpoint,
    save_synthetic_smoke_checkpoint,
    set_reproducibility_seeds,
)
from src.neural.dataset import (
    DIV2KStegoDataset,
    build_stego_dataloader,
    normalize_split_name,
    validate_div2k_manifest,
)
from src.neural.training_engine import (
    DatasetSplitBoundaryError,
    EpochTrainMetrics,
    IncompatibleCheckpointError,
    NonFiniteGradientError,
    NonFiniteLossError,
    PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE,
    PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE,
    TrainingAuthorizationToken,
    TrainingNotAuthorizedError,
    TrainingRunConfig,
    TrainingRunner,
    TrainingRunSummary,
    TrainingTelemetryLogger,
    VALID_TRAINING_SCOPES,
    ValidationMetrics,
    evaluate_validation_epoch,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_training_authorization,
)

__all__ = [
    "ContextPredictorCNN",
    "AdaptiveGainCNN",
    "RoundSTE",
    "RoundSTEFunction",
    "SpatialQuantizeSTE",
    "SpatialQuantizeSTEFunction",
    "round_ste",
    "spatial_quantize_ste",
    "ACSSubINN",
    "HaarDWT2DTorch",
    "NeuralSpatialBridge",
    "SpatialEmbeddingResult",
    "SpatialExtractionResult",
    "SpatialRoundtripMetrics",
    "haar_dwt2d",
    "haar_idwt2d",
    "BitRecoveryLoss",
    "ContextPredictionLoss",
    "AdaptiveGainRegularization",
    "ImageFidelityLoss",
    "ClippingDiagnosticReport",
    "evaluate_clipping_diagnostic",
    "CompositeLoss",
    "LossBreakdown",
    "DatasetSplitManager",
    "TrainingConfig",
    "TrainingStepResult",
    "GateDefinition",
    "GateRegistry",
    "set_reproducibility_seeds",
    "build_model_and_optimizer",
    "execute_training_step",
    "execute_partition_verified_step",
    "evaluate_partition_verified_step",
    "save_synthetic_smoke_checkpoint",
    "load_and_validate_checkpoint",
    "DIV2KStegoDataset",
    "build_stego_dataloader",
    "validate_div2k_manifest",
    "normalize_split_name",
    "TrainingAuthorizationToken",
    "PHASE4B_AUTHORIZED_TRAINING_PASSPHRASE",
    "PHASE4C_AUTHORIZED_TRAINING_PASSPHRASE",
    "VALID_TRAINING_SCOPES",
    "TrainingNotAuthorizedError",
    "NonFiniteLossError",
    "NonFiniteGradientError",
    "IncompatibleCheckpointError",
    "DatasetSplitBoundaryError",
    "TrainingRunConfig",
    "TrainingRunner",
    "TrainingRunSummary",
    "EpochTrainMetrics",
    "ValidationMetrics",
    "evaluate_validation_epoch",
    "save_training_checkpoint",
    "load_training_checkpoint",
    "verify_training_authorization",
    "TrainingTelemetryLogger",
    "ACSSubINNEmbeddingGraph",
    "ACSSubINNExtractionGraph",
    "NeuralSpatialEmbeddingGraph",
    "NeuralSpatialExtractionGraph",
    "check_onnx_dependencies",
    "compute_numerical_parity",
]
