from .models.initial_condition import InitialCondition, InitialConditionFromParameters

from .models.equilibrium import (
    EquilibriumStateInitialization,
    AlternatingDecouplingInTime,
    AlternatingDecouplingInNewton,
    CacheReferenceState,
)

from .models.hydrostatic_pressure import (
    HydrostaticPressureBC,
    HydrostaticPressureInitialCondition,
    HydrostaticPressureInitialization,
)

from .models.lithostatic_pressure import BackgroundStress, LithostaticPressureBC

from .models.normal_permeability import (
    ScalarPermeability,
    NormalPermeabilityFromHigherDimension,
    NormalPermeabilityFromLowerDimension,
)

from .models.background_deformation import BackgroundDeformation

from .models.linear_compressibility import LinearFluidCompressibility

from .models.custom_prepare_simulation import TwoPartedPrepareSimulation

from .numerics.newton_return_map import NewtonReturnMap, SafeNewtonReturnMap

from .setup.model_setup import setup_model
