from pipeline import PreprocessingPipeline
from modules.discretization.discretization import Discretization

# ------------------------------------------------------------------
# Pattern for each:
#   from modules.<folder>.<file> import <ClassName>
#   PreprocessingPipeline.register("<Display Name>", <ClassName>)
# ------------------------------------------------------------------

PreprocessingPipeline.register("Discretization", Discretization)

# PreprocessingPipeline.register("Missing Values", MissingValueEstimator)
# PreprocessingPipeline.register("Smoothing", Smoothing)
# PreprocessingPipeline.register("Data Reduction", DataReduction)
# PreprocessingPipeline.register("Similarity", Similarity)

# ------------------------------------------------------------------
# Register ML modules here once that phase begins.
# The pipeline handles them identically — no structural changes needed.
# ------------------------------------------------------------------

