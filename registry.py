from pipeline import PreprocessingPipeline
from modules.discretization.discretization import Discretization
from modules.data_reduction.data_reduction import DataReduction, Histogram
from modules.similarity.similarity import SimilarityAnalyzer
from modules.smoothing.smoothing import DataSmoother
# ------------------------------------------------------------------
# Pattern for each:
#   from modules.<folder>.<file> import <ClassName>
#   PreprocessingPipeline.register("<Display Name>", <ClassName>)
# ------------------------------------------------------------------

PreprocessingPipeline.register("Discretization", Discretization)

# PreprocessingPipeline.register("Missing Values", MissingValueEstimator)
PreprocessingPipeline.register("Data Smoothing", DataSmoother)
PreprocessingPipeline.register("Data Reduction", DataReduction)
# PreprocessingPipeline.register("Similarity", Similarity)



# ------------------------------------------------------------------
# Analysis modules — terminal, summarize data via analyze()
# These do not modify current_df or pipeline history.
# ------------------------------------------------------------------

PreprocessingPipeline.register_analysis("Histogram", Histogram)
PreprocessingPipeline.register_analysis("Similarity", SimilarityAnalyzer)

# ------------------------------------------------------------------
# Register ML modules here once that phase begins.
# The pipeline handles them identically — no structural changes needed.
# ------------------------------------------------------------------

