from modules.histogram_disc.histogram_discretization import HistogramDiscretizer
from pipeline import PreprocessingPipeline
from modules.discretization.discretization import Discretization
from modules.data_reduction.data_reduction import DataReduction, Histogram
from modules.similarity.similarity import SimilarityAnalyzer
from modules.missing_values.missing_values import MissingValueEstimator
from modules.smoothing.smoothing import DataSmoother
from modules.encoding.encoding import CategoricalEncoder
# ------------------------------------------------------------------
# Pattern for each:
#   from modules.<folder>.<file> import <ClassName>
#   PreprocessingPipeline.register("<Display Name>", <ClassName>)
# ------------------------------------------------------------------

PreprocessingPipeline.register("Discretization", Discretization)
PreprocessingPipeline.register("Histogram Discretization", HistogramDiscretizer)
PreprocessingPipeline.register("Missing Values", MissingValueEstimator)
PreprocessingPipeline.register("Data Smoothing", DataSmoother)
PreprocessingPipeline.register("Data Reduction", DataReduction)
PreprocessingPipeline.register("Encoding", CategoricalEncoder)



# ------------------------------------------------------------------
# Analysis modules — terminal, summarize data via analyze()
# These do not modify current_df or pipeline history.
# ------------------------------------------------------------------

PreprocessingPipeline.register_analysis("Histogram", Histogram)
PreprocessingPipeline.register_analysis("Similarity", SimilarityAnalyzer)

# ------------------------------------------------------------------
# Register ML modules here once that phase begins.
# ------------------------------------------------------------------

from ml_pipeline import MLPipeline
from modules.ClassifierModels.DecisionTree import DecisionTreeModule

MLPipeline.register_model("Decision Tree", DecisionTreeModule)

