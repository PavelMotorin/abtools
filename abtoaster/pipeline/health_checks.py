class HealthCheck:
    pass

class SampleSizeImbalance(HealthCheck):
    """
    Health Check for Sample size imbalance between control and treatment groups
    """
    pass

class Flickers(HealthCheck):
    """
    Health Check for users who changed test variant being in experiment.
    """
    pass