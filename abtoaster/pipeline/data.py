from .validation import FlickersCheck, SampleSizeImbalanceCheck


class InputData:
    """A/B test input data in raw format
    X - user attributes including test_entry_key
    Y - daily target metric dataset with day_key
    
    """
    def __init__(self, X, Y, w, test_entry_key='join_date', day_key='nday'):
        pass


class ABDataset:
    """
    Class for storing A/B/N test data ready for pipeline.

    Dataset defined as following:
    - `X` pandas DataFrame with user's attributes
    - w pandas Series with A/B test variant assignments
    - y pandas Series for Target Metric
    """
    def __init__(self, X, y, w, control_key='control'):
        self.X = X
        self.w = w
        self.y = y
        self.control_key = control_key
