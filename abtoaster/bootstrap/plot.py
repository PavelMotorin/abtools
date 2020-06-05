
def visualise(self, plot_stats=True):
    source = alt.Chart(self.to_frame())
    x = alt.X(f'{self.name}:Q', 
                title=f'{self.name}`s {self.func.__name__} bootstrap distribution',
                bin=alt.Bin(maxbins=100))
    y = alt.Y(f'count()', title='Count of Samples', stack=None)
    histogram = (
        source
        .mark_area(opacity=0.5, interpolate='step')
        .encode(x=x, y=y)
    )
    if plot_stats:
        mean = source.mark_rule(opacity=0.5).encode(x=f'mean({self.name}):Q', size=alt.value(2))
        low = source.mark_rule(opacity=0.3, strokeDash=[6, 4]).encode(x=f'q1({self.name}):Q', size=alt.value(1))
        high = source.mark_rule(opacity=0.3, strokeDash=[6, 4]).encode(x=f'q3({self.name}):Q', size=alt.value(1))
        return histogram + mean + low + high
    else:
        return histogram



def calculate_ci(self, alpha=0.05, is_pivotal=False):
    """Get confidence interval from given bootstrap distribution"""
    if is_pivotal:
        mid = np.mean(self.values)
        low = 2 * mid - np.percentile(self.values, 100 * (1 - alpha / 2.))
        high = 2 * mid - np.percentile(self.values, 100 * (alpha / 2.))
    else:
        low = np.percentile(self.values, 100 * (alpha / 2.))
        mid = np.percentile(self.values, 50)
        high = np.percentile(self.values, 100 * (1 - alpha / 2.))
    return {"low": low, "mid": mid, "high": high}
