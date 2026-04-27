import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def sorted_experiments(series, data):
    """Return experiments in a series sorted by observation_radius attribute."""
    return sorted(data[series], key=lambda exp: data[series][exp].attrs['observation_radius'])

def get_x_radii(series, data):
    """Read the observation_radius attribute for each experiment, sorted ascending."""
    return [data[series][exp].attrs['observation_radius'] for exp in sorted_experiments(series, data)]


def _hex_to_rgba(color, alpha=0.2):
    """Convert a hex color string to an rgba string with the given alpha."""
    color = color.lstrip('#')
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def plot(plots, timesteps=None):
    for plot in plots:
        fig = go.Figure()
        x = plots[plot].get('x', None)
        y_data = plots[plot]['y']
        y_std_data = plots[plot].get('y_std', None)
        show_legend = True

        # Detect nested dict: {series: {experiment: array}} — data with a time dimension.
        is_time_series = (
            isinstance(y_data, dict)
            and len(y_data) > 0
            and isinstance(next(iter(y_data.values())), dict)
        )

        if is_time_series:
            series_names = list(y_data.keys())
            series_names = [name.split(" ")[0] for name in series_names]
            series_names = set(series_names)
            series_colors = {name: color for name, color in zip(series_names, px.colors.qualitative.Plotly)}
            for series, tests in y_data.items():
                series_name = series.split(" ")[0]
                color = series_colors.get(series_name, '#888888')
                for test_name, values in tests.items():
                    x_local = np.arange(1, len(values) + 1)
                    # Add shaded std band if y_std is provided.
                    if y_std_data and series in y_std_data and test_name in y_std_data[series]:
                        std_vals = y_std_data[series][test_name]
                        upper = values + std_vals
                        lower = values - std_vals
                        fill_color = _hex_to_rgba(color, alpha=0.2) if color.startswith('#') else 'rgba(128,128,128,0.2)'
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([x_local, x_local[::-1]]),
                            y=np.concatenate([upper, lower[::-1]]),
                            fill='toself',
                            fillcolor=fill_color,
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo='skip',
                            legendgroup=series_name,
                        ))
                    fig.add_trace(go.Scatter(
                        x=x_local,
                        y=values,
                        mode='lines',
                        name=f"{series} | {test_name}",
                        hovertemplate=f'{series} | {test_name}' + ': %{y:.4f}<extra></extra>',
                        line={
                            "color": color,
                        },
                        legendgroup=series_name,
                        # legendgrouptitle={series_name: series_name},
                    ))
        elif isinstance(y_data, dict):
            for series, values in y_data.items():
                # x may be a per-series dict (keyed by series name) or a shared list.
                x_local = x[series] if isinstance(x, dict) else x
                if x_local is None:
                    x_local = np.linspace(0, 1, len(values))
                std_vals = y_std_data[series] if y_std_data and series in y_std_data else None
                if len(values) == 1:
                    # Plot a bar instead of a point if there's only one value, to make it more visible.
                    fig.add_trace(go.Bar(
                        x=[series],
                        y=values,
                        name=series,
                        hovertemplate=f'{series}' + ': %{y:.4f}<extra></extra>',
                        error_y=dict(type='data', array=std_vals, visible=True) if std_vals is not None else None,
                    ))
                    show_legend = False  # Legend is not needed if we're labeling bars directly.
                else:
                    fig.add_trace(go.Scatter(
                        x=x_local,
                        y=values,
                        mode='lines+markers',
                        name=series,
                        hovertemplate='%{x:.2f}: %{y:.4f}<extra></extra>',
                        error_y=dict(type='data', array=std_vals, visible=True) if std_vals is not None else None,
                    ))
        else:
            values = y_data
            x_local = x
            if x_local is None:
                x_local = np.linspace(0, 1, len(values))
            fig.add_trace(go.Scatter(
                x=x_local,
                y=values,
                mode='lines+markers',
                hovertemplate='%{x:.2f}: %{y:.4f}<extra></extra>',
                error_y=dict(type='data', array=y_std_data, visible=True) if y_std_data is not None else None,
            ))

        # Collect the union of all x tick values across series, if x is defined.
        if x is not None and not is_time_series:
            all_x_vals = sorted(set(v for vals in (x.values() if isinstance(x, dict) else [x]) for v in vals))
            xaxis_opts = dict(tickvals=all_x_vals, ticktext=[str(v) for v in all_x_vals])
        else:
            xaxis_opts = {}

        title = plots[plot]['title']
        xlabel = 'Scenario' if "xlabel" not in plots[plot] else plots[plot]['xlabel']
        ylabel = f"Timesteps out of {timesteps}" if "ylabel" not in plots[plot] else plots[plot]['ylabel']
        
        fig.update_layout(
            # title=f"{title}: avg. over {args['eval_episodes']['value']} episodes, seed {args['seed']['value']}",
            title=f"{title}",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis=xaxis_opts,
            hovermode='closest',
            showlegend=show_legend,
            autosize=True,
            height=600,
            margin=dict(l=60, r=20, t=60, b=120),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.2,
                xanchor='center',
                x=0.5
            ),
        )
        
        fig.show()