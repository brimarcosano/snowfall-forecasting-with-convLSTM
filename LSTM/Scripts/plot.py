import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import plotly.offline as pyo
pio.templates.default = "plotly_white"

def plot_snowfall_predictions(df, train_end, val_end, test_end):
    fig = go.Figure()

    # Plot actual snowfall
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Target_Snowfall'],
        name='Actual Snowfall',
        mode='lines',
        line=dict(color='black', width=.9)
    ))

    # Plot model forecast
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Model forecast'],
        name='Model Forecast', 
        mode='lines',
        line=dict(color='lawngreen', width=1.1)
    ))

    # Add vertical lines using shape
    fig.add_shape(
        type="line",
        x0=train_end,
        x1=train_end,
        y0=0,
        y1=df['Target_Snowfall'].max(),
        line=dict(color="black", width=2, dash="dash")
    )

    fig.add_shape(
        type="line", 
        x0=val_end,
        x1=val_end,
        y0=0,
        y1=df['Target_Snowfall'].max(),
        line=dict(color="gray", width=2, dash="dash")
    )


    # Add annotations for the splits
    fig.add_annotation(
        x=train_end,
        y=df['Target_Snowfall'].max(),
        text="Validation Start",
        showarrow=False,
        yshift=10
    )

    fig.add_annotation(
        x=val_end,
        y=df['Target_Snowfall'].max(),
        text="Test Start",
        showarrow=False,
        yshift=10
    )

    fig.add_shape(
        type="line", 
        x0=test_end,
        x1=test_end,
        y0=0,
        y1=df['Target_Snowfall'].max(),
        line=dict(color="gray", width=2, dash="dash")
    )

    # Update layout with improved date formatting
    fig.update_layout(
        title="Snowfall: Actual vs Forecast",
        xaxis=dict(
            title="Date",
            type='date',
            tickformat='%b %d, %Y',  # Format for main view
            hoverformat='%B %d, %Y', # Format for hover
            rangeslider=dict(visible=True),
            range=[df.index.min(), df.index.max()]
        ),
        yaxis=dict(
            title="Snowfall (inches)",
            gridcolor='lightgray'
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        plot_bgcolor='white'
    )

    return fig


def plot_tricolor(predicts_df, train_preds, val_preds, test_preds, dfs, actuals_col):
    plot_template = go.Layout(
        title="Snowfall Prediction",
        xaxis=dict(title='Date',
                rangeslider=dict(
                visible = True
            ),
            type='date'
        ),
        yaxis=dict(title='Snowfall Actuals (in)', color='black'),
        yaxis2=dict(title='Snow', color='blue',
        overlaying='y', side='right')
    )

    fig = go.Figure(layout=plot_template)   

    line_colors = ['gold', 'powderblue', 'lawngreen']

    for i, df in enumerate(dfs):
        fig.add_trace(go.Scatter(x = dfs[df].index,
                                    y = dfs[df]["Model forecast"], 
                                    name = df,
                                    mode = "lines",
                                    connectgaps=True,
                                    #fill='tozeroy',
                                    #    fillcolor=line_colors[i],
                                    line=dict(color=line_colors[i], width = 1.2)
                                    ))

    fig.add_trace(go.Scatter(x = predicts_df.index,
                            y = predicts_df[actuals_col], 
                            name = 'Actuals',
                            mode = "lines",
                            # fill='tozeroy',
                            # fillcolor="rgba(105, 104, 102, 0.9)",
                            line=dict(color='black', width=1),
                            opacity=0.8
                            ))

    fig.add_trace(go.Scatter(
        x=[val_preds.index[0], val_preds.index[0]],
        y=[min(train_preds[actuals_col]), max(train_preds[actuals_col])],
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name="Validation set start",
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=[test_preds.index[0], test_preds.index[0]],
        y=[min(train_preds[actuals_col]), max(train_preds[actuals_col])],
        mode='lines',
        line=dict(color='black', width=3, dash='dot'),
        name="Test set start",
        connectgaps=True
    ))

    # fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text=f"LR:{lr}, batch size:{batch_size},"
    #                 f"epochs:{num_epochs}", showarrow=False)
    # if start_date_show:
    #     fig.update_xaxes(dtick='M1', range=[start_date_show, preds_df.index.max()])
    # else:
    # fig.update_xaxes(dtick='M1')


    fig.show(config=dict(editable=False))

def plot_bicolor(predicts_df, train_preds, val_preds, test_preds, actuals_col):
    pt = dict(
    layout=go.Layout({
        "title": "Snowfall Prediction",
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
    )

    fig = px.line(predicts_df, labels=dict(created_at="Date", value="Inches Snow"))

    fig.add_trace(go.Scatter(
        x=[val_preds.index[0], val_preds.index[0]],
        y=[min(train_preds[actuals_col]), max(train_preds[actuals_col])],
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        name="Validation set start",
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=[test_preds.index[0], test_preds.index[0]],
        y=[min(train_preds[actuals_col]), max(train_preds[actuals_col])],
        mode='lines',
        line=dict(color='black', width=3, dash='dot'),
        name="Test set start",
        connectgaps=True
    ))
    fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text=f"LR:{lr},"
                    f" shuffle=False", showarrow=False)
    fig.update_layout(
        template=pt, legend=dict(orientation='h', y=1.02)
    )
    fig.show()


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='train loss')
    # plt.plot(test_losses, label='test loss')
    plt.plot(val_losses, label='val loss')
    plt.xlabel('epoch no')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


    

# ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax3 = ax2.twinx()

# dates = predicts_df.index
# ax1.plot(dates, predicts_df[actuals_col], c='b', label='All Actuals')
# ax1.plot(dates[:len(train_preds)], train_preds[actuals_col], label='Train Prediction', c='r')
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Train pred', color='r')
# ax1.legend()
# # Plot on the twin y-axis
# ax2.plot(val_preds.index, val_preds[ystar_col], label='Val Prediction', c='orange')
# ax2.set_ylabel('Val Pred', color='orange')
# ax2.legend(loc='upper left')
# ax2.set_ylim(ax1.get_ylim())
# # ax3
# ax3.plot(test_preds.index, test_preds[ystar_col], label='Test Prediction', c='g')
# ax3.set_ylabel('Test Pred', color='g')
# ax3.legend(loc='upper left')
# ax3.set_ylim(ax2.get_ylim())

# start_date = pd.to_datetime('2013-01-01')
# plt.xlim(start_date, dates[-1])

# plt.show()