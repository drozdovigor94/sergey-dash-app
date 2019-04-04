# -*- coding: utf-8 -*-

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

N_x_points = 200

# Функция для расчета компонент скорости V_x и V_y в точке (x,y) в зависимости от углов θ, β и параметров задачи

def V_fun(theta, x, y, beta, params):
    r_0 = params["r_0"]  # радиус Земли
    h_0 = params["h_0"]  # высота орбиты
    a = r_0 + h_0  # большая полуось орбиты
    mu = params["mu"]  # гравитационная постоянная
    n = np.sqrt(mu / a**3)
    omega =  params["omega"]  # угловая скорость вращения Земли
    e = params["e"]  # эксцентриситет орбиты
    i = np.deg2rad(params["i"])  # наклонение орбиты
    g = params["g"]  # долгота перигея
    d = params["d"]  # фокусное расстояние

    gamma_11 = np.cos(beta[1-1]) * np.cos(beta[3-1]) - np.sin(beta[1-1]) * np.cos(beta[2-1]) * np.sin(beta[3-1])
    gamma_21 = np.sin(beta[1-1]) * np.cos(beta[3-1]) + np.cos(beta[1-1]) * np.cos(beta[2-1]) * np.sin(beta[3-1])
    gamma_31 = np.sin(beta[2-1]) * np.sin(beta[3-1])

    gamma_12 = -np.cos(beta[1-1]) * np.sin(beta[3-1]) - np.sin(beta[1-1]) * np.cos(beta[2-1]) * np.cos(beta[3-1])
    gamma_22 = -np.sin(beta[1-1]) * np.sin(beta[3-1]) + np.cos(beta[1-1]) * np.cos(beta[2-1]) * np.cos(beta[3-1])
    gamma_32 = np.sin(beta[2-1]) * np.cos(beta[3-1])

    gamma_13 = np.sin(beta[1-1]) * np.sin(beta[2-1])
    gamma_23 = -np.cos(beta[1-1]) * np.sin(beta[2-1])
    gamma_33 = np.cos(beta[2-1])

    R = a * (1 - e ** 2) / (1 + e * np.cos(theta))
    Rdot = (a * n * e * np.sin(theta)) / np.sqrt(1 - e ** 2)
    thetadot = ((1 + e * np.cos(theta)) ** 2 / (1 - e ** 2) ** (3 / 2)) * n

    e_1z = np.sin(i) * np.cos(g + theta)
    e_2z = -np.cos(i)
    e_3z = -np.sin(i) * np.sin(g + theta)

    z_M = x * gamma_31 + y * gamma_32 - d * gamma_33

    u1 = (R * z_M + np.sqrt(R ** 2 * z_M ** 2 - (x ** 2 + y ** 2 + d ** 2) * (R ** 2 - r_0 ** 2))) \
        / (x ** 2 + y ** 2 + d ** 2)

    b_1 = (1 / d) * (Rdot * gamma_33 - R * thetadot * gamma_13 + R * omega * (gamma_23 * e_1z - gamma_13 * e_2z))
    b_2 = Rdot * gamma_31 - R * thetadot * gamma_11 - R * omega * (gamma_11 * e_2z - gamma_21 * e_1z)
    b_3 = -(1 / d) * (thetadot * gamma_22 + omega * (e_1z * gamma_12 + e_2z * gamma_22 + e_3z * gamma_32))
    b_4 = (1 / d) * (thetadot * gamma_21 + omega * (e_1z * gamma_11 + e_2z * gamma_21 + e_3z * gamma_31))
    b_5 = -omega * (e_1z * gamma_13 + e_2z * gamma_23 + e_3z * gamma_33) - thetadot * gamma_23
    b_6 = b_3 * d ** 2

    c_1 = b_1
    c_2 = Rdot * gamma_32 - R * thetadot * gamma_12 - R * omega * (gamma_12 * e_2z - gamma_22 * e_1z)
    c_3 = b_4
    c_4 = b_3
    c_5 = -b_5
    c_6 = c_3 * d ** 2

    V_1r = ((b_1 * x + b_2) / u1) + b_3 * (x ** 2) + b_4 * x * y + b_5 * y + b_6
    V_2r = ((c_1 * y + c_2) / u1) + c_3 * (y ** 2) + c_4 * y * x + c_5 * x + c_6

    return np.array([V_1r, V_2r])

# Вспомогательная функция для определения масштаба графика для наглядности изображения скоростей

def optimal_range(V_x_values, V_y_values):
    V_x_range = V_x_values.max() - V_x_values.min()
    V_y_range = V_y_values.max() - V_y_values.min()
    V_x_mean = (V_x_values.max() + V_x_values.min()) / 2
    V_y_mean = (V_y_values.max() + V_y_values.min()) / 2
    max_range = max(V_x_range, V_y_range)
    V_x_yrange = [V_x_mean - max_range/2 - 1, V_x_mean + max_range/2 + 1]
    V_y_yrange = [V_y_mean - max_range/2 - 1, V_y_mean + max_range/2 + 1]

    return V_x_yrange, V_y_yrange

# Функция для построения графиков СДИ в центре в зависимости от аномалии θ

def velocity_anomaly_plot(beta, params):

    # Вычисление значений

    theta_values = np.linspace(0, 2*np.pi, N_x_points)
    V_fun_theta = lambda theta : V_fun(theta, 0, 0, beta, params)
    V_fun_theta_vectorized = np.vectorize(V_fun_theta, signature='(b)->(2,n)')
    V_fun_values = V_fun_theta_vectorized(theta_values)
    V_fun_values = V_fun_values * 1000
    V_x_values = V_fun_values[0, :]
    V_y_values = V_fun_values[1, :]

    V_x_yrange, V_y_yrange = optimal_range(V_x_values, V_y_values)

    # Построение графика

    trace1 = go.Scatter(
        x = theta_values,
        y = V_x_values,
        mode = 'lines',
        name = r'V<sub>x<sub>'
    )
    trace2 = go.Scatter(
        x=theta_values,
        y=V_y_values,
        mode='lines',
        name=r'V<sub>y<sub>'
    )

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('V<sub>x</sub>', 'V<sub>y</sub>'))
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(
        font = dict(
            family = 'XITS Math, serif',
            size = 20
        ),
        xaxis = dict(
            title = u'θ',
            range = [0, 2*np.pi],
            tickvals = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
            ticktext = [u'0', u'π/2', u'π', u'3π/2', u'2π']
        ),
        yaxis = dict(
            title = "мм/сек",
            range = V_x_yrange
        ),
        xaxis2 = dict(
            title = u'θ',
            range = [0, 2*np.pi],
            tickvals=[0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            ticktext=[u'0', u'π/2', u'π', u'3π/2', u'2π']
        ),
        yaxis2 = dict(
            title="мм/сек",
            range=V_y_yrange
        )
    )
    for i in fig['layout']['annotations']:
        i['font'] = dict(
            family='Cambria Math, serif',
            size=25
        )
    return fig

# Функция для заполнения таблицы результатов

def result_table_data(beta_deg, params):
    beta = np.deg2rad(beta_deg)
    theta_values = np.linspace(0, 2*np.pi, N_x_points)
    V_fun_theta = lambda theta : V_fun(theta, 0, 0, beta, params)
    V_fun_theta_vectorized = np.vectorize(V_fun_theta, signature='(b)->(2,n)')
    V_fun_values = V_fun_theta_vectorized(theta_values)
    V_fun_values = V_fun_values * 1000
    V_x_values = V_fun_values[0, :]
    V_y_values = V_fun_values[1, :]

    V_x_0 = V_x_values[0]
    V_y_0 = V_y_values[0]
    delta_V_x = V_x_values.max() - V_x_values.min()
    delta_V_y = V_y_values.max() - V_y_values.min()

    data = [
        {
            'beta_1': beta_deg[0],
            'beta_2': beta_deg[1],
            'beta_3': beta_deg[2],
            'v_1x': f'{V_x_0:.4f}',
            'v_1y': f'{V_y_0:.4f}',
            'delta_v_1x': f'{delta_V_x:.4f}',
            'delta_v_1y': f'{delta_V_y:.4f}'
        }
    ]
    return data

# Функция для построения графиков СДИ в центре в зависимости от одного из углов β

def velocity_beta_plots(theta, betas, params, beta_dep):

    # Вычисление значений

    beta_dep_values = np.linspace(0, theta, N_x_points)

    V_funs_beta = {
        '1': [lambda beta, i=i: V_fun(theta, 0, 0, [beta, betas[i,0], betas[i,1]], params) for i in range(4)],
        '2': [lambda beta, i=i: V_fun(theta, 0, 0, [betas[i,0], beta, betas[i,1]], params) for i in range(4)],
        '3': [lambda beta, i=i: V_fun(theta, 0, 0, [betas[i,0], betas[i,1], beta], params) for i in range(4)],
    }

    V_funs_beta_vectorized = [np.vectorize(f, signature='(b)->(2,n)') for f in V_funs_beta[str(beta_dep)]]
    V_funs_values = [f(beta_dep_values) for f in V_funs_beta_vectorized]

    V_fun_values = np.dstack(V_funs_values)
    V_fun_values = V_fun_values * 1000
    V_x_values = V_fun_values[0,:,:]
    V_y_values = V_fun_values[1,:,:]

    # Построение графика

    V_x_yrange, V_y_yrange = optimal_range(V_x_values, V_y_values)

    traces_V_x = [go.Scatter(
        x=beta_dep_values,
        y=V_x_values[:,i],
        mode='lines',
        name='Случай '+str(i+1)
    ) for i in range(4)]

    traces_V_y = [go.Scatter(
        x=beta_dep_values,
        y=V_y_values[:,i],
        mode='lines',
        name='Случай '+str(i+1)
    ) for i in range(4)]

    layout1 = dict(
        width=650,
        height=720,
        title='V<sub>x</sub>',
        font=dict(
            family='XITS Math, serif',
            size=20
        ),
        xaxis=dict(
            title=u'β'+str(beta_dep)+u' (рад)',
            range=[0, theta]
        ),
        yaxis=dict(
            title="мм/сек",
            range=V_x_yrange
        ),
        # showlegend=False
    )
    layout2 = dict(
        width=650,
        height=720,
        title='V<sub>y</sub>',
        font=dict(
            family='XITS Math, serif',
            size=20
        ),
        xaxis=dict(
            title=u'β'+str(beta_dep)+u' (рад)',
            range=[0, theta]
        ),
        yaxis=dict(
            title="мм/сек",
            range=V_y_yrange
        ),
    )
    fig1 = dict(data=traces_V_x, layout=layout1)
    fig2 = dict(data=traces_V_y, layout=layout2)
    return fig1, fig2


# Функция для вычисления поля скоростей в зависимости от углов θ, β и параметров задачи в точках, задаваемых массивами x и y

def velocity_field(theta, beta, x, y, params):
    V_x0, V_y0 = V_fun(theta, 0, 0, beta, params)
    V_x, V_y = V_fun(theta, x, y, beta, params)

    V_x0 = V_x0 * 1000
    V_y0 = V_y0 * 1000
    V_x = V_x * 1000
    V_y = V_y * 1000

    V_x_diff = (V_x - V_x0)
    V_y_diff = (V_y - V_y0)

    return V_x_diff, V_y_diff


# Функция для построения поля скоростей

def velocity_field_plot(beta, params):

	# Вычисление значений

    thetas = np.linspace(0, 2 * np.pi, 50)
    x, y = np.meshgrid(np.arange(-60, 70, 10), np.arange(-40, 50, 10))
    x = x / 1000
    y = y / 1000

    fields = [velocity_field(theta, beta, x, y, params) for theta in thetas]

    # Построение графика

    vector_lengths = [np.sqrt(f[0] ** 2 + f[1] ** 2) for f in fields]
    vector_lengths_array = np.dstack(vector_lengths)
    scale_coef = np.sqrt(10 ** 2 + 10 ** 2) / vector_lengths_array.max()

    fields_scaled = [(v_x * scale_coef, v_y * scale_coef) for v_x, v_y in fields]
    figs = [ff.create_quiver(x * 1000, y * 1000, v_x, v_y, scale=1, scaleratio=1) for v_x, v_y in fields_scaled]

    main_fig = figs.pop(0)
    fig_datas = [fig.data[0] for fig in figs]
    main_fig.add_traces(fig_datas)

    for i in range(len(main_fig.data)):
        main_fig.data[i]['visible'] = False

    main_fig.data[0]['visible'] = True

    steps = []
    for i in range(len(main_fig.data)):
        step = dict(
            method='restyle',
            label=f'{thetas[i]:.2f}',
            args=['visible', [False] * len(main_fig.data)],
        )
        step['args'][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    slider = [dict(
        active=0,
        currentvalue={"prefix": "θ(рад) = "},
        pad={"t": 50},
        steps=steps
    )]

    main_fig['layout'].update(
        width=800,
        height=800,
        font=dict(
            family='XITS Math, serif',
            size=20
        ),
        xaxis=dict(
            range=[-80, 80],
            dtick=10,
            tickangle=45
        ),
        yaxis=dict(
            range=[-80, 80],
            dtick=10
        ),
        sliders=slider
    )

    return main_fig

params = {
    'r_0': 6.378137 * 10**6,
    'h_0': 2.5 * 10**5,
    'mu': 3.985586 * 10**14,
    'omega': 7.292115 * 10**(-5),
    'e': 0.001,
    'i': 97,
    'g': 0,
    'd': 1.5
}
param_table_data = [
    {'name': u'Высота орбиты (м)', 'symbol': 'h_0', 'value': params['h_0']},
    {'name': u'Эксцентриситет орбиты', 'symbol': 'e', 'value': params['e']},
    {'name': u'Наклонение орбиты (°)', 'symbol': 'i', 'value': params['i']},
    {'name': u'Долгота перигея', 'symbol': 'g', 'value': params['g']},
    {'name': u'Фокусное расстояние (м)', 'symbol': 'd', 'value': params['d']}
]

tab2_beta_table_data = [
    {'case': 1, 'beta_1st': 0, 'beta_2nd': 0},
    {'case': 2, 'beta_1st': 15, 'beta_2nd': 0},
    {'case': 3, 'beta_1st': 15, 'beta_2nd': 15},
    {'case': 4, 'beta_1st': 0, 'beta_2nd': 15},
]


tab1_fig = velocity_anomaly_plot([0, 0, 0], params)
tab2_fig_betas = np.array([[ 0,  0],
                           [15,  0],
                           [15, 15],
                           [ 0, 15]])
tab2_fig1, tab2_fig2 = velocity_beta_plots(np.deg2rad(60), np.deg2rad(tab2_fig_betas), params, 1)
tab3_fig = velocity_field_plot([0,0,0], params)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

tab1_content = [
    html.H2(u'СДИ в центре фокальной плоскости в зависимости от аномалии θ'),

    html.Div(
        [
            # Left div
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    'β',
                                    html.Sub('1'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab1-beta-1-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    'β',
                                    html.Sub('2'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab1-beta-2-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    'β',
                                    html.Sub('3'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab1-beta-3-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id='tab1-param_table',
                            editable=True,
                            columns=[
                                {'name': u'Параметр', 'id': 'name', 'editable': False},
                                {'name': u'Обозначение', 'id': 'symbol', 'editable': False},
                                {'name': u'Значение', 'id': 'value', 'type': 'numeric', 'editable': True},
                            ],
                            data=param_table_data,
                            style_cell={'fontSize': '17'},
                            style_header={
                                'fontWeight': 'bold'
                            }
                        ),
                        style={'padding-top': '20px'},
                    )
                ],
                style={'vertical-align': 'middle',
                       'float': 'left',
                       'padding-top': '5%',
                       'width': '410px',
                       'margin-left': '20px'}
            ),

            # Right div
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id='tab1-velocity-anomaly-plot',
                            figure=tab1_fig
                        )
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id='tab1-result-table',
                            editable=True,
                            columns=[
                                {'name': u'β1°', 'id': 'beta_1'},
                                {'name': u'β2°', 'id': 'beta_2'},
                                {'name': u'β3°', 'id': 'beta_3'},
                                {'name': 'V_1x(0)', 'id': 'v_1x'},
                                {'name': 'V_1y(0)', 'id': 'v_1y'},
                                {'name': 'Δ V_1x', 'id': 'delta_v_1x'},
                                {'name': 'Δ V_1y', 'id': 'delta_v_1y'},
                            ],
                            data=result_table_data([0, 0, 0], params),
                            style_cell={'fontSize': '17'},
                            style_header={
                                'fontWeight': 'bold'
                            }
                        ),
                        style={
                            'padding-top': '20px',
                            'width': '50%',
                            'margin': '0 auto'
                        },
                    )
                ],
                style={
                    'width': 'auto',
                    'overflow': 'hidden',
                    'padding-right': '20px',
                    'padding-left': '30px'
                }
            )
        ],
        style={
            'fontSize': 16
        }
    ),

]

tab2_content = [
    html.H2(u'СДИ в центре фокальной плоскости в зависимости от одного из углов Эйлера β'),

    html.Div(
        [
            # Left div
            html.Div(
                [
                    html.Div(
                        [
                            'θ°',
                            html.Span(
                                dcc.Input(
                                    id='tab2-theta-input',
                                    placeholder='',
                                    type='number',
                                    value=60
                                ),
                                style={'padding-left': '20px'}
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Div('Изменяющийся угол'),
                            dcc.RadioItems(
                                id='tab2-beta-radio',
                                options=[
                                    {'label': 'β1', 'value': 1},
                                    {'label': 'β2', 'value': 2},
                                    {'label': 'β3', 'value': 3}
                                ],
                                value=1
                            ),
                        ],
                        style={'padding-top': '20px'},
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id='tab2-beta-table',
                            editable=True,
                            columns=[
                                {'name': u'Случай #', 'id': 'case'},
                                {'name': u'β2°', 'id': 'beta_1st', 'type': 'numeric'},
                                {'name': u'β3°', 'id': 'beta_2nd', 'type': 'numeric'}
                            ],
                            data=tab2_beta_table_data,
                            style_cell={'fontSize': '17'},
                            style_header={
                                'fontWeight': 'bold'
                            }
                        ),
                        style={'padding-top': '20px'},
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id='tab2-param-table',
                            editable=True,
                            columns=[
                                {'name': u'Параметр', 'id': 'name', 'editable': False},
                                {'name': u'Обозначение', 'id': 'symbol', 'editable': False},
                                {'name': u'Значение', 'id': 'value', 'type': 'numeric', 'editable': True},
                            ],
                            data=param_table_data,
                            style_cell={'fontSize': '17'},
                            style_header={
                                'fontWeight': 'bold'
                            }
                        ),
                        style={'padding-top': '20px'},
                    )
                ],
                style={'vertical-align': 'middle',
                       'float': 'left',
                       'padding-top': '5%',
                       'width': '410px',
                       'margin-left': '20px'}
            ),

            # Right div
            html.Div(
                [
                    dcc.Graph(
                        id='tab2-velocity-x-beta-plot',
                        figure=tab2_fig1,
                        style={
                            #'float': 'left',
                            'display': 'inline-block',
                            'width': 'auto'
                        }
                    ),
                    dcc.Graph(
                        id='tab2-velocity-y-beta-plot',
                        figure=tab2_fig2,
                        style={
                            #'float': 'right',
                            'display': 'inline-block',
                            'width': 'auto'
                        }
                    ),
                ],
                style={
                    'width': 'auto',
                    'overflow': 'hidden',
                    'padding-right': '20px',
                    'padding-left': '30px'
                }
            )
        ]
    )
]

tab3_content = [
    html.Div(
        [
            # Left div
            html.Div(
                [
                    html.H2(u'Поле скоростей'),
                    html.Div(
                        [
                            html.Div(
                                [
                                    'θ (рад)',
                                    html.Span(
                                        dcc.Input(
                                            id='tab3-theta-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    'β',
                                    html.Sub('1'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab3-beta-1-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    'β',
                                    html.Sub('2'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab3-beta-2-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    'β',
                                    html.Sub('3'),
                                    u'°',
                                    html.Span(
                                        dcc.Input(
                                            id='tab3-beta-3-input',
                                            placeholder='',
                                            type='number',
                                            value=0
                                        ),
                                        style={'padding-left': '20px'}
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        dash_table.DataTable(
                            id='tab3-param-table',
                            editable=True,
                            columns=[
                                {'name': u'Параметр', 'id': 'name', 'editable': False},
                                {'name': u'Обозначение', 'id': 'symbol', 'editable': False},
                                {'name': u'Значение', 'id': 'value', 'type': 'numeric', 'editable': True},
                            ],
                            data=param_table_data,
                            style_cell={'fontSize': '17'},
                            style_header={
                                'fontWeight': 'bold'
                            }
                        ),
                        style={'padding-top': '20px'},
                    )
                ],
                style={'display': 'inline-block',
                       'vertical-align': 'middle',
                       'float': 'left',
                       'padding-top': '5%',
                       'width': '20vw',
                       'margin-left': '20px'}
            ),

            # Right div
            html.Div(
                [
                    html.Div(
                        dcc.Graph(
                            id='tab3-velocity-field',
                            figure=tab3_fig
                        ),
                        style={
                            'width': '80%',
                            'margin': '0 auto'
                        }
                    ),
                    html.Div(
                        html.P(
                            id='tab3-summary',
                            children='Placeholder for summary',
                            style={
                                'font-size': '18'
                            }
                        ),
                        style={
                            'padding-top': '20px'
                        }
                    ),
                    # html.Div(
                    #     [
                    #         dcc.Slider(
                    #             id='tab3-theta-slider',
                    #             min=0,
                    #             max=2*np.pi,
                    #             marks={
                    #                 0: {'label': u'0', 'style': {'font-size': '16'}},
                    #                 np.pi / 6: {'label': u'π/6', 'style': {'font-size': '16'}},
                    #                 np.pi / 4: {'label': u'π/4', 'style': {'font-size': '16'}},
                    #                 np.pi / 3: {'label': u'π/3', 'style': {'font-size': '16'}},
                    #                 np.pi / 2: {'label': u'π/2', 'style': {'font-size': '16'}},
                    #                 2 * np.pi / 3: {'label': u'2π/3', 'style': {'font-size': '16'}},
                    #                 3 * np.pi / 4: {'label': u'3π/4', 'style': {'font-size': '16'}},
                    #                 5 * np.pi / 6: {'label': u'5π/6', 'style': {'font-size': '16'}},
                    #                 np.pi: {'label': u'π', 'style': {'font-size': '16'}},
                    #                 7 * np.pi / 6: {'label': u'7π/6', 'style': {'font-size': '16'}},
                    #                 5 * np.pi / 4: {'label': u'5π/4', 'style': {'font-size': '16'}},
                    #                 4 * np.pi / 3: {'label': u'4π/3', 'style': {'font-size': '16'}},
                    #                 3 * np.pi / 2: {'label': u'3π/2', 'style': {'font-size': '16'}},
                    #                 5 * np.pi / 3: {'label': u'5π/3', 'style': {'font-size': '16'}},
                    #                 7 * np.pi / 4: {'label': u'7π/4', 'style': {'font-size': '16'}},
                    #                 11 * np.pi / 6: {'label': u'11π/6', 'style': {'font-size': '16'}},
                    #                 2 * np.pi: {'label': u'2π', 'style': {'font-size': '16'}},
                    #             },
                    #             step=0.01,
                    #             updatemode='drag',
                    #             value=0,
                    #         ),
                    #         html.Center(u'θ',
                    #                     style={
                    #                         'padding-top': '30px'
                    #                     })
                    #     ],
                    #     style={
                    #         'padding-top': '20px',
                    #         'margin': '0 auto',
                    #         'width': '85%',
                    #         'font-size': '18',
                    #         'font-family': 'XITS Math, serif'
                    #     }
                    # )
                ],
                style={'display': 'inline-block',
                       'width': '70vw',
                       'float': 'right',
                       'padding-right': '20px'}
            )
        ]
    )
]

#tabs_content = [tab1_content, tab2_content, tab3_content]

app.layout = html.Div(children=[
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='СДИ в центре от θ', value='tab1'),
        dcc.Tab(label='СДИ в центре от β', value='tab2'),
        dcc.Tab(label='Поле скоростей', value='tab3')
    ]),
    html.Div(id='current-tab-content')
])

app.config['suppress_callback_exceptions'] = True


@app.callback(Output('current-tab-content', 'children'),
              [Input('tabs', 'value')])
def render_content(selected_tab):
    if selected_tab == 'tab1':
        return tab1_content
    elif selected_tab == 'tab2':
        return tab2_content
    elif selected_tab == 'tab3':
        return tab3_content
    else:
        return "Some error happened when selecting a tab"

# region tab1-callbacks
@app.callback(
    Output('tab1-velocity-anomaly-plot', 'figure'),
    [Input('tab1-beta-1-input', 'n_blur'),
     Input('tab1-beta-2-input', 'n_blur'),
     Input('tab1-beta-3-input', 'n_blur'),
     Input('tab1-param_table', 'data'),
     Input('tab1-param_table', 'columns')],
    [State('tab1-beta-1-input', 'value'), State('tab1-beta-2-input', 'value'), State('tab1-beta-3-input', 'value')]
)
def update_velocity_anomaly_plot(nb1, nb2, nb3, rows, columns, input1, input2, input3):

    beta1 = np.deg2rad(input1)
    beta2 = np.deg2rad(input2)
    beta3 = np.deg2rad(input3)
    beta = [beta1, beta2, beta3]

    new_params = {r['symbol']: r['value'] for r in rows}
    new_params = {**params, **new_params}

    return velocity_anomaly_plot(beta, new_params)


@app.callback(
    Output('tab1-result-table', 'data'),
    [Input('tab1-beta-1-input', 'n_blur'),
     Input('tab1-beta-2-input', 'n_blur'),
     Input('tab1-beta-3-input', 'n_blur'),
     Input('tab1-param_table', 'data'),
     Input('tab1-param_table', 'columns')],
    [State('tab1-beta-1-input', 'value'), State('tab1-beta-2-input', 'value'), State('tab1-beta-3-input', 'value')]
)
def update_result_table(nb1, nb2, nb3, rows, columns, input1, input2, input3):
    beta = [input1, input2, input3]
    new_params = {r['symbol']: r['value'] for r in rows}
    new_params = {**params, **new_params}

    return result_table_data(beta, new_params)
# endregion


# region tab 2 callbacks
@app.callback(
    Output('tab2-beta-table', 'columns'),
    [Input('tab2-beta-radio', 'value')]
)
def update_beta_table_columns(value):
    column_variants = {
        '1': [{'name': u'Случай #', 'id': 'case'},
              {'name': u'β2°', 'id': 'beta_1st', 'type': 'numeric'},
              {'name': u'β3°', 'id': 'beta_2nd', 'type': 'numeric'}],
        '2': [{'name': u'Случай #', 'id': 'case'},
              {'name': u'β1°', 'id': 'beta_1st', 'type': 'numeric'},
              {'name': u'β3°', 'id': 'beta_2nd', 'type': 'numeric'}],
        '3': [{'name': u'Случай #', 'id': 'case'},
              {'name': u'β1°', 'id': 'beta_1st', 'type': 'numeric'},
              {'name': u'β2°', 'id': 'beta_2nd', 'type': 'numeric'}]
    }
    return column_variants[str(value)]

@app.callback(
    [Output('tab2-velocity-x-beta-plot', 'figure'),
     Output('tab2-velocity-y-beta-plot', 'figure')],
    [Input('tab2-theta-input', 'n_blur'),
     Input('tab2-beta-radio', 'value'),
     Input('tab2-beta-table', 'data'),
     Input('tab2-param-table', 'data')],
    [State('tab2-theta-input', 'value')]
)
def update_velocity_beta_plots(nb_theta, value_beta_radio, beta_table_data, param_table_data, input_theta):

    theta = np.deg2rad(input_theta)
    beta_dep = value_beta_radio

    betas = [[r['beta_1st'], r['beta_2nd']] for r in beta_table_data]
    betas = np.array(betas)
    betas = np.deg2rad(betas)

    new_params = {r['symbol']: r['value'] for r in param_table_data}
    new_params = {**params, **new_params}

    return velocity_beta_plots(theta, betas, new_params, beta_dep)
# endregion

@app.callback(
    Output('tab3-velocity-field', 'figure'),
    [Input('tab3-beta-1-input', 'n_blur'),
     Input('tab3-beta-2-input', 'n_blur'),
     Input('tab3-beta-3-input', 'n_blur'),
     Input('tab3-param-table', 'data'),
     Input('tab3-param-table', 'columns')],
    [State('tab3-beta-1-input', 'value'), State('tab3-beta-2-input', 'value'), State('tab3-beta-3-input', 'value')]
)
def update_velocity_plot(nb1, nb2, nb3, rows, columns, input1, input2, input3):
    beta1 = np.deg2rad(input1)
    beta2 = np.deg2rad(input2)
    beta3 = np.deg2rad(input3)
    beta = [beta1, beta2, beta3]

    new_params = {r['symbol']: r['value'] for r in rows}
    new_params = {**params, **new_params}

    fig = velocity_field_plot(beta, new_params)

    return fig

@app.callback(
    Output('tab3-summary', 'children'),
    [Input('tab3-theta-input', 'n_blur'),
     Input('tab3-beta-1-input', 'n_blur'),
     Input('tab3-beta-2-input', 'n_blur'),
     Input('tab3-beta-3-input', 'n_blur'),
     Input('tab3-param-table', 'data'),
     Input('tab3-param-table', 'columns')],
    [State('tab3-theta-input', 'value'),
     State('tab3-beta-1-input', 'value'),
     State('tab3-beta-2-input', 'value'),
     State('tab3-beta-3-input', 'value')]
)
def update_summary(nb1, nb2, nb3, nb4, rows, columns, input1, input2, input3, input4):
    theta = input1
    beta1 = np.deg2rad(input2)
    beta2 = np.deg2rad(input3)
    beta3 = np.deg2rad(input4)
    beta = [beta1, beta2, beta3]

    new_params = {r['symbol']: r['value'] for r in rows}
    new_params = {**params, **new_params}

    x, y = np.meshgrid(np.arange(-60, 70, 10), np.arange(-40, 50, 10))
    x = x / 1000
    y = y / 1000

    V_x_diff, V_y_diff = velocity_field(theta, beta, x, y, new_params)

    vector_length = np.sqrt(V_x_diff ** 2 + V_y_diff ** 2)
    max_idx = np.unravel_index(vector_length.argmax(), vector_length.shape)

    summary = f'Максимальная разница составляет [{V_x_diff[max_idx]:.3f}  {V_y_diff[max_idx]:.3f}] в точке [{x[max_idx]*1000}  {y[max_idx]*1000}]'

    return summary



if __name__ == '__main__':
    app.run_server(debug=True)
