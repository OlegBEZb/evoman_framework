import os

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.io as pio

def get_stats(directory, experiment_name):
    fitness_mean = {}
    fitness_max = {}
    individualgain_mean = []

    i = 1
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith(experiment_name):
            results = os.path.join(directory, filename) + '/results.csv'
            test_results = os.path.join(directory, filename) + '/test_results.txt'
            run = pd.read_csv(results, sep=",", index_col='generation')
            fitness_mean['Run '+str(i)] = run['fitness_mean'].tolist()
            fitness_max['Run '+str(i)] = run['best_score'].tolist()
            try:
                igain5test = pd.read_csv(test_results, header=None)
                individualgain_mean.append(igain5test.mean().tolist()[0])
            except:
                print("no test")
            i += 1
        else:
            continue

    fitness_meanDf = pd.DataFrame(fitness_mean)
    fitness_maxDf = pd.DataFrame(fitness_max)

    fitness_meanDf['mean'] = fitness_meanDf.mean(axis=1)
    fitness_maxDf['mean'] = fitness_maxDf.mean(axis=1)

    fitness_meanDf['std'] = fitness_meanDf.std(axis=1)
    fitness_maxDf['std'] = fitness_maxDf.std(axis=1)

    return fitness_meanDf, fitness_maxDf, individualgain_mean

def get_upper_line(fitness, sd):
    return fitness + sd

def get_lower_line(fitness, sd):
    return fitness - sd

def lineplot(x, y, name, color = 'rgb(0,100,80)', dash = 'solid'):
    lineplot = go.Scatter(
        x=x,
        y=y,
        name=name,
        line=dict(color=color, dash=dash),
        mode='lines')
    return lineplot

def errorbands(x, fitness, sd, fillcolor = 'rgba(0,100,80,0.2)'):
    y_upper = get_upper_line(fitness, sd)
    y_lower = get_lower_line(fitness, sd)

    errorbands = [go.Scatter(
        x=x,
        y=y_upper,
        mode = 'lines',
        line=dict(width=0),
        showlegend=False),
    go.Scatter(
        x=x,  # x, then x reversed
        y=y_lower,  # upper, then lower reversed
        line=dict(width=0),
        mode='lines',
        fillcolor=fillcolor,
        fill='tonexty',
        showlegend=False)]
    return errorbands

def create_lineplot(ea1data, ea2data):
    ngen1 = len(ea1data[0])
    ngen2 = len(ea2data[0])
    generation1 = np.linspace(1, ngen1, ngen1)
    generation2 = np.linspace(1, ngen2, ngen2)

    meanEA1 = ea1data[:2]
    maxEA1 = ea1data[2:]
    meanEA2 = ea2data[:2]
    maxEA2 = ea2data[2:]

    errorbandEA1_mean = errorbands(generation1, meanEA1[0], meanEA1[1])
    errorbandEA1_max = errorbands(generation1, maxEA1[0], maxEA1[1])
    errorbandEA2_mean = errorbands(generation2, meanEA2[0], meanEA2[1], fillcolor='rgba(217,95,2,0.2)')
    errorbandEA2_max = errorbands(generation2, maxEA2[0], maxEA2[1], fillcolor='rgba(217,95,2,0.2)')

    fig = go.Figure([
        lineplot(generation1, meanEA1[0], 'mean fitness EA1'),
        errorbandEA1_mean[0],
        errorbandEA1_mean[1],
        lineplot(generation1, maxEA1[0], 'max fitness EA1', dash='longdash'),
        errorbandEA1_max[0],
        errorbandEA1_max[1],
        lineplot(generation2, meanEA2[0], 'mean fitness EA2', color='rgb(217,95,2)'),
        errorbandEA2_mean[0],
        errorbandEA2_mean[1],
        lineplot(generation2, maxEA2[0], 'max fitness EA2', color='rgb(217,95,2)', dash='longdash'),
        errorbandEA2_max[0],
        errorbandEA2_max[1]])

    fig.update_layout(
        font_family = 'Serif',
        font_color = 'black',
        font_size=20,
        xaxis_title='Generation',
        yaxis_title='Fitness',
        title='<b>Enemy '+enemynumber+'</b>',
        title_x=0.5,
        title_font_size=25,
        legend=dict(
            font_size=20,
            bgcolor = 'rgba(255,255,255,0.4)',
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99)
    )
    return fig

def create_boxplot(gainmeansEA1: list, gainmeansEA2: list):
    fig = go.Figure()
    fig.add_trace(go.Box(y=gainmeansEA1, name='EA1', line=dict(color='rgb(0,100,80)'), showlegend=False))
    fig.add_trace(go.Box(y=gainmeansEA2, name='EA2', line=dict(color='rgb(217,95,2)'), showlegend=False))
    fig.update_layout(
        font_family='Serif',
        font_color='black',
        xaxis_title='Evolutionary Algorithm',
        yaxis_title='Fitness mean',
        title='<b>Enemy ' + enemynumber + '</b>',
        title_x=0.5,
        title_font_size=20,
        legend=dict(
            bgcolor='rgba(255,255,255,0.4)',
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99))
    return fig

def get_data(dataframe_list:list):
    data = []
    for element in dataframe_list:
        for stat in ['mean','std']:
            data.append(np.array(element[stat].tolist()))
    return data


#GET RESULTS
enemyList = ['2','3','5']#,'5','6','7','8']
directory = os.path.dirname(os.path.realpath(__file__)) + '/experiments'
EA1 = 'tournamentselProportional{k_2}_mating3_pop100_patience17_DPR0.3_DRWRP0.75_mutGaus_mu0sigma1prob0.8_cxUniform0.6'
EA2 = 'mating4_pop100_patience3_DPR0.4_DRWRP0.85_mutGaus_mu0sigma1prob03_cxUniform_4parents_expfitness'

for enemynumber in enemyList:
    print("Enemy"+enemynumber)
    experiment_names = ['enemy'+enemynumber+'_'+EA1,'enemy'+enemynumber+'_'+EA2]

    fitness_meanDf1, fitness_maxDf1, individualgain_mean1 = get_stats(directory, experiment_names[0])
    fitness_meanDf2, fitness_maxDf2, individualgain_mean2 = get_stats(directory, experiment_names[1])

    EA1data = get_data([fitness_meanDf1,fitness_maxDf1])
    EA2data = get_data([fitness_meanDf2, fitness_maxDf2])

    if not os.path.exists("plots"):
        os.mkdir("plots")

    pio.write_image(create_lineplot(EA1data, EA2data), "plots/fitnessEnemy"+enemynumber+".png", format="png", scale=1.5)

    pio.write_image(create_boxplot(individualgain_mean1,individualgain_mean2),"plots/boxplotEnemy"+enemynumber+".png", format="png", scale=1.5)

    EAlist = ["EA1"]*10+["EA2"]*10
    boxplotData = []
    for ea,value in zip(EAlist,individualgain_mean1+individualgain_mean2):
        boxplotData.append([ea,value])

    boxplotDf = pd.DataFrame(boxplotData, columns = ["EA","fitness"])
    boxplotDf.to_csv('plots/boxplotEnemy'+str(enemynumber)+'.csv')