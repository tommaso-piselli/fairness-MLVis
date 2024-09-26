import reader
import torch
import model
import csvHandler
from datetime import datetime
import time
from utils import *
import argparse

# Set up argument parser
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use the index from command-line argument
name = f'graph_spa_500_4'
dataset = f'AVN'
filepath = f'data/{dataset}/{name}.graphml'

# Rest of your script goes here
feature = 'region'

print(f"Processing {name} graph with feature:{feature}")
lr = 0.01
maxIters = 15000

graph = reader.Graph(name, filepath)
graph.load_graphml(feature, dataset)  
if dataset == 'AVN':
    model = model.Model_AVN(feature, lr, maxIters)
elif dataset == 'DrugNet':
    model = model.Model_DrugNet(feature, lr, maxIters)


csv_handler = csvHandler.Handler()
csv_handler.set_path_csv(graph, feature=feature, dataset=dataset)
csv_handler.init_csv()

for _ in range(10):
    now = datetime.now()
    print(f'\nIteration {_}')
    print(f'[Start Time: {now.strftime("%d/%m/%Y %H:%M:%S")}]')

    graph.set_pos()
    savepath_img = f'output/figs/{dataset}/{
        feature}/{graph.name}/{graph.name}_{_}'
    

    start = time.time()
    stress_min, unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20 = model.train(
        graph, savepath_img)
    end = time.time()
    time_elapsed = end - start
    print(f'[Computation time: {round(time_elapsed, 2)}s]')

    csv_row = [graph.name, _, stress_min,
               unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20, time_elapsed]

    csv_handler.write_csv(csv_row)
