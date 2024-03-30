import reader
import torch
import model
import csvHandler
from datetime import datetime
import time


# Variables
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

name = 'lesmis'
filepath = f'data/{name}.mat'
lr = 0.01
maxIters = 15000


graph = reader.Graph(name, filepath)
graph.load_mat()
#graph.load_nx(6)

model = model.ModelMostStress(lr, maxIters)

csv_handler = csvHandler.Handler()
csv_handler.set_path_csv(graph)
csv_handler.init_csv()

for _ in range(5):
        now = datetime.now()
        print(f'\nIteration {_}')
        print(f'[Start Time: {now.strftime("%d/%m/%Y %H:%M:%S")}]')

        graph.set_pos()
        save_path_svg = f'output/figs/{graph.name}/{graph.name}_{_}'

        start = time.time()
        stress_min, unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20 = model.train(graph, save_path_svg)
        end = time.time()
        time_elapsed = end - start
        print(f'[Computation time: {round(time_elapsed, 2)}s]')

        csv_row = [graph.name, _, stress_min,
                   unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20, time_elapsed]

        csv_handler.write_csv(csv_row)