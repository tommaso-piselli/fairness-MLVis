# **Graph Layout Optimization with Stress and Unfairness Objectives using Gradient Descent (GD)**

## Overview

This code aims to generate optimized graph layouts by minimizing both stress and unfairness objectives. It works by iteratively adjusting node positions to reduce stress while ensuring a fair distribution of nodes with different colors.

## Key Features

- **Utilizes a two-step optimization process:**
  - First, minimizes stress to achieve an initial layout called $\Gamma_0$.
  - Second, focuses on minimizing unfairness while maintaining stress below certain thresholds. This produces $\Gamma_1$ if the 5% threshold is surpassed, or $\Gamma_2$ if the 20% threshold is surpassed. 
- **Incorporates node coloring for visual clarity:** Assigns colors to nodes based on a specified percentage, aiding in visual differentiation. There are different implementations: `ModelRandom` colors an input percentage of randomly chosen nodes, `ModelMostStress` colors an input percentage of the most stress nodes (computed when the overall stress of the graph is optimized).
- **Saves visualizations at different stages:** Generates SVG images of the graph layout at the end.
- **Records performance metrics:** Logs stress and unfairness values at different stages, as well as execution time, into a CSV file for analysis.

## Usage

Check `requirements.txt` for libraries. Run `computeLayout.py` to generate layouts. You can select a different input by changing the `name` variable with the name of the corresponding graph name.
You can find a list of possible graphs in the `data` folder.


If you want to test the ModelRandom, you can modify the `computeLayout.py` with these key changes:
``` python
    percentages = [10, 20, 30, 40, 50]
    
    for _ in range(10):
       
        graph.set_pos()
        
        for percentage in percentages:
            
            save_path_svg = f'output/figs/{graph.name}/{graph.name}_{_}_{percentage}'

            start = time.time()
            
            stress_min, unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20 = model.train(
                graph, percentage, save_path=save_path_svg)
            
            end = time.time()
            time_elapsed = end - start
            
            csv_row = [graph_name, _, percentage, stress_start, unfair_start, stress_min,
                   unfair_stress_min, stress_5, unfair_5, stress_20, unfair_20, time_elapsed]
            
            [... same as before ...]

```

If you want to test it with smaller graphs from the NetworkX library, you can use the `load_nx` function. Inside the function, there is a dictionary
to select the graphs. Here are the keys:
```
key -   Graph.Name , NetworX_generator
------------------------------------------------
0   -   ('karate', nx.karate_club_graph())
1   -   ('path-5', nx.path_graph(5))
2   -   ('path-10', nx.path_graph(10))
3   -   ('cycle-10', nx.cycle_graph(10))
4   -   ('grid-5-5', nx.grid_graph(dim=[5, 5]))
5   -   ('grid-10-6', nx.grid_graph(dim=[10, 6]))
6   -   ('tree-2-3', nx.balanced_tree(2, 3))
7   -   ('tree-2-4', nx.balanced_tree(2, 4))
8   -   ('tree-2-5', nx.balanced_tree(2, 5))
9   -   ('tree-2-6', nx.balanced_tree(2, 6))
10  -   ('k-5', nx.complete_graph(5))
11  -   ('k-20', nx.complete_graph(20))
12  -   ('bipartite-graph-3-3', nx.complete_bipartite_graph(3, 3))
13  -   ('bipartite-graph-5-5', nx.complete_bipartite_graph(5, 5))
14  -   ('dodecahedron', nx.dodecahedral_graph())
15  -   ('cube', nx.hypercube_graph(3))
```

## Output

- Generates SVG visualizations in the `output/figs/` directory.
- Creates a CSV file with performance metrics in the `output/csv/` directory.


## Main Execution

- Loads the graph.
- Iterates through different position initializations for the nodes.
- Runs the optimization process for each combination.
- Records results in the CSV file.

## Additional Notes

- The code utilizes a GPU if available (CUDA).
