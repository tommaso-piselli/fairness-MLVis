import csv

class Handler(object):
    def __init__(self):
        self.save_path_csv = ''
        self.csv_header = ["graph_name", "configuration", "stress_min", "unfair_stress_min", "stress_5", "unfair_5",
                           "stress_20", "unfair_20", "time"]


    def set_path_csv(self, graph):
        self.save_path_csv = f'output/csv/{graph.name}.csv'

    def init_csv(self):
        with open(self.save_path_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.csv_header)
            csvfile.close()

    def write_csv(self, csv_row):
        with open(self.save_path_csv, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(csv_row)
                csvfile.close()
