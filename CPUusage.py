from cProfile import label

import psutil
import torch
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import webbrowser

from networkx.algorithms.bipartite import color
from pandas.io.common import file_path_to_url


def python_sort(data):
    return sorted(data)

def torch_sort(data):
    tensor_data = torch.tensor(data)
    return torch.sort(tensor_data).values

def cpu_utilization(sort_func, data):
    cpu_usage_before = psutil.cpu_percent(interval=None)
    start_time = time.time()
    sort_func(data)
    end_time= time.time()
    cpu_usage_after= psutil.cpu_percent(interval=None)
    cpu_usage_during = abs((cpu_usage_after-cpu_usage_before)/2)
    elapsed_time = end_time-start_time
    return cpu_usage_during, elapsed_time

list_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

results = []
pcpu = []
tcpu = []

for size in list_sizes:
    test = [random.randint(0, size) for i in range(size)]
    python_cpu, python_time = cpu_utilization(python_sort, test)
    torch_cpu, torch_time = cpu_utilization(torch_sort, test)
    results.append({
        'Size':size,
        'Python CPU usage':python_cpu,
        'Python Time': python_time,
        'PyTorch CPU usage': torch_cpu,
        'PyTorch Time': torch_time,
    })
    pcpu.append(python_cpu)
    tcpu.append(torch_cpu)
pd.options.display.float_format = '{:.2f}'.format
df = pd.DataFrame(results)
styled_df = (df.style
             .set_properties(**{'text-align': 'left'})
             .set_table_styles([{
                'selector':'th',
                'props' : [('font-size', '110%'), ('text-align', 'center')]
              }])
            .highlight_max(subset=["Python CPU usage"], color='lightgreen')
            .highlight_min(subset=["Python CPU usage"], color='lightcoral')
            .highlight_max(subset=["PyTorch CPU usage"], color='lightgreen')
            .highlight_min(subset=["Python CPU usage"], color='lightcoral'))

print(df)

with open("CPUusage.html",'w') as f:
    df.to_html(f)



plt.plot(list_sizes, pcpu, marker='o', label='Python CPU Usage')
plt.plot(list_sizes, tcpu, marker='o', label='PyTorch CPU Usage')

plt.xlabel("List sizes")
plt.ylabel("CPU Usage (%)")
plt.title("CPU usage by List Sizes")
plt.ylim(0, max(max(pcpu), max(tcpu)) + 5)
plt.legend()
plt.show()