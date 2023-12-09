# https://github.com/timnon/pyschedule
import math
import numpy as np
from pyschedule import Scenario, solvers, plotters, alt
import pandas as pd
import plotly as px
import plotly.figure_factory as ff
import datetime



# file_path = 'data_simple.csv'
file_path = 'data_hard.csv'
df = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)
# Pulizia e organizzazione dei dati
# Separazione delle presse in una lista

groub_by_hour = 6
nome_colonna_per_computo = f'Giorni produzione {groub_by_hour}'
div = 24 / groub_by_hour
horizon = int(365 / div)

df['Pressa'] = df['Pressa'].apply(lambda x: eval(x))
df = df.groupby('Stampo', as_index=False).agg({'Giorni produzione': 'sum', 'Pressa': 'first'})
df[nome_colonna_per_computo] = (df['Giorni produzione']/6).apply(np.ceil)
# Identificazione dell'elenco unico delle presse
unique_presses = set()
for presses in df['Pressa']:
    unique_presses.update(presses)
precision = 10
# the planning horizon has 10 periods
S = Scenario('Scehduling', horizon=horizon)

# two resources: Alice and Bob
# presse = [S.Resource(str(r)) for r in unique_presses]
presse = {str(r): S.Resource(str(r)) for r in unique_presses}
# Alice, Bob = S.Resource('Alice'), S.Resource('Bob')

# # three tasks: cook, wash, and clean
# cook = S.Task('cook',length=1,delay_cost=1)
# wash = S.Task('wash',length=2,delay_cost=1)
# clean = S.Task('clean',length=3,delay_cost=2)
# stampi = [S.Task(r["Stampo"], length=math.ceil(r['Giorni produzione']*100)) for i, r in df.iterrows()]
stampi = [S.Task(r["Stampo"], length=math.ceil(r[nome_colonna_per_computo]), delay_cost=1) for i, r in df.iterrows()]


# every task can be done either by Alice or Bobù
# i vari stampi possono essere fatti solo con determinate presse
for t in stampi:
    who = None
    for s in df[df['Stampo'] == str(t)]['Pressa']:
        for p in s:
            if who is None:
                who = presse[str(p)]
            else:
                who |= presse[str(p)]
    t += who
# cook += Alice | Bob
# wash += Alice | Bob
# clean += Alice | Bob

# compute and print a schedule
solvers.mip.solve(S, msg=1)
print(S.solution())

plotters.matplotlib.plot(S, img_filename='gantt.png')
import plotly.express as px
import pandas as pd

# df = pd.DataFrame([
#     dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),
#     dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Resource="Alex"),
#     dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Resource="Max")
# ])

# Data base (puoi scegliere qualunque data)
base_date = datetime.datetime(2023, 1, 1)
gantt_task = [dict(Task=str(t[0]),  Resource=str(t[1]), Start=t[2]*groub_by_hour, Finish=t[3]*groub_by_hour) for t in S.solution()]
for task in gantt_task:
    task['Start'] = base_date + datetime.timedelta(days=task['Start'])
    task['Finish'] = base_date + datetime.timedelta(days=task['Finish'])
df = pd.DataFrame(gantt_task)

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", color="Resource")
fig.show()

# # Crea il diagramma di Gantt
# fig = ff.create_gantt(gantt_task, show_colorbar=True, group_tasks=True)
#
# # Aggiusta l'asse x per mostrare i giorni relativi all'anno
# fig.update_xaxes(type='linear', range=[0, 365], tickvals=list(range(0, 366, 30)), ticktext=[f"Giorno {i}" for i in range(0, 366, 30)])
#
# # Mostra il grafico
# fig.show()