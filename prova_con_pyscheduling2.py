# https://github.com/timnon/pyschedule
import math
import os

import numpy as np
from pyschedule import Scenario, solvers, plotters, alt
import pandas as pd
import plotly as px
import plotly.figure_factory as ff
import datetime
import plotly.express as px
import pandas as pd
from collections import OrderedDict


# file_path = 'data_simple.csv'
# file_path = 'per diegolino.csv'
# df = pd.read_csv(file_path, sep=';', skipinitialspace=True)
# file_path = "C:\\Users\\buckler\Downloads\per diegolino - per diegolino.csv"
file_path = "C:\\Users\\buckler\Downloads\chiarella.csv"
df = pd.read_csv(file_path, sep=',', skipinitialspace=True)

# Pulizia e organizzazione dei dati
# Separazione delle presse in una lista
# df['Giorni produzione'] = df['Giorni produzione'].str.replace(",", ".").astype(float)
# df.to_csv("per diegolino.csv", sep=';', index=None)
nome_colonna_per_computo = f'Giorni produzione'
horizon = 365
base_date = datetime.datetime(2023, 1, 1)

df['Pressa'] = df['Pressa'].apply(lambda x: eval(x))
df = df.groupby('Stampo', as_index=False).agg({'Geometria': 'first',  'Pressa': 'first', 'Giorni produzione': 'sum', })
df[nome_colonna_per_computo] = (df['Giorni produzione']).apply(np.ceil).astype(int)
df['presses_length'] = df['Pressa'].apply(len)

# rocesso prima gli stampi ceh hanno poche presse da cui scegliere, e processo priuma gli stampi che occupano più tempo
df = df.sort_values(by=['presses_length', nome_colonna_per_computo, ], ascending=[True, False])
# Identificazione dell'elenco unico delle presse
unique_presses = set()
for presses in df['Pressa']:
    unique_presses.update(presses)

manual = True
if manual:
    # w_presss = OrderedDict()
    # for i, row in df.iterrows():
    #     for p in row['Pressa']:
    #         w_presss[p] = w_presss.get(p, 0) + row['Giorni produzione']
    # w_presss = OrderedDict(sorted(w_presss.items(), key=lambda x: x[1]))
    def get_lazy_press(g: OrderedDict, possible_presses: list):

        sorted_g = dict(sorted(g.items(), key=lambda x: x[1]['tot']))
        # sorted_g = OrderedDict(sorted(list(g.items()), key=lambda x: (x[1]['tot'], -x[1]['w'])))
        for p in sorted_g:
            if p in possible_presses:
                return p
        raise Exception("Infeasable")
        # return None

    gantt = {
        key: OrderedDict(
            tot=0, tasks=[]
            # w=w
        )
        # for key, w in w_presss.items()
        for key in unique_presses
    }
    for i, row in df.iterrows():
        press = get_lazy_press(gantt, row["Pressa"])
        gantt[press]["tasks"].append(
            OrderedDict(
                Task=row["Stampo"],  # stampi
                # Resource: None, # presse, sarebbe sempre uguale a key
                Start=gantt[press]["tot"],
                Finish=gantt[press]["tot"] + row[nome_colonna_per_computo]
            )
        )
        gantt[press]["tot"] += row[nome_colonna_per_computo]
    gantt_task = list()
    for press, tasks in gantt.items():
        for task in tasks['tasks']:
            gantt_task.append(OrderedDict(
                Task=task['Task'],
                Resource=press,
                Start=task['Start'],
                Finish=task['Finish'],
                NDay=task['Finish'] - task['Start']
            ))
    pass
else:
    # horizon = df[nome_colonna_per_computo].max()
    print(f"Horizon: {horizon}")



    # the planning horizon has "horizon" periods
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
    # stampi = [S.Task(r["Stampo"], length=math.ceil(r[nome_colonna_per_computo]), delay_cost=1) for i, r in df.iterrows()]
    stampi = [S.Task(r["Stampo"], length=r[nome_colonna_per_computo], delay_cost=1) for i, r in df.iterrows()]


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


    # df = pd.DataFrame([
    #     dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28', Resource="Alex"),
    #     dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15', Resource="Alex"),
    #     dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30', Resource="Max")
    # ])

    # Data base (puoi scegliere qualunque data)
    gantt_task = [
        dict(
            Task=str(t[0]),
            Resource=str(t[1]),
            Start=t[2],
            Finish=t[3]
        ) for t in S.solution()
    ]


gantt_task = sorted(gantt_task, key=lambda item: item['Resource'])
for task in gantt_task:
    task['Start'] = base_date + datetime.timedelta(days=task['Start'])
    task['Finish'] = base_date + datetime.timedelta(days=task['Finish'])
df = pd.DataFrame(gantt_task)

fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", hover_name="NDay", color="Task")
fig.update_xaxes(tickformat="%d-%m-%Y")

# fig.show()

# # Crea il diagramma di Gantt
# fig = ff.create_gantt(gantt_task, show_colorbar=True, group_tasks=True)
#
# # Aggiusta l'asse x per mostrare i giorni relativi all'anno
# fig.update_xaxes(type='linear', range=[0, 365], tickvals=list(range(0, 366, 30)), ticktext=[f"Giorno {i}" for i in range(0, 366, 30)])
#
# # Mostra il grafico
# fig.show()

import datetime
import numpy as np
import holidays
it_holidays = holidays.country_holidays('IT', subdiv='AP', years=2023)
analysis_types = [
    {
        "title": "robot",
        "holydays": [],
        "work_day_per_week": 7
    },
    {
        "title": "7 day + holydays",
        "holydays": it_holidays,
        "work_day_per_week": 7
    },
    {
        "title": "6 day + holydays",
        "holydays": it_holidays,
        "work_day_per_week": 6
    },
    {
        "title": "5 day + holydays",
        "holydays": it_holidays,
        "work_day_per_week": 5
    }

]

# Funzione per controllare se un giorno è un giorno lavorativo
def is_workday(day, h, work_day_week):
    return day.weekday() < work_day_week and day not in h

# Funzione per calcolare la data di fine
def calculate_end_date(start_date, duration, h, work_day_week):
    additional_days = 0
    current_date = start_date
    while duration > 0:
        if is_workday(current_date, h, work_day_week):
            duration -= 1
        else:
            additional_days += 1
        current_date += datetime.timedelta(days=1)

    return current_date, datetime.timedelta(days=additional_days)

# # Esempio di utilizzo
# start_date = datetime.date(2023, 1, 1) # Data di inizio
# duration = 10 # X giorni lavorativi
# holidays = [datetime.date(2023, 1, 6), datetime.date(2023, 4, 25)] # Esempio di giorni festivi
def modify_event(events, task_to_modify, resource, extra_duration):
    modification_time = None
    for event in events:
        if event['Task'] == task_to_modify and event['Resource'] == resource:
            # Modify the selected event
            event['Finish'] += extra_duration
            modification_time = event['Finish']
            break

    if modification_time:
        for event in events:
            if event['Resource'] == resource and event['Start'] >= modification_time:
                # Shift subsequent events of the same resource
                event['Start'] += extra_duration
                event['Finish'] += extra_duration

for analysis_type in analysis_types:


    gantt_task_mod = sorted(gantt_task, key=lambda x: x['Start'])
    for task in gantt_task_mod:
    #     task['Finish'] = calculte_end_date(task['Start'],  task['Finish'] - task['Start'], holidays)
        extra_duration, additional_days = calculate_end_date(task['Start'],  (task['Finish'] - task['Start']).days, analysis_type["holydays"], work_day_week=analysis_type["work_day_per_week"])
        resource_events_indexes = [i for i, gt in enumerate(gantt_task_mod) if gt["Resource"] == task["Resource"] and gt["Start"] >= task["Start"]]
        for i, idx in enumerate(resource_events_indexes):
            gantt_task_mod[idx]['Finish'] += additional_days
            if i > 0:
                gantt_task_mod[idx]['Start'] += additional_days
                # gantt_task[idx]['Finish'] += additional_days
        # modify_event(gantt_task, task_to_modify=task["Task"], resource=task["Resource"], extra_duration=additional_days)

    df = pd.DataFrame(gantt_task_mod)
    df.to_csv(os.path.join("results_custom", f"{analysis_type['title']}.csv"))

    fig = px.timeline(df, title=analysis_type["title"], x_start="Start", x_end="Finish", y="Resource", hover_name="NDay", color="Task")
    fig.update_xaxes(tickformat="%d-%m-%Y")

    fig.show()

# end_date = calculate_end_date(start_date, duration, holidays)
# print("Data di fine:", end_date)