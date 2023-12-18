import copy

import pandas as pd
import pulp as pl
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum
from itertools import chain
from itertools import product
from pandas.tseries.holiday import USFederalHolidayCalendar as Calendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
from pulp import PULP_CBC_CMD
import os
import plotly.express as px
import itertools
import datetime
import holidays
from datetime import datetime, timedelta
import json

file_path = "data/dati-presse-tampi_V1.csv"
data = pd.read_csv(file_path, sep=',', skipinitialspace=True)


data = data.groupby('Stampo', as_index=False).agg({'Pressa': 'first', 'Giorni produzione': 'sum', })

# Anno di riferimento per il calcolo (assumendo l'anno corrente)
year = 2024
base_date = datetime(year, 1, 1)

# Creazione di un calendario personalizzato che esclude i giorni festivi standard
calendar = Calendar()
holidays_var = calendar.holidays(start=f'{year}-01-01', end=f'{year}-12-31')
custom_business_day_5d = CustomBusinessDay(holidays=holidays_var)
custom_business_day_6d = CustomBusinessDay(holidays=holidays_var, weekmask='Mon Tue Wed Thu Fri Sat')
custom_business_day_7d = CustomBusinessDay(holidays=holidays_var, weekmask='Mon Tue Wed Thu Fri Sat Sun')

# Calcolo dei giorni lavorativi per ogni configurazione (5, 6, 7 giorni alla settimana)
business_days_5d = len(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq=custom_business_day_5d))
business_days_6d = len(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq=custom_business_day_6d))
business_days_7d = len(pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq=custom_business_day_7d))
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
def calculate_end_date(duration, start_date, h, work_day_week):
    additional_days = 0
    current_date = start_date
    duration_counter = copy.deepcopy(duration)
    while duration_counter > 0:
        if is_workday(current_date, h, work_day_week):
            duration_counter -= 1
        else:
            additional_days += 1
        current_date += timedelta(days=1)
    total_days = duration + additional_days
    return current_date, timedelta(days=additional_days), total_days

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

def calculate_end_date_apply(*args):
    return calculate_end_date(*args)[2]

data['Giorni produzione 5'] = data['Giorni produzione'].apply(calculate_end_date_apply, args=(base_date, analysis_types[3]["holydays"], analysis_types[3]["work_day_per_week"]))
data['Giorni produzione 6'] = data['Giorni produzione'].apply(calculate_end_date_apply, args=(base_date, analysis_types[2]["holydays"], analysis_types[2]["work_day_per_week"]))
data['Giorni produzione 7'] = data['Giorni produzione'].apply(calculate_end_date_apply, args=(base_date, analysis_types[1]["holydays"], analysis_types[1]["work_day_per_week"]))
# Preparazione dei dati
stampi = data['Stampo'].unique()
unique_presses = set(sum([eval(p) for p in data['Pressa']], []))  # Unione di tutte le liste di presse
# giorni_per_stampo = dict(zip(data['Stampo'], data['Giorni produzione']))
giorni_per_stampo = {
    7: dict(zip(data['Stampo'], data['Giorni produzione 7'])),
    6: dict(zip(data['Stampo'], data['Giorni produzione 6'])),
    5: dict(zip(data['Stampo'], data['Giorni produzione 5'])),
}
# giorni_per_stampo_7 = dict(zip(data['Stampo'], data['Giorni produzione 7']))
# giorni_per_stampo_6 = dict(zip(data['Stampo'], data['Giorni produzione 6']))
# giorni_per_stampo_5 = dict(zip(data['Stampo'], data['Giorni produzione 5']))

# Creazione del modello di programmazione lineare
model = LpProblem("Scheduling_Produzione", LpMinimize)

# Variabili decisionali
# x[(i, j)] = 1 se lo stampo i è assegnato alla pressa j
x = LpVariable.dicts("Assegnazione", (stampi, unique_presses), 0, 1, LpInteger)

# y[j][d] = 1 if press j works d days per week (where d can be 5, 6, or 7)
y = LpVariable.dicts("Working_days", (unique_presses, giorni_per_stampo.keys()), 0, 1, LpInteger)
# Modify the objective function to prioritize 5-day schedules
penalty_6d = 1  # A large number to penalize 6-day schedules
penalty_7d = 2  # A larger number to penalize 7-day schedules

# total_time_per_press[i] is the production time for press i
total_time_per_press = LpVariable.dicts("total_time", unique_presses, lowBound=0)
# z is the maximum production time across all presses
z = LpVariable("max_production_time", lowBound=0)
# Introduciamo variabili ausiliarie per gestire le differenze assolute
# differenze = []
# for pressa1, pressa2 in itertools.combinations(unique_presses, 2):
#     diff = LpVariable(f"diff_p{pressa1}_p{pressa2}", lowBound=0)
#     differenze.append(diff)
#     model += diff >= total_time_per_press[pressa1] - total_time_per_press[pressa2]
#     model += diff >= total_time_per_press[pressa2] - total_time_per_press[pressa1]
# Objective function: Minimize the maximum production time
# model += z + sum(differenze)
model += (z +
        # sum(differenze) +
          penalty_6d * lpSum(y[j][6] for j in unique_presses) + penalty_7d * lpSum(y[j][7] for j in unique_presses))
# model += z <= 500
# Add constraints to ensure only one schedule is chosen for each press
for j in unique_presses:
    model += lpSum(y[j][d] for d in giorni_per_stampo.keys()) == 1

# Introduzione della nuova variabile decisionale
xy = LpVariable.dicts("Prod_var", (stampi, unique_presses, giorni_per_stampo.keys()), 0, 1, LpInteger)

# Aggiunta dei vincoli per linearizzare il prodotto delle variabili
for i in stampi:
    for j in unique_presses:
        for w in giorni_per_stampo.keys():
            model += xy[i][j][w] <= x[i][j]
            model += xy[i][j][w] <= y[j][w]
            model += xy[i][j][w] >= x[i][j] + y[j][w] - 1
# The production time for each press is the sum of times for the jobs assigned to it
for j in unique_presses:
    constraints_for_press_j = []
    for w in giorni_per_stampo.keys():
        constraints_for_press_j += [giorni_per_stampo[w][i] *
                                    xy[i][j][w]
                                    # x[i][j]
                                    for i in stampi]

    model += total_time_per_press[j] == lpSum(constraints_for_press_j)

# The production time for each press should be less than or equal to the maximum production time
for j in unique_presses:
    model += total_time_per_press[j] <= z
    # Ensure production time does not exceed available working days
    # model += total_time_per_press[j] <= business_days_5d * y[j][5]
    # model += total_time_per_press[j] <= business_days_6d * y[j][6]
    # model += total_time_per_press[j] <= business_days_7d * y[j][7]
# Vincoli
# 1. Ogni stampo deve essere assegnato a una pressa
for i in stampi:
    constraints = [x[i][j] for j in unique_presses]
    # aggiunge un vincolo al modello. Questo vincolo richiede che la somma delle variabili decisionali per ciascuno
    # stampo i (considerando tutte le presse j) sia esattamente uguale a 1. In termini pratici, ciò significa che ogni
    # stampo i deve essere assegnato esattamente a una pressa. Non più di una (perché la somma non può superare 1) e non
    # meno di una (perché la somma non può essere inferiore a 1).
    model += lpSum(constraints) == 1

# Mappatura dello stampo alle sue presse compatibili
stampo_to_presse = {row['Stampo']: eval(row['Pressa']) for index, row in data.iterrows()}

# Riscrittura dei vincoli di compatibilità
for i in stampi:
    presse_compatibili = stampo_to_presse[i]
    for j in unique_presses:
        if j not in presse_compatibili:
            model += x[i][j] == 0

# Impostazione del time limit e del gap di ottimalità
solver_list = pl.listSolvers(onlyAvailable=True)
# from pulp import CPLEX_CMD#, GUROBI_CMD, GLPK_CMD
solver = PULP_CBC_CMD(timeLimit=60, gapRel=0.001)  # 600 secondi e 1% di gap
# solver = CPLEX_CMD(timeLimit=60, gapRel=0.001)  # 600 secondi e 1% di gap

if not os.path.exists(os.path.join("results_pupl", f"results.csv")):
    # Risoluzione del modello corretto
    model.solve(solver)
    # print(model)





    # Estrazione dei risultati corretti
    assegnazioni = [(i, j) for i in stampi for j in unique_presses if x[i][j].varValue > 0]
    # Estrazione delle informazioni sulle giornate lavorative per ogni pressa
    # giorni_lavorativi_per_pressa = {j: d for j in unique_presses for d in [5, 6, 7] if y[j][d].varValue > 0}
    #  Estrazione delle informazioni sui giorni lavorativi per ogni pressa in base al valore massimo
    giorni_lavorativi_per_pressa = {}
    for j in unique_presses:
        max_val = 0
        giorni_opt = 0
        print(f"Pressa {j}: {y[j][5].varValue}, {y[j][6].varValue}, {y[j][7].varValue}")
        for d in [5, 6, 7]:
            if y[j][d].varValue > max_val:
                max_val = y[j][d].varValue
                giorni_opt = d
        giorni_lavorativi_per_pressa[j] = giorni_opt

    # Stampa dei risultati
    for pressa, giorni in giorni_lavorativi_per_pressa.items():
        print(f"La pressa {pressa} lavora {giorni} giorni a settimana")



    # Conversione delle assegnazioni in un formato utilizzabile
    assegnazioni_df = pd.DataFrame(assegnazioni, columns=['Stampo', 'Pressa'])
    # determinazione della durata di ciascuno stampo in base all'assegnazione dei giorni lavoriativi per la pressa e lo specifico stampo
    assegnazioni_df['Durata'] = assegnazioni_df.apply(lambda x: giorni_per_stampo[giorni_lavorativi_per_pressa[x['Pressa']]][x['Stampo']], axis=1)
    assegnazioni_df = assegnazioni_df.sort_values(by=['Pressa', 'Stampo'])
    assegnazioni_df["work_day_per_week"] = assegnazioni_df.apply(lambda x: giorni_lavorativi_per_pressa[x['Pressa']], axis=1)
    assegnazioni_df.to_csv(os.path.join("results_pupl", f"results.csv"))
    #save giorni_lavorativi_per_pressa
    with open(os.path.join("results_pupl", f"giorni_lavorativi_per_pressa.json"), "w") as f:
        json.dump(giorni_lavorativi_per_pressa, f)
else:
    assegnazioni_df = pd.read_csv(os.path.join("results_pupl", f"results.csv"))
    assegnazioni_df["Pressa"] = assegnazioni_df["Pressa"].astype(str)
    # load giorni_lavorativi_per_pressa
    with open(os.path.join("results_pupl", f"giorni_lavorativi_per_pressa.json"), "r") as f:
        giorni_lavorativi_per_pressa = json.load(f)

# Calcolo delle date di inizio e fine per ciascun stampo
date_inizio = {}
date_fine = {}

for index, row in assegnazioni_df.iterrows():
    pressa = row['Pressa']
    durata = row['Durata']
    if pressa not in date_inizio:
        # Se è il primo stampo per la pressa, inizia oggi
        inizio = base_date
    else:
        # Altrimenti inizia dopo la fine del precedente stampo sulla stessa pressa
        inizio = date_fine[pressa]
    fine = inizio + timedelta(days=durata)
    date_inizio[pressa] = inizio
    date_fine[pressa] = fine
    assegnazioni_df.at[index, 'Inizio'] = inizio
    assegnazioni_df.at[index, 'Fine'] = fine

# Preparazione dei dati per Plotly, assicurandosi che tutti i valori siano coerenti come stringhe
gantt_task = []
for index, row in assegnazioni_df.iterrows():
    giorni = giorni_lavorativi_per_pressa[row['Pressa']]
    resource_name = "Pressa " + str(row['Pressa'])
    # Formatta il nome della risorsa in base ai giorni di lavoro
    if giorni == '7':
        resource_name = f"<b>{resource_name}</b>"
    elif giorni == '6':
        # Usare un tag <span> per un grassetto più leggero (non sempre supportato)
        resource_name = f"<span style='font-weight: 600;'>{resource_name}</span>"

    gantt_task.append(dict
        (
            Task=str(row['Stampo']),
            Start=row['Inizio'],#.strftime("%Y-%m-%d"),
            Finish=row['Fine'],#.strftime("%Y-%m-%d"),
            Resource=resource_name,
            Mode=f"Giorni lavoro {giorni}",
            NDay=(row['Fine']-row['Inizio']).days,
        )
    )
# order discendente element by Mode
gantt_task = sorted(gantt_task, key=lambda x: x['Mode'])
fig = px.timeline(gantt_task, title="Schedulazione Presse", x_start="Start", x_end="Finish", y="Resource",
                  # hover_name="Mode",
                  # color="Task",
                  hover_name="Task",
                  color="Mode",
                  )
fig.update_xaxes(tickformat="%d-%m-%Y")

fig.show()


do_plot = False
if do_plot:
    for analysis_type in analysis_types:


        gantt_task_mod = sorted(gantt_task, key=lambda x: x['Start'])
        for task in gantt_task_mod:
        #     task['Finish'] = calculte_end_date(task['Start'],  task['Finish'] - task['Start'], holidays)
            extra_duration, additional_days, _ = calculate_end_date((task['Finish'] - task['Start']).days, task['Start'], analysis_type["holydays"], work_day_week=analysis_type["work_day_per_week"])
            resource_events_indexes = [i for i, gt in enumerate(gantt_task_mod) if gt["Resource"] == task["Resource"] and gt["Start"] >= task["Start"]]
            for i, idx in enumerate(resource_events_indexes):
                gantt_task_mod[idx]['Finish'] += additional_days
                if i > 0:
                    gantt_task_mod[idx]['Start'] += additional_days
                    # gantt_task[idx]['Finish'] += additional_days
            # modify_event(gantt_task, task_to_modify=task["Task"], resource=task["Resource"], extra_duration=additional_days)

        df = pd.DataFrame(gantt_task_mod)
        df.to_csv(os.path.join("results_pupl", f"{analysis_type['title']}.csv"))

        fig = px.timeline(df, title=analysis_type["title"], x_start="Start", x_end="Finish", y="Resource", hover_name="NDay", color="Task")
        fig.update_xaxes(tickformat="%d-%m-%Y")

        fig.show()
