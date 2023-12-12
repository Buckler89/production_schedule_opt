import pandas as pd
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


# file_path = "C:\\Users\\buckler\Downloads\chiarella.csv"
# df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
file_path = "data/chiarella - Sheet1.csv"
data = pd.read_csv(file_path, sep=',', skipinitialspace=True)

# df['Pressa'] = df['Pressa'].apply(lambda x: eval(x))
# df = df.groupby('Stampo', as_index=False).agg({'Geometria': 'first',  'Pressa': 'first', 'Giorni produzione': 'sum', })
# # df["Giorni produzione ceil"] = (df['Giorni produzione']).apply(np.ceil).astype(int)
# df['presses_length'] = df['Pressa'].apply(len)

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




# Preparazione dei dati
stampi = data['Stampo'].unique()
unique_presses = set(sum([eval(p) for p in data['Pressa']], []))  # Unione di tutte le liste di presse
giorni_per_stampo = dict(zip(data['Stampo'], data['Giorni produzione']))

# Creazione del modello di programmazione lineare
model = LpProblem("Scheduling_Produzione", LpMinimize)

# Variabili decisionali
# x[(i, j)] = 1 se lo stampo i è assegnato alla pressa j
x = LpVariable.dicts("Assegnazione", (stampi, unique_presses), 0, 1, LpInteger)


# con tempo massimo ---------------------------------------------------
# tempo_massimo = LpVariable("TempoMassimo", lowBound=0, ) #cat=LpInteger)
# model += tempo_massimo
# # Vincoli per assicurare che il tempo di produzione per ogni pressa non superi il tempo massimo
# for j in unique_presses:
#     model += lpSum([giorni_per_stampo[i] * x[i][j] for i in stampi]) <= tempo_massimo
# ---------------------------------------------------------------------------
# # con tempo medio --------------------------------------------------- non funziona
# # Variabile per il tempo totale di produzione su ogni pressa
# total_time_per_press = LpVariable.dicts("total_time", unique_presses, lowBound=0)
# # Funzione obiettivo: minimizzare la media dei tempi di produzione per ogni pressa
# model += lpSum(total_time_per_press)
#
# # Vincoli per assicurare che il tempo di produzione per ogni pressa non superi il tempo massimo
# for j in unique_presses:
#     constraints_for_press_j = [giorni_per_stampo[i] * x[i][j] for i in stampi]
#     model += lpSum(constraints_for_press_j) <= total_time_per_press[j]
# ---------------------------------------------------------------------------
# total_time_per_press[i] is the production time for press i
total_time_per_press = LpVariable.dicts("total_time", unique_presses, lowBound=0)
# z is the maximum production time across all presses
z = LpVariable("max_production_time", lowBound=0)
# Introduciamo variabili ausiliarie per gestire le differenze assolute
differenze = []
for pressa1, pressa2 in itertools.combinations(unique_presses, 2):
    diff = LpVariable(f"diff_p{pressa1}_p{pressa2}", lowBound=0)
    differenze.append(diff)
    model += diff >= total_time_per_press[pressa1] - total_time_per_press[pressa2]
    model += diff >= total_time_per_press[pressa2] - total_time_per_press[pressa1]
# Objective function: Minimize the maximum production time
model += z + sum(differenze)
# z is the maximum production time across all presses
# z = LpVariable.dicts("max_production_time", unique_presses, lowBound=0)
# Objective function: Minimize the maximum production time
# model += lpSum(z)
# The production time for each press is the sum of times for the jobs assigned to it
for j in unique_presses:
    constraints_for_press_j = [giorni_per_stampo[i] * x[i][j] for i in stampi]
    model += total_time_per_press[j] == lpSum(constraints_for_press_j)
    # model += total_time_per_press[i] == lpSum(x[(i, j)] * giorni_per_stampo[j] for j in range(len(giorni_per_stampo)))

# The production time for each press should be less than or equal to the maximum production time
for j in unique_presses:
    # model += total_time_per_press[j] <= total_time_per_press[j]
    model += total_time_per_press[j] <= z
    # model += total_time_per_press[j] <= z[j]
# ---------------------------------------------------------------------------


# # Funzione obiettivo: Minimizzare il tempo totale di produzione
# eqs = [giorni_per_stampo[i] * x[i][j] for i, j in product(stampi, unique_presses)]
# model += lpSum(eqs)

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
solver = PULP_CBC_CMD(timeLimit=60, gapRel=0.01)  # 600 secondi e 1% di gap

# Risoluzione del modello corretto
model.solve(solver)
print(model)
# Estrazione dei risultati corretti
assegnazioni = [(i, j) for i in stampi for j in unique_presses if x[i][j].varValue > 0]


# Preparazione dei dati per Plotly, assicurandosi che tutti i valori siano coerenti come stringhe
gantt_task = []

# Conversione delle assegnazioni in un formato utilizzabile
assegnazioni_df = pd.DataFrame(assegnazioni, columns=['Stampo', 'Pressa'])
assegnazioni_df['Durata'] = assegnazioni_df['Stampo'].map(giorni_per_stampo)
assegnazioni_df = assegnazioni_df.sort_values(by=['Pressa', 'Stampo'])

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


for index, row in assegnazioni_df.iterrows():
    gantt_task.append(dict(Task=str(row['Stampo']),
                          Start=row['Inizio'],#.strftime("%Y-%m-%d"),
                          Finish=row['Fine'],#.strftime("%Y-%m-%d"),
                          Resource="Pressa " + str(row['Pressa']),
                          NDay=(row['Fine']-row['Inizio']).days
                       )
                      )

fig = px.timeline(gantt_task, title="robot", x_start="Start", x_end="Finish", y="Resource", hover_name="NDay", color="Task")
fig.update_xaxes(tickformat="%d-%m-%Y")

fig.show()




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
        current_date += timedelta(days=1)

    return current_date, timedelta(days=additional_days)

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
    df.to_csv(os.path.join("results_pupl", f"{analysis_type['title']}.csv"))

    fig = px.timeline(df, title=analysis_type["title"], x_start="Start", x_end="Finish", y="Resource", hover_name="NDay", color="Task")
    fig.update_xaxes(tickformat="%d-%m-%Y")

    fig.show()

