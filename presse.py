import pulp
import pandas as pd

file_path = 'data_simple.csv'
data = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)
# Pulizia e organizzazione dei dati
# Separazione delle presse in una lista
# data['Pressa'] = data['Pressa'].apply(lambda x: x.split())
data['Pressa'] = data['Pressa'].apply(lambda x: eval(x))

# Convertendo "Giorni produzione" in numeri float per sicurezza
data['Giorni produzione'] = data['Giorni produzione'].astype(float)


# Creazione di un modello di programmazione lineare
model = pulp.LpProblem("Minimizzazione_Cambi_Stampo", pulp.LpMinimize)

# Identificazione dell'elenco unico delle presse
unique_presses = set()
for presses in data['Pressa']:
    unique_presses.update(presses)

# Variabili di decisione: una per ogni possibile combinazione stampo-pressa
assignments = pulp.LpVariable.dicts("assignment",
                                    ((stampo, pressa) for stampo in data.Stampo.unique() for pressa in unique_presses),
                                    cat='Binary')

# Aggiunta della funzione obiettivo: Minimizzare il numero totale di cambi di stampo
# Si assume che un cambio avvenga ogni volta che uno stampo viene assegnato a una pressa
model += pulp.lpSum(assignments[stampo, pressa] for stampo in data.Stampo.unique() for pressa in unique_presses)

# Vincolo: la produzione totale per ogni stampo deve soddisfare la richiesta
for stampo in data.Stampo.unique():
    stampo_data = data[data.Stampo == stampo]
    total_days_required = stampo_data['Giorni produzione'].sum()
    model += pulp.lpSum(assignments[stampo, pressa] for pressa in unique_presses) >= total_days_required

# Vincolo: ogni stampo può essere assegnato solo alle presse elencate per quel stampo
for stampo, row in data.iterrows():
    for pressa in unique_presses:
        if str(pressa) not in row['Pressa']:
            model += assignments[row['Stampo'], pressa] == 0

model.writeLP("presse.lp")

# Risoluzione del modello
model.solve()

# Stato della soluzione
sol_status = pulp.LpStatus[model.status]
print(sol_status)
# Controllo se il modello ha trovato una soluzione
if sol_status == 'Optimal':
    solution_found = True
    # Estrazione dei risultati
    results = []
    for stampo in data.Stampo.unique():
        for pressa in unique_presses:
            if pulp.value(assignments[stampo, pressa]) == 1:
                results.append((stampo, pressa))
else:
    solution_found = False

solution_found, sol_status, results[:10]  # Visualizziamo solo i primi 10 risultati per brevità
