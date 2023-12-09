import numpy as np
import pulp
import pandas as pd
from scipy.optimize import minimize

stampi = ["A54", "A50", "A49", "A82"]
stampi_presse_associazione = {
    "A54": [4, 5, 7, 11, 18],
    "A50": [17],
    "A49": [17],
    "A82": [13, 17]
}

giorni_produzione = {
    "A54": 46.86+6.8,
    "A50": 0.6,
    "A49": 11.70,
    "A82": 36.48
}

scheduling = {stampo: [] for stampo in stampi}
for pressa, giorni_di_produzione in giorni_produzione.items():
    # determino lo stamo per la pressa in esame che sia meno possibile utilizzarlo in altre presse e scelgo quello
    for p in giorni_di_produzione.items():
        if p != pressa:

# prova https://github.com/timnon/pyschedule
# https://github.com/tpaviot/ProcessScheduler
# https://www.pyomo.org/