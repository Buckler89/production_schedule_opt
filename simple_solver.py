import pulp as lp

x = lp.LpVariable("x", 0, 3)
y = lp.LpVariable("y", 0, 1)
prob = lp.LpProblem("myProblem", lp.LpMinimize)
prob += x + y <= 2
prob += -4*x + y

status = prob.solve()
# status = prob.solve(GLPK(msg = 0))
print()
# Stampa i risultati
for var in prob.variables():
    print(f"{var.name}: {var.value()}")