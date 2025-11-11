# simplex_report_fixed.py
# Исправленная версия двухфазного симплекс-решателя с подробным отчетом

import textwrap
EPS = 1e-9

def print_tableau_simple(tab, var_names, basis, title=None, report_lines=None):
    m = len(tab) - 1
    n = len(tab[0]) - 1
    header = ["Basis", "b"] + var_names
    widths = [max(6, len(h)) for h in header]
    sep = "-" * (sum(widths) + 3 * len(widths))
    lines = []
    if title:
        lines.append(title)
    lines.append(sep)
    lines.append(" | ".join(h.center(w) for h,w in zip(header, widths)))
    lines.append(sep)
    for i in range(m):
        base = ("x"+str(basis[i]+1)) if basis[i] >= 0 else "?"
        row = [base, f"{tab[i][-1]:.6g}"] + [f"{tab[i][j]:.6g}" for j in range(n)]
        lines.append(" | ".join(str(s).rjust(w) for s,w in zip(row, widths)))
    lines.append(sep)
    obj = ["Z", f"{tab[-1][-1]:.6g}"] + [f"{tab[-1][j]:.6g}" for j in range(n)]
    lines.append(" | ".join(str(s).rjust(w) for s,w in zip(obj, widths)))
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    if report_lines is not None:
        report_lines.append(text + "\n")

def make_tableau(A, b, obj_row):
    m = len(A); n = len(A[0])
    T = [[0.0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            T[i][j] = A[i][j]
        T[i][-1] = b[i]
    for j in range(n):
        T[-1][j] = obj_row[j]
    T[-1][-1] = obj_row[-1] if len(obj_row) > n else 0.0
    return T

def simplex_core(tableau, basis, var_names, maximize=True, report=None):
    m = len(tableau)-1
    n = len(tableau[0]) - 1
    iteration = 0
    if report is None:
        report = []
    report.append("Start simplex iteration\n")
    while True:
        iteration += 1
        obj = tableau[-1]
        entering = -1
        if maximize:
            best = 0.0
            for j in range(n):
                if obj[j] > best + EPS:
                    best = obj[j]; entering = j
        else:
            best = 0.0
            for j in range(n):
                if obj[j] < best - EPS:
                    best = obj[j]; entering = j
        if entering == -1:
            report.append(f"Optimality reached (iteration {iteration-1}).\n")
            break
        # ratio test
        leaving = -1; min_ratio = float("inf")
        for i in range(m):
            a = tableau[i][entering]
            if a > EPS:
                ratio = tableau[i][-1] / a
                if ratio < min_ratio - 1e-12:
                    min_ratio = ratio; leaving = i
        if leaving == -1:
            report.append("Unbounded detected.\n")
            return {"status":"unbounded", "report":report}
        report.append(f"Iter {iteration}: entering col {entering}, leaving row {leaving}\n")
        report.append("Before pivot:\n")
        print_tableau_simple(tableau, var_names, basis, title=f"Iteration {iteration} - before pivot", report_lines=report)
        # pivot
        pivot = tableau[leaving][entering]
        tableau[leaving] = [v / pivot for v in tableau[leaving]]
        for i in range(m+1):
            if i == leaving: continue
            factor = tableau[i][entering]
            if abs(factor) > EPS:
                tableau[i] = [tableau[i][j] - factor * tableau[leaving][j] for j in range(n+1)]
        basis[leaving] = entering
        report.append("After pivot:\n")
        print_tableau_simple(tableau, var_names, basis, title=f"Iteration {iteration} - after pivot", report_lines=report)
        if iteration > 1000:
            raise RuntimeError("Too many iterations in simplex")
    return {"status":"optimal", "tableau":tableau, "basis":basis, "report":report}

def build_extended(A, senses):
    # создаём расширенную матрицу A_ext и список имен переменных,
    # добавляя по одному столбцу на каждую добавленную переменную
    m = len(A)
    n = len(A[0])
    A_ext = [row[:] for row in A]
    var_names = ["x"+str(j+1) for j in range(n)]
    art_indices = []
    # для каждого ограничения добавим соответствующие новые столбцы в A_ext
    for i, rel in enumerate(senses):
        if rel == "<=":
            # добавить slack
            for r in range(m):
                A_ext[r].append(1.0 if r==i else 0.0)
            var_names.append(f"s{i+1}")
        elif rel == ">=":
            # surplus (-1) и artificial (+1)
            for r in range(m):
                A_ext[r].append(-1.0 if r==i else 0.0)
            var_names.append(f"sur{i+1}")
            for r in range(m):
                A_ext[r].append(1.0 if r==i else 0.0)
            var_names.append(f"a{i+1}")
            art_indices.append(len(var_names)-1)
        elif rel == "=":
            for r in range(m):
                A_ext[r].append(1.0 if r==i else 0.0)
            var_names.append(f"a{i+1}")
            art_indices.append(len(var_names)-1)
        else:
            raise ValueError("Unknown relation: "+str(rel))
    return A_ext, var_names, art_indices

def two_phase_solve(c, A, senses, b):
    # build extended
    A_ext, var_names, art_indices = build_extended(A, senses)
    m = len(A_ext); n_ext = len(A_ext[0])
    report = []
    # initial basis: choose unit columns (prefer slack or artificial)
    basis = [-1]*m
    for j in range(n_ext):
        col = [A_ext[i][j] for i in range(m)]
        ones = [i for i,v in enumerate(col) if abs(v-1.0) < EPS]
        if len(ones)==1 and all(abs(v) < EPS or abs(v-1.0) < EPS for v in col):
            i = ones[0]
            if basis[i] == -1:
                basis[i] = j
    # ensure every row has basic var
    for i in range(m):
        if basis[i] == -1:
            # try to locate an artificial variable in this row
            found = False
            for j in art_indices:
                if abs(A_ext[i][j] - 1.0) < EPS:
                    basis[i] = j; found=True; break
            if not found:
                # fallback: pick any nonzero column index
                for j in range(n_ext):
                    if abs(A_ext[i][j]) > EPS:
                        basis[i] = j; found=True; break
            if not found:
                raise RuntimeError("Cannot find basis for row "+str(i))
    # Фаза I: минимизация суммы искусственных -> приведём как MAXIMIZE (для удобства можно сделать минимизацию, но здесь сделаем maximize of -sum)
    c_phase1 = [0.0]*n_ext
    for j in art_indices: c_phase1[j] = 1.0
    # build tableau for phase1 (we will minimize sum -> convert to MAX by negating)
    tableau = make_tableau(A_ext, b, [-v for v in c_phase1] + [0.0])
    # adjust bottom row for initial basic variables
    for i in range(m):
        bj = basis[i]
        coeff = -c_phase1[bj]  # because bottom row uses -c_phase1
        if abs(coeff) > EPS:
            tableau[-1] = [tableau[-1][j] - coeff * tableau[i][j] for j in range(n_ext+1)]
    report.append("=== Phase I: minimize sum of artificial vars ===\n")
    print_tableau_simple(tableau, var_names, basis, title="Initial Phase I tableau", report_lines=report)
    res1 = simplex_core(tableau, basis[:], var_names, maximize=True, report=[])
    if res1["status"] != "optimal":
        return {"status":"phase1_failed", "report": res1.get("report", [])}
    # compute min sum of artificials: since we maximized -sum, the value is -obj
    min_sum_art = -res1["tableau"][-1][-1]
    report += res1["report"]
    report.append(f"Phase I minimal sum of artificials = {min_sum_art}\n")
    if abs(min_sum_art) > 1e-6:
        report.append("Problem is infeasible (artificials > 0)\n")
        return {"status":"infeasible", "report":report}
    # take tableau and basis from phase1
    tableau = res1["tableau"]
    basis = res1["basis"]
    # try to pivot out artificial variables if they remain basic
    for i in range(m):
        bj = basis[i]
        if bj in art_indices:
            for j in range(n_ext):
                if j in art_indices: continue
                if abs(tableau[i][j]) > EPS:
                    # pivot on (i,j)
                    pivot = tableau[i][j]
                    tableau[i] = [v / pivot for v in tableau[i]]
                    for r in range(m+1):
                        if r == i: continue
                        fac = tableau[r][j]
                        if abs(fac) > EPS:
                            tableau[r] = [tableau[r][k] - fac * tableau[i][k] for k in range(n_ext+1)]
                    basis[i] = j
                    break
    # remove artificial columns
    keep_cols = [j for j in range(n_ext) if j not in art_indices]
    new_var_names = [var_names[j] for j in keep_cols]
    new_n = len(keep_cols)
    new_tableau = [[0.0]*(new_n+1) for _ in range(m+1)]
    for i in range(m):
        for newj,j in enumerate(keep_cols):
            new_tableau[i][newj] = tableau[i][j]
        new_tableau[i][-1] = tableau[i][-1]
    for newj,j in enumerate(keep_cols):
        new_tableau[-1][newj] = tableau[-1][j]
    new_tableau[-1][-1] = tableau[-1][-1]
    # update basis indices to new indexing where possible
    new_basis = [-1]*m
    for i in range(m):
        bj = basis[i]
        if bj in keep_cols:
            new_basis[i] = keep_cols.index(bj)
    # fix missing basis rows
    for i in range(m):
        if new_basis[i] == -1:
            found=False
            for j in range(new_n):
                col = [new_tableau[r][j] for r in range(m)]
                ones = [r for r,val in enumerate(col) if abs(val-1.0)<EPS]
                if len(ones)==1 and ones[0]==i and all(abs(col[r])<EPS for r in range(m) if r!=i):
                    new_basis[i] = j; found=True; break
            if not found:
                for j in range(new_n):
                    if abs(new_tableau[i][j])>EPS:
                        pivot = new_tableau[i][j]
                        new_tableau[i] = [v/pivot for v in new_tableau[i]]
                        for r in range(m+1):
                            if r==i: continue
                            fac = new_tableau[r][j]
                            if abs(fac)>EPS:
                                new_tableau[r] = [new_tableau[r][k] - fac*new_tableau[i][k] for k in range(new_n+1)]
                        new_basis[i]=j; found=True; break
            if not found:
                raise RuntimeError("Cannot establish basis after removing artificials for row "+str(i))
    # Phase II: build objective row from original c (keep only keep_cols)
    c_full = [0.0]*len(var_names)
    for i in range(len(c)): c_full[i] = c[i] if i < len(c) else 0.0
    c_no_art = [c_full[j] for j in keep_cols]
    # convert min to max by negation
    # NOTE: original problem is minimization; use maximize with -c
    c_no_art = [-ci for ci in c_no_art]
    # compute reduced cost row: obj = c - sum(cb * row)
    obj_row = [ci for ci in c_no_art] + [0.0]
    for i in range(m):
        bj = new_basis[i]
        if bj < 0: continue
        cb = c_no_art[bj]
        if abs(cb) > EPS:
            obj_row = [obj_row[j] - cb * new_tableau[i][j] for j in range(new_n+1)]
    new_tableau[-1] = obj_row[:]
    report.append("\n=== Phase II initial tableau ===\n")
    print_tableau_simple(new_tableau, new_var_names, new_basis, title="Phase II initial tableau", report_lines=report)
    res2 = simplex_core(new_tableau, new_basis[:], new_var_names, maximize=True, report=[])
    report += res2["report"]
    if res2["status"] != "optimal":
        return {"status":"phase2_failed", "report":report}
    final_tableau = res2["tableau"]; final_basis = res2["basis"]
    # map solution to original variables (first len(c))
    sol_ext = [0.0]*len(new_var_names)
    for i in range(m):
        bj = final_basis[i]
        if 0 <= bj < len(new_var_names):
            sol_ext[bj] = final_tableau[i][-1]
    sol_orig = [0.0]*len(c)
    for newj, orig_col in enumerate(keep_cols):
        if orig_col < len(c):
            sol_orig[orig_col] = sol_ext[newj]
    final_obj = sum(c[i]*sol_orig[i] for i in range(len(c)))
    report.append("Final solution: x = " + str([round(v,6) for v in sol_orig]) + "\n")
    report.append(f"Final objective Z = {final_obj}\n")
    return {"status":"optimal", "x":sol_orig, "obj":final_obj, "report":report}

# -----------------------------
# Test example (ваш вариант)
# -----------------------------
if __name__ == "__main__":
    c = [1,2,3,1]  # целевая (min)
    A = [
        [1,2,1,0],
        [0,1,1,1],
        [1,0,0,1]
    ]
    senses = ["<=", "=", ">="]
    b = [7,6,2]

    res = two_phase_solve(c, A, senses, b)
    print("\n===== RESULT =====")
    if res["status"] == "optimal":
        for i,val in enumerate(res["x"], start=1):
            print(f"x{i} = {val:.6f}")
        print(f"Z = {res['obj']:.6f}")
    else:
        print("Status:", res["status"])
    # записать отчёт в файл
    with open("report.txt", "w", encoding="utf-8") as f:
        f.writelines(res.get("report", []))
    print("\nFull report written to report.txt")
