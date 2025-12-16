import random
from sympy import (
    symbols, Matrix, diff, integrate, exp, sin, cos, tan, log, # <--- Added log, tan
    latex, eye, simplify, expand, oo, sqrt, pi, together, Rational     # <--- Added together
)

from sympy.abc import x

class MathEngine:
    def __init__(self):
        self.C1, self.C2 = symbols('C1 C2')

    def generate_eigenvalue_problem(self):
        n = random.choice([2, 3])
        evals = [random.randint(-4, 4) for _ in range(n)]
        D = eye(n)
        for i in range(n):
            D[i, i] = evals[i]

        while True:
            P = Matrix(n, n, lambda i, j: random.randint(-2, 2))
            if abs(P.det()) == 1: 
                break
                
        A = P * D * P.inv()
        
        return {
            "topic": "Linear Algebra",   # <--- This was likely missing
            "type": "Eigenvalues",
            "problem": f"Find the eigenvalues of matrix A:\n\n$$ A = {latex(A)} $$",
            "solution": f"$$ \\lambda = {sorted(evals)} $$"
        }

    def generate_definite_integral_reverse(self):
        term1 = random.choice([x, x**2, exp(x), exp(2*x)])
        term2 = random.choice([sin(x), cos(x), exp(-x), 1])
        coeff = random.randint(1, 3)
        F = coeff * term1 * term2
        
        f_problem = diff(F, x)
        f_problem = simplify(f_problem)

        a = 0
        if F.has(sin) or F.has(cos):
            b = random.choice([symbols('pi'), symbols('pi')/2])
        else:
            b = random.randint(1, 2)

        exact_value = integrate(f_problem, (x, a, b))

        return {
            "topic": "Calculus",         # <--- This was likely missing
            "type": "Definite Integral",
            "problem": f"Evaluate the definite integral:\n\n$$ \\int_{{{latex(a)}}}^{{{latex(b)}}} \\left( {latex(f_problem)} \\right) \\, dx $$",
            "solution": f"$$ {latex(exact_value)} $$"
        }

    def generate_second_order_ode(self):
        case = random.choice(['real_distinct', 'real_repeated', 'complex'])
        
        if case == 'real_distinct':
            r1 = random.randint(-4, 4)
            r2 = random.randint(-4, 4)
            while r1 == r2: r2 = random.randint(-4, 4)
            b = -(r1 + r2)
            c = r1 * r2
            solution_latex = f"y(x) = C_1 e^{{{r1}x}} + C_2 e^{{{r2}x}}"

        elif case == 'real_repeated':
            r = random.randint(-3, 3)
            b = -2 * r
            c = r**2
            solution_latex = f"y(x) = C_1 e^{{{r}x}} + C_2 x e^{{{r}x}}"

        elif case == 'complex':
            alpha = random.randint(-2, 2)
            beta = random.randint(1, 4)
            b = -2 * alpha
            c = alpha**2 + beta**2
            if alpha == 0: exp_part = ""
            else: exp_part = f"e^{{{alpha}x}}"
            solution_latex = f"y(x) = {exp_part}(C_1 \\cos({beta}x) + C_2 \\sin({beta}x))"

        b_str = f"+ {b}" if b >= 0 else f"{b}"
        c_str = f"+ {c}" if c >= 0 else f"{c}"
        
        ode_tex = "y''"
        if b != 0:
            term = f"{abs(b)}y'" if abs(b) != 1 else "y'"
            sign = "+" if b > 0 else "-"
            ode_tex += f" {sign} {term}"
        if c != 0:
            term = f"{abs(c)}y" if abs(c) != 1 else "y"
            sign = "+" if c > 0 else "-"
            ode_tex += f" {sign} {term}"
        ode_tex += " = 0"

        return {
            "topic": "Differential Equations",  # <--- This was likely missing
            "type": "2nd Order Homogeneous",
            "problem": f"Find the general solution:\n\n$$ {ode_tex} $$",
            "solution": f"$$ {solution_latex} $$"
        }
    
    def generate_partial_fractions(self):
        """
        Reverse Engineer: Start with A/(x+a) + B/(x+b), combine them,
        and ask the user to integrate the combined rational function.
        """
        # 1. Choose simple integer roots for the denominator (to be easily factorable)
        # We use positive shifts (x+a) to ensure no singularities in the integration range [0, 1]
        r1 = random.randint(1, 3)
        r2 = random.randint(4, 6) # distinct from r1
        
        # 2. Choose integer numerators
        A = random.randint(1, 5)
        B = random.choice([-3, -2, -1, 1, 2, 3])
        
        # 3. Construct the 'Answer' form (separated)
        # We integrate from 0 to 1. Since r1,r2 > 0, the denominator is never 0.
        term1 = A / (x + r1)
        term2 = B / (x + r2)
        separated_expr = term1 + term2
        
        # 4. Construct the 'Problem' form (combined)
        # together() merges them: (A(x+r2) + B(x+r1)) / ((x+r1)(x+r2))
        problem_integrand = together(separated_expr)
        
        # 5. Define limits (0 to 1 is usually safe and clean for these logs)
        a, b = 0, 1
        
        # 6. Solve
        solution = integrate(problem_integrand, (x, a, b))
        
        return {
            "topic": "Calculus",
            "type": "Integration by Partial Fractions",
            "problem": f"Evaluate the integral:\n\n$$ \\int_{{{a}}}^{{{b}}} {latex(problem_integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
        }

    def generate_u_substitution(self):
        """
        Reverse Engineer: Create f(g(x)) * g'(x).
        Guarantees that u = g(x) is the valid substitution.
        """
        # 1. Pick an inner function u = g(x)
        # e.g., x^2 + 1, cos(x), etc.
        g_options = [
            (x**2 + 1, 2*x),      # u = x^2+1, du = 2x dx
            (sin(x), cos(x)),     # u = sin(x), du = cos(x) dx
            (log(x + 1), 1/(x+1)) # u = ln(x+1), du = 1/(x+1) dx
        ]
        g, g_prime = random.choice(g_options)
        
        # 2. Pick an outer function f(u)
        # e.g., u^3, 1/u^2, sqrt(u)
        n = random.randint(2, 4)
        f_u_options = [
            lambda u: u**n,          # Power rule
            lambda u: 1/u,           # Log rule
            lambda u: u**Rational(1, 2) # Sqrt
        ]
        outer_func = random.choice(f_u_options)
        
        # 3. Build the integrand: f(g(x)) * g'(x)
        integrand = outer_func(g) * g_prime
        
        # 4. Set limits suitable for the chosen inner function
        # We need limits where g(x) is clean.
        if g.has(sin):
            a, b = 0, pi/2
        elif g.has(log):
            a, b = 0, exp(1)-1 # ln(e-1+1) = ln(e) = 1
        else:
            a, b = 0, 1
            
        solution = integrate(integrand, (x, a, b))
        
        return {
            "topic": "Calculus",
            "type": "Integration by Substitution",
            "problem": f"Evaluate using substitution:\n\n$$ \\int_{{{latex(a)}}}^{{{latex(b)}}} {latex(integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
        }

    def _is_problem_acceptable(self, problem_data):
        sol = problem_data['solution']
        if sol.strip() in ["$$ 0 $$", "$$ 1 $$", "$$ 0.0 $$"]: return False
        if len(sol) > 300: return False
        return True

    def get_random_problem(self):
        generators = [
            self.generate_eigenvalue_problem,
            self.generate_definite_integral_reverse,
            self.generate_second_order_ode,
            self.generate_partial_fractions, # <--- Added
            self.generate_u_substitution  # <--- Added
        ]
        
        for _ in range(10):
            generator = random.choice(generators)
            try:
                problem_data = generator()
                if self._is_problem_acceptable(problem_data):
                    return problem_data
            except Exception:
                continue
        
        # Fallback must ALSO have the 'topic' key
        return {
            "topic": "System",
            "type": "Error",
            "problem": "Could not generate a clean problem. Please retry.",
            "solution": ""
        }