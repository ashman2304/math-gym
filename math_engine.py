import random
from sympy import (
    symbols, Matrix, diff, integrate, exp, sin, cos, tan, log, # <--- Added log, tan
    latex, eye, simplify, expand, oo, sqrt, pi, together, Rational     # <--- Added together
)

from sympy.abc import x

class MathEngine:
    def __init__(self):
        self.C1, self.C2 = symbols('C1 C2')

        self.topic_map = {
            "Linear Algebra": [
                self.generate_eigenvalue_problem
            ],
            "Calculus": [
                self.generate_definite_integral_reverse,
                self.generate_improper_gamma,
                self.generate_gaussian_integral,
                self.generate_partial_fractions,
                self.generate_u_substitution,
                self.generate_integration_by_parts
            ],
            "Differential Equations": [
                self.generate_second_order_ode
            ]
        }

    def generate_eigenvalue_problem(self):
        # Here we first randomly choose either a 2 or a 3 dimensional matrix
        # With create a list called 'evals' we then choose two or three 
        # values for the off diagonal elements
        # We construct a diagonal matrix using 'eye' then set the off diagonal
        # elements to the chosen values
        n = random.choice([2, 3])
        evals = [random.randint(-4, 4) for _ in range(n)]
        D = eye(n)
        for i in range(n):
            D[i, i] = evals[i]

        # We create a random n x n integer matrix P
        # The lambda function ensures the elements are randomly chosen
        # each time the function is called.
        # The if loop ensures the determinant of P is 1 or -1
        # if it is, we break the loop and use P, otherwise we try again
        # This ensures P is invertible and has integer inverse
        while True:
            P = Matrix(n, n, lambda i, j: random.randint(-2, 2))
            if abs(P.det()) == 1: 
                break


        A = P * D * P.inv()
        
        return {
            "topic": "Linear Algebra",
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
    
    def generate_improper_gamma(self):
        n = random.randint(1, 5)
        a = random.randint(1, 4)
        integrand = x**n * exp(-a * x)
        solution = integrate(integrand, (x, 0, oo))
        return {
            "topic": "Calculus", "type": "Improper Integral (Gamma)",
            "problem": f"Evaluate:\n\n$$ \\int_{{0}}^{{\\infty}} {latex(integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
        }
    
    def generate_gaussian_integral(self):
        a = random.choice([1, 2, 4, 9, 16])
        integrand = exp(-a * x**2)
        solution = integrate(integrand, (x, -oo, oo))
        return {
            "topic": "Calculus", "type": "Gaussian Integral",
            "problem": f"Evaluate:\n\n$$ \\int_{{-\\infty}}^{{\\infty}} {latex(integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
        }
    
    
    def generate_u_substitution(self):
        g_options = [(x**2 + 1, 2*x), (sin(x), cos(x)), (log(x + 1), 1/(x+1))]
        g, g_prime = random.choice(g_options)
        n = random.randint(2, 4)
        f_u_options = [lambda u: u**n, lambda u: 1/u, lambda u: u**Rational(1, 2)]
        outer_func = random.choice(f_u_options)
        integrand = outer_func(g) * g_prime
        if g.has(sin): a, b = 0, pi/2
        elif g.has(log): a, b = 0, exp(1)-1
        else: a, b = 0, 1
        solution = integrate(integrand, (x, a, b))
        return {
            "topic": "Calculus", "type": "U-Substitution",
            "problem": f"Evaluate:\n\n$$ \\int_{{{latex(a)}}}^{{{latex(b)}}} {latex(integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
        }
    
    def generate_integration_by_parts(self):
        """
        Generates int u dv.
        Covers x*e^x, x*sin(x), and ln(x) types.
        """
        case = random.choice(['x_exp', 'x_trig', 'log'])
        
        if case == 'x_exp':
            # Type: x * e^(ax)
            a_val = random.choice([1, 2, -1, -2])
            integrand = x * exp(a_val * x)
            # Limits: 0 to 1 is clean
            a, b = 0, 1
            
        elif case == 'x_trig':
            # Type: x * sin(kx) or x * cos(kx)
            func = random.choice([sin, cos])
            k = random.randint(1, 2)
            integrand = x * func(k * x)
            # Limits: 0 to pi or pi/2
            a, b = 0, pi
            if k == 2: b = pi/2
            
        elif case == 'log':
            # Type: x^n * ln(x) (Requires swapping u and dv)
            n = random.randint(0, 2) # 0 means just ln(x)
            integrand = x**n * log(x)
            # Limits: 1 to e (eliminates the ln(1)=0 and ln(e)=1 nicely)
            a, b = 1, exp(1)
            
        solution = integrate(integrand, (x, a, b))
        
        return {
            "topic": "Calculus",
            "type": "Integration by Parts",
            "problem": f"Evaluate using integration by parts:\n\n$$ \\int_{{{latex(a)}}}^{{{latex(b)}}} {latex(integrand)} \\, dx $$",
            "solution": f"$$ {latex(solution)} $$"
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

    def get_random_problem(self, topic_request="Mix"):
            """
            topic_request: "Mix", "Linear Algebra", "Calculus", etc.
            """
            # 1. Decide which pool of functions to draw from
            if topic_request == "Mix" or topic_request is None:
                # Flatten all lists into one big list of choices
                valid_generators = []
                for func_list in self.topic_map.values():
                    valid_generators.extend(func_list)
            else:
                # Pick only the list for the requested topic
                valid_generators = self.topic_map.get(topic_request, [])

            # Safety check: if topic not found, default to everything
            if not valid_generators:
                for func_list in self.topic_map.values():
                    valid_generators.extend(func_list)

            # 2. Try to generate a valid problem
            for _ in range(10):
                generator = random.choice(valid_generators)
                try:
                    problem_data = generator()
                    if self._is_problem_acceptable(problem_data):
                        return problem_data
                except Exception:
                    continue
            
            return {
                "topic": "System", "type": "Error",
                "problem": "Could not generate a clean problem. Please retry.", "solution": ""
            }