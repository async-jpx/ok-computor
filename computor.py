import argparse
import logging
import re
import unittest
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

A = "X^2"
B = "X^1"
C = "X^0"


def sqrt(num: float, epsilon=1e-10):
    """Compute the square root of a number using the binary search (mid) method."""
    if num < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    if num == 0 or num == 1:
        return num
    low, high = (0, num) if num > 1 else (num, 1)
    while high - low > epsilon:
        mid = (low + high) / 2
        if mid * mid < num:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def abs(num: float | int):
    if num < 0:
        return num * -1
    return num


class Polynomial:
    degree: int
    terms: list[str]
    max_degree = 2

    class MaxDegree(Exception):
        def __init__(self, message: str, degree: int):
            super().__init__(message)
            self.message = message
            self.degree = degree

    class Unsolvable(Exception):
        def __init__(self, message, degree):
            super().__init__(message)
            self.message = message
            self.degree = degree

    class AllRealSolution(Unsolvable):
        pass

    def __init__(self, expression: str) -> None:
        self.expression: str = expression
        self.terms = self._extract_terms()
        self.degree = self.extract_degree()

        if self.degree > self.max_degree:
            raise self.MaxDegree(
                message=f"Polynomial of degree {self.degree} is not supported, max degree is {self.max_degree}",
                degree=self.degree,
            )

    def __str__(self) -> str:
        return self.expression

    @property
    def reduced_form(self) -> str:
        coefficients = self.reduce()
        reduced = ""
        items = enumerate(coefficients.items())
        for idx, (c, v) in items:
            is_first = bool(idx == 0)

            value = int(v) if v == int(v) else v

            if not is_first:
                reduced += " + " if value >= 0 else " - "
                reduced += f"{abs(value)} * {c}"
            else:
                reduced += f"{value} * {c}"

        reduced += " = 0"
        return reduced

    def extract_degree(self) -> int:
        """Extract the degree of the polynomial from the expression."""
        coefficients = self.reduce()
        whole_coefficients = {k: v for k, v in coefficients.items() if v != 0}
        sorted_key = sorted(whole_coefficients.keys(), reverse=True)
        if len(sorted_key) == 0:
            return 0
        else:
            return int(sorted_key[0].split("^")[1])

    def reduce(self) -> dict[str, float]:
        coefficients: dict[str, float] = {}
        for term in self.terms:
            coefficient, var = term.split("*")
            if var not in coefficients:
                coefficients[var] = float(coefficient)
            else:
                coefficients[var] += float(coefficient)
        return coefficients

    def _filp_term_sign(self, term: str) -> str:
        """Flip the sign of a term."""
        coefficient_str, degree = term.split("*")
        coefficient = float(coefficient_str)
        coefficient = coefficient * -1
        return f"{str(coefficient)}*{degree}"

    def _extract_terms(self) -> list:
        """Extract the terms of the polynomial from the expression."""

        left_side, right_side = self.expression.replace(" ", "").split("=")
        term_pattern = r"([+-]?[^-+]+)"
        right_terms = list(
            map(lambda t: self._filp_term_sign(t), re.findall(term_pattern, right_side))
        )
        left_terms = re.findall(term_pattern, left_side)
        return left_terms + right_terms

    @classmethod
    def from_expression(cls, expression: str):
        """Create a Computor instance from a polynomial expression."""
        return cls(Polynomial(expression))

    def _solve_linear(self):
        coefficients = self.reduce()
        b = coefficients[B]
        c = coefficients[C]
        x = (-1 * c) / b
        return x

    def _solve_quadratic(self) -> tuple[float, float] | float | tuple[complex, complex]:
        coefficients = self.reduce()
        a = coefficients[A]
        b = coefficients[B]
        c = coefficients[C]

        if not a and not b and not c:
            raise ValueError("no solution")

        if not a:
            return self._solve_linear()

        discriminant = (b * b) - (4 * a * c)
        if discriminant < 0:
            real = (-b) / (2 * a)
            imag = sqrt(-discriminant) / (2 * a)
            x1 = complex(real, -imag)
            x2 = complex(real, imag)
            return (x1, x2)

        x1 = ((-1 * b) - sqrt(discriminant)) / (2 * a)
        x2 = ((-1 * b) + sqrt(discriminant)) / (2 * a)
        if x1 == x2:
            return x1
        return (x1, x2)

    def _solve_constant(self):
        coefficients = self.reduce()
        c = coefficients[C]
        if c != 0:
            raise self.Unsolvable("no solution", self.degree)
        else:
            raise self.AllRealSolution("solution is R", self.degree)

    def solve(self) -> tuple[float, float] | float | tuple[complex, complex]:
        """Solve the polynomial equation."""
        match self.degree:
            case 2:
                return self._solve_quadratic()
            case 1:
                return self._solve_linear()
            case 0:
                self._solve_constant()
            case _:
                raise self.MaxDegree("not supported", self.degree)


def is_validate_polynomial(polynomial: Any) -> bool:
    """Validate the polynomial expression.

    Args:
        polynomial (Any): The polynomial expression to validate.

    Returns:
        bool: True if the polynomial is valid, False otherwise.
    """

    try:
        if not isinstance(polynomial, str):
            raise ValueError("polynomial is not a string")
        return True
    except Exception as e:
        logging.error(e)
        return False


def parse_polynomial_from_args(args) -> str:
    """Parse the polynomial expression from command-line arguments.

    Returns:
             str: The polynomial expression.
    """
    polynomial = args.polynomial
    if not is_validate_polynomial(polynomial):
        raise ValueError("Invalid polynomial expression.")
    return polynomial


def main() -> None:
    try:
        parser = argparse.ArgumentParser(description="A simple polynomial calculator.")
        parser.add_argument("polynomial", type=str, help="The polynomial to calculate.")
        args = parser.parse_args()

        polynomial_expression = parse_polynomial_from_args(args)
        p = Polynomial(polynomial_expression)
        logging.info(f"Reduced Form: {p.reduced_form}")
        logging.info(f"Polynomial degree {p.degree}")
        solution: tuple[float, float] | float = p.solve()
        logging.info(f"The Solution is: {solution}")
    except Polynomial.MaxDegree as e:
        logging.error(f"Error: {e.message}")
    except Polynomial.Unsolvable as e:
        logging.info(e.message)
    except Exception as e:
        logging.error(f"Error getting polynomial: {e}")
        return


class TestPolynomialSolver(unittest.TestCase):
    def assertComplexRoots(self, roots, expected):
        self.assertEqual(len(roots), 2)
        for r, e in zip(roots, expected):
            self.assertAlmostEqual(r.real, e.real, places=3)
            self.assertAlmostEqual(r.imag, e.imag, places=3)

    def test_no_solution(self):
        with self.assertRaises(Polynomial.Unsolvable):
            Polynomial("5 * X^0 = 3 * X^0").solve()

    def test_infinite_solutions(self):
        with self.assertRaises(Polynomial.AllRealSolution):
            Polynomial("5 * X^0 = 5 * X^0").solve()

    def test_single_real(self):
        result = Polynomial("2 * X^1 + 4 * X^0 = 0 * X^0").solve()
        self.assertAlmostEqual(result, -2.0, places=3)

    def test_degenerate_infinite(self):
        with self.assertRaises(Polynomial.AllRealSolution):
            Polynomial("0 * X^1 + 3 * X^0 = 3 * X^0").solve()

    def test_degenerate_no_solution(self):
        with self.assertRaises(Polynomial.Unsolvable):
            Polynomial("0 * X^1 + 3 * X^0 = 5 * X^0").solve()

    def test_two_distinct_real(self):
        result = Polynomial("1 * X^2 - 3 * X^1 + 2 * X^0 = 0 * X^0").solve()
        self.assertTrue(isinstance(result, tuple))
        self.assertAlmostEqual(result[0], 1.0, places=3)
        self.assertAlmostEqual(result[1], 2.0, places=3)

    def test_one_double_real(self):
        result = Polynomial("1 * X^2 - 2 * X^1 + 1 * X^0 = 0 * X^0").solve()
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_two_complex(self):
        result = Polynomial("1 * X^2 + 0 * X^1 + 1 * X^0 = 0 * X^0").solve()
        self.assertTrue(isinstance(result, tuple))
        expected = (complex(0, -1), complex(0, 1))
        self.assertComplexRoots(result, expected)

    def test_degenerate_linear(self):
        result = Polynomial("0 * X^2 + 3 * X^1 + 6 * X^0 = 0 * X^0").solve()
        self.assertAlmostEqual(result, -2.0, places=3)

    def test_degenerate_no_solution2(self):
        with self.assertRaises(Polynomial.Unsolvable):
            Polynomial("0 * X^2 + 0 * X^1 + 5 * X^0 = 0 * X^0").solve()

    def test_degenerate_infinite2(self):
        with self.assertRaises(Polynomial.AllRealSolution):
            Polynomial("0 * X^2 + 0 * X^1 + 0 * X^0 = 0 * X^0").solve()

    def test_two_complex2(self):
        result = Polynomial("-1.5 * X^2 + 3.2 * X^1 - 4.8 * X^0 = 0 * X^0").solve()
        self.assertTrue(isinstance(result, tuple))
        # x = (–3.2 ± i·√18.56)/(–3)
        real = (-3.2) / (-3)
        imag = (sqrt(18.56)) / abs(-3)
        expected = (complex(real, -imag), complex(real, imag))
        self.assertComplexRoots(result, expected)

    def test_one_double_real2(self):
        result = Polynomial("2.5 * X^2 - 5 * X^1 + 2.5 * X^0 = 0 * X^0").solve()
        self.assertAlmostEqual(result, 1.0, places=3)

    def test_two_distinct_real2(self):
        result = Polynomial("0.5 * X^2 - 1 * X^1 - 1 * X^0 = 0 * X^0").solve()
        self.assertTrue(isinstance(result, tuple))
        # x = (1 ± sqrt(3))/1 ≈ 2.732, –0.732
        self.assertAlmostEqual(result[0], 2.732, places=3)
        self.assertAlmostEqual(result[1], -0.732, places=3)


if __name__ == "__main__":
    # main()
    unittest.main()
