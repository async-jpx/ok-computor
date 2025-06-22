import argparse
import logging
import re
from typing import Any

logging.basicConfig(
    level=logging.DEBUG,
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
        exponents_pattern = r"\^(\d+)"
        exponents = re.findall(exponents_pattern, self.expression)
        max_degree = max(map(int, exponents))
        return max_degree

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
        coefficient_str = term.split("*")[0]
        coefficient = float(term.split("*")[0])
        coefficient = coefficient * -1
        return term.replace(coefficient_str, str(coefficient))

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

    def _solve_quadratic(self) -> tuple[float, float] | float:
        coefficients = self.reduce()
        a = coefficients[A]
        b = coefficients[B]
        c = coefficients[C]

        if not a and not b and not c:
            raise ValueError("no solution")

        if not a:
            return self._solve_linear()

        x1 = ((-1 * b) - sqrt((b * b) - (4 * a * c))) / (2 * a)
        x2 = ((-1 * b) + sqrt((b * b) - (4 * a * c))) / (2 * a)
        return (x1, x2)

    def solve(self) -> tuple[float, float] | float:
        """Solve the polynomial equation."""
        match self.degree:
            case 2:
                return self._solve_quadratic()
            case 1:
                return self._solve_linear()
            case 0:
                raise ValueError("solution is R")
            case _:
                raise Exception("not supported")


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


def parse_polynomial_from_args() -> str:
    """Parse the polynomial expression from command-line arguments.

    Returns:
             str: The polynomial expression.
    """
    parser = argparse.ArgumentParser(description="A simple polynomial calculator.")
    parser.add_argument("polynomial", type=str, help="The polynomial to calculate.")
    args = parser.parse_args()
    polynomial = args.polynomial
    if not is_validate_polynomial(polynomial):
        raise ValueError("Invalid polynomial expression.")
    return polynomial


def main() -> None:
    try:
        polynomial_expression = parse_polynomial_from_args()
        p = Polynomial(polynomial_expression)
        logging.info(f"Reduced form {p.reduced_form}")
        logging.info(f"Degree {p.degree}")
        solution: tuple[float, float] | float = p.solve()
        logging.info(f"Solution: {solution}")
    except Polynomial.MaxDegree as e:
        logging.error(f"Error: {e.message}")
    except Polynomial.Unsolvable as e:
        logging.info(e.message)
    except Exception as e:
        logging.error(f"Error getting polynomial: {e}")
        return


if __name__ == "__main__":
    main()
