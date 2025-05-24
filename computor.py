import argparse
import logging
import re

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

example_expression = "5 * X^0 + 4 * X^1 - 9.3 * X^2 = 1 * X^0"


class Polynomial:
    degree: int
    terms: list

    def __init__(self, expression: str) -> None:
        self.expression: str = expression
        self.terms = self._extract_terms()
        self.degree = self._extract_degree()

    def __str__(self) -> str:
        return self.expression

    def _extract_degree(self) -> int:
        """Extract the degree of the polynomial from the expression."""
        return 2

    def _filp_term_sign(self, term: str) -> str:
        """Flip the sign of a term."""
        if term.startswith("-"):
            return term[1:]

    def _extract_terms(self) -> list:
        """Extract the terms of the polynomial from the expression."""

        left_side, right_side = self.expression.replace(" ", "").split("=")
        term_pattern = r"([+-]?[^-+]+)"
        right_terms = re.findall(term_pattern, right_side)
        left_terms = re.findall(term_pattern, left_side)
        return left_terms + right_terms

    @classmethod
    def from_expression(cls, expression: str):
        """Create a Computor instance from a polynomial expression."""
        return cls(Polynomial(expression))

    def solve(self):
        """Solve the polynomial equation."""
        pass


def validate_polynomial(polynomial: str) -> bool:
    """Validate the polynomial expression.

    Args:
        polynomial (str): The polynomial expression to validate.

    Returns:
        bool: True if the polynomial is valid, False otherwise.
    """
    pattern = r"^()$"
    try:
        return True
    except Exception as e:
        logging.error(f"Error validating polynomial: {e}")
        return False


def get_polynomial() -> str:
    """Get the polynomial expression from the user and va.

    Returns:
                    str: The polynomial expression.
    """
    parser = argparse.ArgumentParser(description="A simple polynomial calculator.")
    parser.add_argument("polynomial", type=str, help="The polynomial to calculate.")
    args = parser.parse_args()
    polynomial = args.polynomial
    if not validate_polynomial(polynomial):
        raise ValueError("Invalid polynomial expression.")
    return polynomial


def main() -> None:
    try:
        polynomial = get_polynomial()
        p = Polynomial(polynomial)
        logging.info(f"Polynomial: {p}, Degree: {p.degree}, Terms: {p.terms}")
    except Exception as e:
        logging.error(f"Error getting polynomial: {e}")
        return


if __name__ == "__main__":
    main()
