def get_equation_straight(dot_a: tuple[int, int], dot_b: tuple[int, int]) -> str:
    x1, y1 = dot_a
    x2, y2 = dot_b
    coefficient_a = y2 - y1
    coefficient_b = x1 - x2
    coefficient_c = coefficient_a * x1 + coefficient_b * y1

    equation_straight = f"{coefficient_a}x"

    if coefficient_b > 0:
        equation_straight += f"+{coefficient_b}y={coefficient_c}"
    elif coefficient_b < 0:
        equation_straight += f"-{abs(coefficient_b)}y={coefficient_c}"
    else:
        equation_straight = f"{coefficient_a}x={coefficient_c}"
    if coefficient_a == 0:
        equation_straight = f"{coefficient_b}y={coefficient_c}"

    return equation_straight

a = (3,2)
b = (5,7)
print("A", a, "B", b)
equation = get_equation_straight(a, b)
print(equation)
