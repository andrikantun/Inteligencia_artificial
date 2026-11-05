from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    predicate: str
    args: tuple[str, ...]

    def __str__(self):
        return f"{self.predicate}({', '.join(self.args)})"


def parse_atom(text):
    predicate, arguments = text.split("(", 1)
    arguments = arguments.rstrip(")")
    args = tuple(arg.strip() for arg in arguments.split(","))
    return Atom(predicate.strip(), args)


def is_variable(term):
    return term[0].islower()


def apply_term(term, subst):
    while is_variable(term) and term in subst:
        term = subst[term]
    return term


def apply_atom(atom, subst):
    return Atom(
        atom.predicate,
        tuple(apply_term(arg, subst) for arg in atom.args)
    )


def unify(a, b, subst=None):
    if subst is None:
        subst = {}

    else:
        subst = dict(subst)

    if isinstance(a, Atom) and isinstance(b, Atom):

        if a.predicate != b.predicate or len(a.args) != len(b.args):
            return None

        for left, right in zip(a.args, b.args):
            subst = unify(left, right, subst)

            if subst is None:
                return None

        return subst

    a = apply_term(a, subst)
    b = apply_term(b, subst)

    if a == b:
        return subst

    if is_variable(a):
        subst[a] = b
        return subst

    if is_variable(b):
        subst[b] = a
        return subst

    return None


def format_subst(subst):
    if not subst:
        return "{}"

    return "{" + ", ".join(
        f"{key}/{value}" for key, value in sorted(subst.items())
    ) + "}"
