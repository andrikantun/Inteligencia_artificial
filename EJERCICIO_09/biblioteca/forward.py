from dataclasses import dataclass, field

from atoms import Atom, apply_atom, unify, format_subst


@dataclass
class Rule:
    rule_id: str
    premises: tuple[Atom, ...]
    head: Atom


@dataclass
class InferenceResult:
    proved: bool
    subst: dict | None
    inferred: list[Atom]
    events: list[str] = field(default_factory=list)


def standardize(rule, counter):
    mapping = {}

    def rename(atom):
        args = []

        for arg in atom.args:
            if arg[0].islower():
                mapping.setdefault(arg, f"{arg}_{counter}")
                args.append(mapping[arg])
            else:
                args.append(arg)

        return Atom(atom.predicate, tuple(args))

    return Rule(
        rule.rule_id,
        tuple(rename(p) for p in rule.premises),
        rename(rule.head)
    )


def satisfy(premises, facts, subst):
    if not premises:
        return [subst]

    first = premises[0]
    rest = premises[1:]

    goal = apply_atom(first, subst)
    results = []

    for fact in facts:
        theta = unify(goal, fact, subst)

        if theta is not None:
            results.extend(satisfy(rest, facts, theta))

    return results


def forward_chain(initial_facts, rules, query, max_rounds=50):
    facts = list(initial_facts)
    inferred = []
    events = []

    events.append("Hechos iniciales:")
    events.append(", ".join(str(f) for f in facts))
    events.append(f"Consulta: {query}")

    counter = 0

    for round_no in range(1, max_rounds + 1):

        new_atoms = []

        events.append(f"\n--- Ronda {round_no} ---")

        for rule in rules:
            counter += 1
            fresh = standardize(rule, counter)

            substitutions = satisfy(fresh.premises, facts, {})

            for theta in substitutions:
                head = apply_atom(fresh.head, theta)

                if head in facts or head in new_atoms:
                    continue

                new_atoms.append(head)
                inferred.append(head)

                events.append(
                    f"Dispara {rule.rule_id} "
                    f"{format_subst(theta)} => {head}"
                )

        if not new_atoms:
            events.append("No hay hechos nuevos. Fin del algoritmo.")
            break

        facts.extend(new_atoms)

    else:
        events.append("Se alcanzó el máximo de rondas.")

    # Comprobar la consulta después de terminar todas las rondas.
    for fact in facts:
        theta = unify(query, fact, {})

        if theta is not None:
            events.append(f"Consulta probada: {query}")

            return InferenceResult(
                True, theta, inferred, events
            )

    events.append(f"Consulta no probada: {query}")

    return InferenceResult(
        False, None, inferred, events
    )