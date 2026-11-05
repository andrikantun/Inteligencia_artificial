from forward import standardize
from atoms import apply_atom, unify


def backward_chain(initial_facts, rules, query):
    events = []
    counter = 0

    def bc_or(goal, subst, stack):
        nonlocal counter

        goal = apply_atom(goal, subst)

        events.append(f"Objetivo: {goal}")

        # Evitar ciclos.
        if str(goal) in stack:
            events.append(f"Ciclo detectado: {goal}")
            return None

        # 1. Buscar primero en los hechos.
        for fact in initial_facts:
            theta = unify(goal, fact, subst)

            if theta is not None:
                events.append(f"Hecho encontrado: {fact}")
                return theta

        # 2. Intentar las reglas en su orden original.
        new_stack = stack | {str(goal)}

        for rule in rules:
            counter += 1
            fresh = standardize(rule, counter)

            theta = unify(goal, fresh.head, subst)

            if theta is None:
                continue

            events.append(f"Intentando regla: {rule.rule_id}")

            result = bc_and(
                fresh.premises,
                theta,
                new_stack
            )

            if result is not None:
                events.append(
                    f"Regla demostrada: {rule.rule_id}"
                )
                return result

        events.append(f"No se pudo demostrar: {goal}")
        return None

    def bc_and(goals, subst, stack):
        if not goals:
            return subst

        first = goals[0]
        rest = goals[1:]

        theta = bc_or(first, subst, stack)

        if theta is None:
            return None

        return bc_and(rest, theta, stack)

    result = bc_or(query, {}, set())

    return {
        "proved": result is not None,
        "subst": result,
        "events": events
    }
