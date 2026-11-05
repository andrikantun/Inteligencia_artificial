import yaml

from atoms import parse_atom
from forward import Rule, forward_chain
from backward import backward_chain


def load_biblioteca():
    with open("biblioteca.yaml", "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    facts = [
        parse_atom(fact)
        for fact in data["facts"]
    ]

    rules = []

    for item in data["rules"]:
        rules.append(
            Rule(
                item["id"],
                tuple(parse_atom(p) for p in item["if"]),
                parse_atom(item["then"])
            )
        )

    return facts, rules


def show_forward(facts, rules, queries):
    print("\n========== ENCADENAMIENTO HACIA ADELANTE ==========")

    for query in queries:
        print(f"\nCONSULTA: {query}")

        result = forward_chain(facts, rules, query)

        for event in result.events:
            print(event)

        print("RESULTADO:", "SI" if result.proved else "NO")
        print("HECHOS INFERIDOS:", [str(x) for x in result.inferred])


def show_backward(facts, rules, queries):
    print("\n========== ENCADENAMIENTO HACIA ATRÁS ==========")

    for query in queries:
        print(f"\nCONSULTA: {query}")

        result = backward_chain(facts, rules, query)

        for event in result["events"]:
            print(event)

        print("RESULTADO:", "SI" if result["proved"] else "NO")


def additional_test(facts, rules):
    print("\n========== PRUEBA ADICIONAL: CREDENCIAL DE LUIS ==========")

    new_facts = facts + [parse_atom("TieneCredencial(Luis)")]
    query = parse_atom("AccesoNoche(Luis)")

    print("\n--- Hacia adelante ---")
    forward_result = forward_chain(new_facts, rules, query)

    for event in forward_result.events:
        print(event)

    print("RESULTADO:", "SI" if forward_result.proved else "NO")
    print("HECHOS INFERIDOS:", [str(x) for x in forward_result.inferred])

    print("\n--- Hacia atrás ---")
    backward_result = backward_chain(new_facts, rules, query)

    for event in backward_result["events"]:
        print(event)

    print("RESULTADO:", "SI" if backward_result["proved"] else "NO")


def main():
    facts, rules = load_biblioteca()

    queries = [
        parse_atom("AccesoNoche(Ana)"),
        parse_atom("AccesoNoche(Luis)"),
        parse_atom("AccesoNoche(Marta)")
    ]

    show_forward(facts, rules, queries)
    show_backward(facts, rules, queries)
    additional_test(facts, rules)


if __name__ == "__main__":
    main()
