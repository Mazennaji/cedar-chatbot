from src.chatbot import CedarChatbot


TEST_INPUTS = [
    ("EN  greeting",        "Hi Cedar"),
    ("EN  knowledge",       "What is NLP?"),
    ("EN  knowledge",       "What is deep learning?"),
    ("EN  identity",        "What is your name?"),
    ("EN  open chat",       "Tell me something interesting"),

    ("AR  knowledge",       "ما هو الذكاء الاصطناعي؟"),
    ("AR  knowledge",       "اشرح لي التعلم العميق"),
    ("AR  greeting",        "مرحبا"),
    ("AR  identity",        "ما اسمك؟"),
    ("AR  open",            "حدثني عن أي شيء"),

    ("ARZ greeting",        "keefak ya zalame?"),
    ("ARZ chitchat",        "ana mni7"),
    ("ARZ chitchat",        "shu 3am ta3mel?"),
    ("ARZ blessing",        "ya3tek el 3afye"),
    ("ARZ knowledge",       "shu el AI?"),
    ("ARZ identity",        "shu esmak?"),
    ("ARZ open",            "2elli shi 3an el faza2"),
]


def main():
    print("Loading Cedar (this loads BlenderBot + mt0, ~20-30s)...\n")
    bot = CedarChatbot()

    print("=" * 90)
    print(f"{'LABEL':<16}{'LANG':<10}{'INTENT':<12}{'ENGINE':<20}RESPONSE")
    print("=" * 90)

    engine_counts = {}

    for label, message in TEST_INPUTS:
        result = bot.chat(message)
        meta = result.metadata
        engine = meta.get("generator", "?")
        lang = meta.get("detected_language", "?")
        intent = meta.get("intent", "?")

        engine_counts[engine] = engine_counts.get(engine, 0) + 1

        resp = result.response.replace("\n", " ")
        if len(resp) > 50:
            resp = resp[:50] + "..."

        print(f"{label:<16}{lang:<10}{intent:<12}{engine:<20}{resp}")

    print("=" * 90)
    print("\nEngine usage summary:")
    for engine, count in sorted(engine_counts.items(), key=lambda x: -x[1]):
        print(f"  {engine:<20} {count}")

    decoder_hits = engine_counts.get("mt5_decoder", 0)
    print()
    if decoder_hits > 0:
        print(f"DECODER VERIFIED: the mt0 decoder generated {decoder_hits} response(s).")
    else:
        print("WARNING: the decoder never fired. Check that mt0 loaded (look for 'Multilingual decoder loaded successfully' on startup).")
    print()


if __name__ == "__main__":
    main()