def wants_chart(text):
    keywords = ["chart", "graph", "plot", "visual", "compare"]
    return any(k in text.lower() for k in keywords)

def wants_table(text):
    return "table" in text.lower()
