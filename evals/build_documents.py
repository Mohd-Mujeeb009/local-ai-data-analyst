"""
Generate the document corpus and question set for the retrieval evaluation.

Five synthetic reports, each long enough that a 12,000-character truncation
loses most of it. Questions are attached to specific sections, so the target of
each question is known by construction and Recall@k is measurable without an
LLM in the loop.

The corpus is plain text rather than PDF on purpose: the retrieval pipeline
consumes *extracted* text, so `pypdf` output is where its input begins. Writing
real PDFs only to extract them again would test pypdf, not retrieval, and would
add a PDF-authoring dependency for no signal.
"""

import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "evals" / "documents"

FILLER = (
    "Management continues to monitor conditions across the portfolio and to "
    "allocate capital toward the highest-returning opportunities available. "
    "The board reviewed performance against plan and against the prior "
    "comparable period, and considered the sensitivity of the outlook to the "
    "assumptions described elsewhere in this document. "
)


def pad(paragraphs=3):
    """Filler prose, so the target facts are not trivially adjacent."""
    return "\n\n".join(FILLER * 2 for _ in range(paragraphs))


# Each entry: (section number, heading, body carrying the fact, questions)
# Questions late in a document are the interesting ones - truncation cannot
# reach them, so they separate retrieval from prefix-reading.
REPORTS = {
    "acme_annual_2025": {
        "title": "ACME CORPORATION ANNUAL REPORT 2025",
        "sections": [
            ("1", "Executive Summary",
             "Group revenue reached 1,240 million dollars, up 7% year over year. "
             "Operating margin was 15.4%.",
             [("What was ACME group revenue?", None),
              ("What was ACME's operating margin?", None)]),
            ("2", "Business Overview", "ACME operates in four segments.", []),
            ("2.1", "Industrial Products",
             "Industrial Products generated 512 million dollars, growing 4%.",
             [("How did the Industrial Products segment perform?", None)]),
            ("2.2", "Consumer Goods",
             "Consumer Goods generated 388 million dollars, declining 2% on "
             "weaker discretionary demand.",
             [("How did ACME Consumer Goods perform?", None),
              ("Why did Consumer Goods decline?", None)]),
            ("3", "Financial Review", "The following sections review results.", []),
            ("3.1", "Revenue Analysis",
             "Organic growth contributed 5 points and acquisitions 2 points.",
             [("How much of ACME's growth was organic?", None)]),
            ("3.2", "Cost of Goods Sold",
             "Cost of goods sold was 742 million dollars, or 59.8% of revenue.",
             [("What was ACME's cost of goods sold?", None)]),
            ("3.3", "Operating Expenses",
             "Operating expenses totalled 306 million dollars, including 44 "
             "million dollars of restructuring charges.",
             [("How much did ACME spend on restructuring?", None)]),
            ("4", "Risk Factors", "The principal risks are set out below.", []),
            ("4.1", "Supply Chain Concentration",
             "We depend on three contract manufacturers located in two "
             "countries. A disruption at any one would delay shipments by an "
             "estimated six to nine weeks.",
             [("How long would a manufacturing disruption delay shipments?", None)]),
            ("4.2", "Currency Exposure",
             "Approximately 41% of revenue is denominated in currencies other "
             "than the reporting currency.",
             [("What proportion of ACME revenue is in foreign currency?", None)]),
            ("4.3", "Regulatory Compliance",
             "Pending environmental legislation may add 18 million dollars of "
             "annual compliance cost from 2027.",
             [("What is the expected cost of pending environmental rules?", None)]),
            ("5", "Governance",
             "The board met eleven times during the year.",
             [("How often did the ACME board meet?", None)]),
            ("6", "Outlook",
             "We expect revenue growth of 5 to 7% and margin expansion toward "
             "16.5% in the coming year.",
             [("What is ACME's revenue outlook?", None)]),
        ],
    },
    "northwind_q3": {
        "title": "NORTHWIND LOGISTICS Q3 TRADING UPDATE",
        "sections": [
            ("1", "Highlights",
             "Freight volumes rose 9% and revenue per shipment fell 3%.",
             [("What happened to Northwind revenue per shipment?", None),
              ("How much did Northwind freight volumes rise?", None)]),
            ("2", "Network Performance", "Performance varied by corridor.", []),
            ("2.1", "Transatlantic",
             "Transatlantic volumes grew 14%, the strongest corridor in the "
             "network.",
             [("Which corridor grew fastest for Northwind?", None)]),
            ("2.2", "Intra-Asia",
             "Intra-Asia volumes were flat, held back by port congestion at two "
             "hubs.",
             [("Why were Northwind's Intra-Asia volumes flat?", None)]),
            ("3", "Fleet and Capacity", "Capacity was broadly stable.", []),
            ("3.1", "Owned Fleet",
             "The owned fleet comprised 218 vehicles at quarter end, after 12 "
             "retirements and 20 additions.",
             [("How many vehicles does Northwind own?", None)]),
            ("3.2", "Chartered Capacity",
             "Chartered capacity represented 27% of total capacity, up from "
             "22%.",
             [("What share of Northwind capacity is chartered?", None)]),
            ("4", "Cost Base", "Costs are analysed below.", []),
            ("4.1", "Fuel",
             "Fuel represented 31% of operating cost. A ten percent move in "
             "fuel price changes operating profit by 6.2 million dollars.",
             [("What is Northwind's sensitivity to fuel prices?", None)]),
            ("4.2", "Labour",
             "Labour cost rose 6% following the collective agreement concluded "
             "in May.",
             [("Why did Northwind's labour costs rise?", None)]),
            ("5", "Outlook",
             "Full year volume growth is expected at 7 to 8%.",
             [("What volume growth does Northwind expect?", None)]),
        ],
    },
    "helios_product": {
        "title": "HELIOS PLATFORM TECHNICAL SPECIFICATION",
        "sections": [
            ("1", "Introduction", "This document specifies the Helios platform.", []),
            ("2", "Architecture", "Helios is a distributed system.", []),
            ("2.1", "Ingestion Layer",
             "The ingestion layer accepts up to 40,000 events per second per "
             "region.",
             [("What is the Helios ingestion throughput?", None)]),
            ("2.2", "Storage Layer",
             "Events are retained for 90 days in hot storage and 400 days in "
             "cold storage.",
             [("How long does Helios retain events?", None),
              ("How long is Helios cold storage retention?", None)]),
            ("2.3", "Query Layer",
             "The query layer targets a p99 latency of 250 milliseconds.",
             [("What is the Helios query latency target?", None)]),
            ("3", "Deployment", "Helios deploys to three regions.", []),
            ("3.1", "Regional Failover",
             "Failover between regions completes within four minutes of "
             "detection.",
             [("How fast is Helios regional failover?", None)]),
            ("4", "Security", "Security controls are described below.", []),
            ("4.1", "Encryption",
             "Data is encrypted with AES-256 at rest and TLS 1.3 in transit.",
             [("What encryption does Helios use?", None)]),
            ("4.2", "Access Control",
             "Access uses role-based control with a maximum of 64 roles per "
             "tenant.",
             [("How many roles can a Helios tenant define?", None)]),
            ("5", "Limits and Quotas",
             "Each tenant is limited to 500 concurrent queries and 2 terabytes "
             "of hot storage by default.",
             [("What are the default Helios tenant quotas?", None)]),
            ("6", "Support",
             "Priority one incidents receive a response within 15 minutes.",
             [("What is the Helios priority one response time?", None)]),
        ],
    },
    "vertex_market": {
        "title": "VERTEX RESEARCH MARKET OUTLOOK",
        "sections": [
            ("1", "Summary", "The market grew 11% to 84 billion dollars.",
             [("How large is the market in the Vertex outlook?", None)]),
            ("2", "Demand Drivers", "Three drivers dominate.", []),
            ("2.1", "Enterprise Adoption",
             "Enterprise adoption reached 62% of surveyed organisations, up "
             "from 51%.",
             [("What is the enterprise adoption rate?", None)]),
            ("2.2", "Small Business",
             "Small business adoption remains low at 19%, constrained by "
             "implementation cost.",
             [("Why is small business adoption low?", None)]),
            ("3", "Competitive Landscape", "The market remains fragmented.", []),
            ("3.1", "Market Share",
             "The top three vendors hold a combined 38% share; no single vendor "
             "exceeds 16%.",
             [("What share do the top three vendors hold?", None)]),
            ("3.2", "Pricing Trends",
             "Average selling prices declined 4% as entry-tier products "
             "proliferated.",
             [("What happened to average selling prices?", None)]),
            ("4", "Regional Analysis", "Growth varied by region.", []),
            ("4.1", "North America",
             "North America grew 9% and represents 44% of the market.",
             [("What share of the market is North America?", None),
              ("How fast did North America grow?", None)]),
            ("4.2", "Europe",
             "Europe grew 7%, held back by extended procurement cycles "
             "averaging 8.5 months.",
             [("How long are European procurement cycles?", None)]),
            ("4.3", "Asia Pacific",
             "Asia Pacific grew 21%, the fastest of any region, reaching 18 "
             "billion dollars.",
             [("Which region grew fastest in the Vertex outlook?", None)]),
            ("5", "Forecast",
             "We forecast a compound annual growth rate of 13% through 2029.",
             [("What is the forecast growth rate?", None)]),
        ],
    },
    "orion_policy": {
        "title": "ORION GROUP EMPLOYEE POLICY HANDBOOK",
        "sections": [
            ("1", "Purpose", "This handbook sets out group policy.", []),
            ("2", "Working Arrangements", "Arrangements are described below.", []),
            ("2.1", "Hybrid Working",
             "Employees are expected on site a minimum of two days per week.",
             [("How many days on site are required at Orion?", None)]),
            ("2.2", "Core Hours",
             "Core hours are 10:00 to 16:00 in the employee's local time zone.",
             [("What are Orion's core hours?", None)]),
            ("3", "Leave", "Leave entitlements follow.", []),
            ("3.1", "Annual Leave",
             "Full-time employees receive 28 days of annual leave plus public "
             "holidays.",
             [("How much annual leave do Orion employees get?", None),
              ("Are public holidays included in Orion annual leave?", None)]),
            ("3.2", "Parental Leave",
             "Primary carers receive 26 weeks at full pay; secondary carers "
             "receive 12 weeks.",
             [("What parental leave does Orion offer secondary carers?", None)]),
            ("3.3", "Carry Over",
             "A maximum of five days may be carried into the following year.",
             [("How many leave days can be carried over at Orion?", None)]),
            ("4", "Expenses", "Expense policy is set out below.", []),
            ("4.1", "Travel",
             "Economy class is standard; business class requires director "
             "approval for flights over six hours.",
             [("When can Orion staff fly business class?", None)]),
            ("4.2", "Equipment",
             "Employees may claim up to 400 dollars every three years for home "
             "office equipment.",
             [("What is Orion's home office equipment allowance?", None)]),
            ("5", "Conduct",
             "Breaches are handled under the disciplinary procedure, which "
             "provides for a written response within ten working days.",
             [("How long does Orion's disciplinary response take?", None)]),
        ],
    },
}


def render(report):
    """Render one report as extracted-PDF-style text."""
    parts = [report["title"], "", pad(1), ""]
    for number, heading, body, _ in report["sections"]:
        parts.append(f"{number} {heading}")
        parts.append("")
        parts.append(body)
        parts.append("")
        parts.append(pad(2))
        parts.append("")
    return "\n".join(parts)


def build():
    """Write the corpus and question set."""
    DOCS.mkdir(parents=True, exist_ok=True)

    questions = []
    stats = []

    for doc_name, report in REPORTS.items():
        text = render(report)
        (DOCS / f"{doc_name}.txt").write_text(text, encoding="utf-8")

        # Character offset of each section, to mark which fall past a 12k cut.
        for number, heading, body, qs in report["sections"]:
            # A section may carry more than one question, so the index is part
            # of the id - otherwise they collide and the set silently shrinks.
            for position, (question, _) in enumerate(qs, 1):
                offset = text.index(body)
                questions.append({
                    "id": f"{doc_name}_{number.replace('.', '_')}_q{position}",
                    "doc": doc_name,
                    "question": question,
                    "target_heading": heading,
                    "target_offset": offset,
                    "beyond_truncation": offset > 12_000,
                })

        stats.append((doc_name, len(text)))

    with open(ROOT / "evals" / "pdf_questions.yaml", "w", encoding="utf-8") as handle:
        handle.write(
            "# Retrieval evaluation set.\n"
            "#\n"
            "# Each question targets a known section, so Recall@k is measurable\n"
            "# without an LLM. `beyond_truncation` marks questions whose answer\n"
            "# lies past the first 12,000 characters - the ones a truncated\n"
            "# context provably cannot reach.\n"
            "#\n"
            "# Regenerate with evals/build_documents.py\n\n"
        )
        yaml.safe_dump(
            json.loads(json.dumps(questions)),
            handle, sort_keys=False, allow_unicode=True, width=100,
        )

    print(f"wrote {len(REPORTS)} documents, {len(questions)} questions")
    for name, size in stats:
        print(f"  {name:22} {size:>7,} chars")
    beyond = sum(1 for q in questions if q["beyond_truncation"])
    print(f"  {beyond}/{len(questions)} questions answerable only beyond 12k chars")


if __name__ == "__main__":
    build()
