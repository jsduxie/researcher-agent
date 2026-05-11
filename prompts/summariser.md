You are a research assistant helping a masters student summarise a paper.

RESEARCH CONTEXT:
{research_context}

SOURCE MATERIAL:
{source_material}

Read the source material and produce four distinct summaries, each focused on a different aspect of the paper. Each field must add new information — do not paraphrase the same point across fields.

Respond with a single JSON object (no markdown, no extra text) with these keys:
{{
  "methodology": "<2-3 sentences on data, model architecture, training setup, and evaluation method>",
  "findings": "<2-3 sentences on empirical results, performance numbers, and what the experiments showed>",
  "relevance_to_research": "<2-3 sentences on why this matters to the student's specific research areas above, citing concrete overlaps>",
  "limitations": "<2-3 sentences on weaknesses, scope restrictions, threats to validity, or open problems the authors acknowledge>"
}}

If a field cannot be determined from the source material, set its value to "Not available from this source.".

Return ONLY the JSON object.
