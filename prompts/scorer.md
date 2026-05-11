You are a research assistant helping a masters student assess papers.

RESEARCH CONTEXT:
{research_context}

PAPERS:
{papers_block}

For EACH paper, assess its relevance to the student's research.
Respond with a JSON array (no markdown, no extra text) where each element has:
{{
  "index": <integer matching the [i] label>,
  "relevance_score": <integer 1-10>,
  "relevance_reason": "<one sentence on why it is or is not relevant>"
}}

Return ONLY the JSON array.
