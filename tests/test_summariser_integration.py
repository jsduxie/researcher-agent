import os

import pytest

import fetcher
import summariser
from summariser import (
	MISSING_FIELD_PLACEHOLDER,
	download_pdf,
	generate_with_file,
	parse_summary_response,
	summarise_paper,
	upload_pdf_to_gemini,
)

_GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Mirror test_db_integration.py: empty string covers GH Actions' fork-PR secret behaviour.
pytestmark = [
	pytest.mark.integration,
	pytest.mark.skipif(
		not _GEMINI_API_KEY,
		reason='GEMINI_API_KEY not set; live integration tests against Semantic Scholar and Gemini require a key',
	),
]


# A stable open-access paper with a well-known arXiv URL. Used by the live
# download and full-chain tests. arXiv PDF URLs do not change once published.
_ARXIV_PDF_URL = 'https://arxiv.org/pdf/1706.03762.pdf'
_ARXIV_PAPER_ID = 'attention-is-all-you-need'


def _build_minimal_pdf(body_text):
	"""Hand-roll a syntactically valid PDF with one line of text.

	Avoids committing a binary fixture file and avoids a reportlab dependency.
	The xref byte offsets are computed dynamically so the PDF parses cleanly
	regardless of small string-length changes to body_text.
	"""
	stream_payload = f'BT /F1 14 Tf 50 700 Td ({body_text}) Tj ET\n'.encode()
	objects = [
		b'<</Type/Catalog/Pages 2 0 R>>',
		b'<</Type/Pages/Kids[3 0 R]/Count 1>>',
		(
			b'<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R'
			b'/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>>'
		),
		b'<</Length ' + str(len(stream_payload)).encode() + b'>>stream\n' + stream_payload + b'endstream',
	]
	out = bytearray(b'%PDF-1.4\n')
	offsets = []
	for i, obj in enumerate(objects, start=1):
		offsets.append(len(out))
		out += f'{i} 0 obj'.encode() + obj + b'\nendobj\n'
	xref_offset = len(out)
	out += b'xref\n'
	out += f'0 {len(objects) + 1}\n'.encode()
	out += b'0000000000 65535 f \n'
	for off in offsets:
		out += f'{off:010d} 00000 n \n'.encode()
	out += b'trailer\n'
	out += f'<</Size {len(objects) + 1}/Root 1 0 R>>\n'.encode()
	out += b'startxref\n'
	out += f'{xref_offset}\n'.encode()
	out += b'%%EOF\n'
	return bytes(out)


# -- Semantic Scholar live --


def test_fetch_papers_returns_results_from_semantic_scholar():
	# A query broad enough to keep returning hits even as the index updates.
	results = fetcher.fetch_papers('transformer attention')
	assert isinstance(results, list)
	assert len(results) > 0
	# Confirm the live shape still matches what fetcher.PAPER_FIELDS asked for.
	first = results[0]
	assert 'paperId' in first
	assert 'title' in first


# -- Gemini Files API live --


def test_download_pdf_against_live_arxiv_url():
	max_bytes = summariser.PDF_MAX_SIZE_MB * 1024 * 1024
	pdf_bytes = download_pdf(_ARXIV_PDF_URL, max_bytes)
	assert pdf_bytes.startswith(b'%PDF')
	assert len(pdf_bytes) > 1000


def test_upload_and_generate_against_gemini_files_api():
	# Round-trips a tiny synthetic PDF through the Files API: upload, then
	# generate. We don't care what Gemini returns about the content, only
	# that the two-step protocol completes and yields non-empty text.
	pdf_bytes = _build_minimal_pdf('This paper proposes a transformer for BPD detection.')
	file_uri = upload_pdf_to_gemini(pdf_bytes, 'integration-test.pdf', _GEMINI_API_KEY)
	assert file_uri
	response = generate_with_file('Reply with the single word ok.', file_uri, _GEMINI_API_KEY)
	assert response.strip()


# -- Full chain: live PDF + live Gemini --


def test_summarise_paper_full_pdf_chain_returns_four_populated_fields():
	# Constructed paper dict mimicking Semantic Scholar's shape, with a real
	# arXiv PDF. Tests the full PDF path inside summarise_paper end to end.
	paper = {'paperId': _ARXIV_PAPER_ID, 'title': 'Attention Is All You Need', 'openAccessPdf': {'url': _ARXIV_PDF_URL}}
	result = summarise_paper(paper, gemini_fn=None, conn=None, api_key=_GEMINI_API_KEY)

	assert result is not None
	for field in ('methodology', 'findings', 'relevance', 'limitations'):
		assert field in result
		assert isinstance(result[field], str)
		assert result[field].strip()
	# At least one field should be a real summary, not just the missing-data placeholder.
	# (A full PDF of this paper produces all four; this is a soft check that survives
	# transient Gemini quirks.)
	assert any(result[f] != MISSING_FIELD_PLACEHOLDER for f in ('methodology', 'findings', 'relevance', 'limitations'))


# -- parse_summary_response sanity against a live PDF --


def test_summariser_response_parses_when_run_against_a_real_pdf():
	# Confirms that the live Gemini response with a real PDF parses cleanly
	# through the same code path the production pipeline uses.
	pdf_bytes = download_pdf(_ARXIV_PDF_URL, summariser.PDF_MAX_SIZE_MB * 1024 * 1024)
	file_uri = upload_pdf_to_gemini(pdf_bytes, 'attention.pdf', _GEMINI_API_KEY)
	prompt = summariser.SUMMARISER_PROMPT.format(
		research_context='Transformer architectures for text classification.', source_material='See the attached PDF.'
	)
	response = generate_with_file(prompt, file_uri, _GEMINI_API_KEY)
	fields = parse_summary_response(response)
	assert set(fields.keys()) == {'methodology', 'findings', 'relevance', 'limitations'}
