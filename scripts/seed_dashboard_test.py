import os
import sys

from dotenv import load_dotenv

import db

# Run from the repo root: `python -m scripts.seed_dashboard_test`.
load_dotenv()


_PAPERS = [
	{
		'paperId': 'seed-attention',
		'title': 'Attention Is All You Need',
		'abstract': (
			'The dominant sequence transduction models are based on complex recurrent or convolutional neural '
			'networks that include an encoder and a decoder. We propose a new simple network architecture, the '
			'Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.'
		),
		'year': 2017,
		'citationCount': 110234,
		'url': 'https://arxiv.org/abs/1706.03762',
		'externalIds': {'DOI': '10.48550/arXiv.1706.03762'},
		'openAccessPdf': {'url': 'https://arxiv.org/pdf/1706.03762.pdf'},
		'authors': [{'name': 'Vaswani A.'}, {'name': 'Shazeer N.'}, {'name': 'Parmar N.'}, {'name': 'Uszkoreit J.'}],
		'summary': {
			'methodology': (
				'Introduces the Transformer architecture: encoder-decoder stack built from multi-head '
				'scaled-dot-product self-attention and position-wise feed-forward networks; positional '
				'encodings substitute for recurrence.'
			),
			'findings': (
				'New state-of-the-art on WMT-14 English-to-German (28.4 BLEU) and WMT-14 English-to-French '
				'(41.0 BLEU) with substantially shorter training time than prior best models.'
			),
			'relevance': (
				'Foundational for the transformer-track of the BPD work; every encoder used downstream '
				'(BERT, MentalBERT) descends from this architecture.'
			),
			'limitations': (
				'No clinical-domain evaluation. Quadratic attention cost limits direct application to '
				'long Reddit posts without hierarchical truncation.'
			),
		},
	},
	{
		'paperId': 'seed-bert',
		'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
		'abstract': (
			'We introduce a new language representation model called BERT, which stands for Bidirectional '
			'Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional '
			'representations from unlabeled text by jointly conditioning on both left and right context.'
		),
		'year': 2018,
		'citationCount': 92011,
		'url': 'https://arxiv.org/abs/1810.04805',
		'externalIds': {'DOI': '10.48550/arXiv.1810.04805'},
		'openAccessPdf': {'url': 'https://arxiv.org/pdf/1810.04805.pdf'},
		'authors': [{'name': 'Devlin J.'}, {'name': 'Chang M.-W.'}, {'name': 'Lee K.'}, {'name': 'Toutanova K.'}],
		'summary': {
			'methodology': (
				'Masked-language-modelling and next-sentence-prediction pre-training over BooksCorpus and '
				'English Wikipedia, followed by single-task fine-tuning on GLUE, SQuAD, and SWAG.'
			),
			'findings': (
				'Sets state-of-the-art on eleven NLP tasks including a 7.6 absolute improvement on GLUE '
				'and 5.1 absolute on SQuAD v1.1.'
			),
			'relevance': (
				'Direct backbone of MentalBERT used in the BPD word-level encoder. Pre-training objective '
				'transferable to clinical text with minimal architectural change.'
			),
			'limitations': (
				'General-domain pre-training underperforms on clinical text without domain adaptation. '
				'512-token cap forces hierarchical chunking for long Reddit posts.'
			),
		},
	},
	{
		'paperId': 'seed-mentalbert',
		'title': 'MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare',
		'abstract': (
			'We train and release a mental health domain-specific language model (MentalBERT) by continued '
			'pre-training on Reddit posts from mental health-related subreddits. The model is evaluated on '
			'several mental health detection tasks.'
		),
		'year': 2022,
		'citationCount': 312,
		'url': 'https://arxiv.org/abs/2110.15621',
		'externalIds': {'DOI': '10.48550/arXiv.2110.15621'},
		'openAccessPdf': {'url': 'https://arxiv.org/pdf/2110.15621.pdf'},
		'authors': [{'name': 'Ji S.'}, {'name': 'Zhang T.'}, {'name': 'Ansari L.'}],
		'summary': {
			'methodology': (
				'Continued pre-training of BERT-base on 13.7M Reddit posts from mental-health subreddits, '
				'evaluated on depression and stress detection benchmarks.'
			),
			'findings': (
				'Outperforms BERT-base and RoBERTa-base on every mental-health detection benchmark tested, '
				'with the largest gains on tasks where clinical vocabulary is dense.'
			),
			'relevance': (
				'Drop-in replacement for the BERT encoder in the HAN word-level pipeline; the BPD work '
				'uses this checkpoint directly.'
			),
			'limitations': (
				'No BPD-specific evaluation. Reddit pre-training corpus skews toward depression and anxiety '
				'discourse, so transfer to personality-disorder language is not guaranteed.'
			),
		},
	},
	{
		'paperId': 'seed-han',
		'title': 'Hierarchical Attention Networks for Document Classification',
		'abstract': (
			'We propose a hierarchical attention network for document classification. Our model has two '
			'distinctive characteristics: a hierarchical structure that mirrors the hierarchical structure '
			'of documents, and two levels of attention mechanisms applied at the word- and sentence-level.'
		),
		'year': 2016,
		'citationCount': 5421,
		'url': 'https://aclanthology.org/N16-1174/',
		'externalIds': {'DOI': '10.18653/v1/N16-1174'},
		'openAccessPdf': {'url': 'https://aclanthology.org/N16-1174.pdf'},
		'authors': [
			{'name': 'Yang Z.'},
			{'name': 'Yang D.'},
			{'name': 'Dyer C.'},
			{'name': 'He X.'},
			{'name': 'Smola A.'},
			{'name': 'Hovy E.'},
		],
		'summary': {
			'methodology': (
				'Two-level architecture: bidirectional GRU encoders at word and sentence level, each followed '
				'by an attention layer that produces a context vector via softmax over learned scores.'
			),
			'findings': (
				'Outperforms prior state-of-the-art on six document-classification benchmarks. Attention '
				'visualisations highlight informative words and sentences.'
			),
			'relevance': (
				'Direct architectural template for the BPD HAN extension; the work replaces BiGRUs with '
				'MentalBERT and a transformer sentence-level encoder.'
			),
			'limitations': (
				'BiGRU encoders are slow to train and underperform transformer alternatives on long-document '
				'tasks. Attention is not formally a faithful explanation.'
			),
		},
	},
	{
		'paperId': 'seed-lime',
		'title': '"Why Should I Trust You?": Explaining the Predictions of Any Classifier',
		'abstract': (
			'We propose LIME, a novel explanation technique that explains the predictions of any classifier '
			'in an interpretable and faithful manner, by learning an interpretable model locally around the '
			'prediction.'
		),
		'year': 2016,
		'citationCount': 18877,
		'url': 'https://arxiv.org/abs/1602.04938',
		'externalIds': {'DOI': '10.1145/2939672.2939778'},
		'openAccessPdf': {'url': 'https://arxiv.org/pdf/1602.04938.pdf'},
		'authors': [{'name': 'Ribeiro M. T.'}, {'name': 'Singh S.'}, {'name': 'Guestrin C.'}],
		'summary': {
			'methodology': (
				'Fits a sparse linear surrogate model locally around each prediction by sampling perturbed '
				'inputs and weighting them by proximity to the instance being explained.'
			),
			'findings': (
				'Produces explanations that improve human trust in model decisions across image and text '
				'classification tasks; surfaces spurious correlations in trained classifiers.'
			),
			'relevance': (
				'Candidate model-agnostic baseline for the explainability strand of the thesis; comparison '
				'against attention-based and SHAP-based explanations sits inside the planned evaluation matrix.'
			),
			'limitations': (
				'Local surrogate fidelity degrades when the underlying classifier is highly non-linear in '
				'the neighbourhood. Explanation stability sensitive to the kernel-width hyperparameter.'
			),
		},
	},
]

_SEED_RATINGS = (('seed-attention', 5), ('seed-bert', 4), ('seed-mentalbert', 5))


def seed():
	url = os.environ.get('DATABASE_URL_TEST')
	if not url:
		sys.exit('DATABASE_URL_TEST is not set; export it (or add to .env) before seeding.')

	print(f'Seeding test database with {len(_PAPERS)} example papers...')
	conn = db.connect(url)
	db.init_schema(conn)

	for paper in _PAPERS:
		db.upsert_paper(conn, paper)
		db.upsert_summary(conn, paper['paperId'], paper['summary'], 'test-model-v1')

	for paper_id, rating in _SEED_RATINGS:
		db.insert_rating(conn, paper_id, rating)

	print('Done. Run `DASHBOARD_USE_TEST_DB=1 streamlit run streamlit_app.py` to view them.')


if __name__ == '__main__':
	seed()
