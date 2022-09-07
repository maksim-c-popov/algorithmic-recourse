ACCEPTABLE_POINT_RECOURSE = {'m0_true', 'm1_alin', 'm1_akrr'}
ACCEPTABLE_DISTR_RECOURSE = {'m1_gaus', 'm1_cvae', 'm2_true', 'm2_gaus', 'm2_cvae', 'm2_cvae_ps'}

EXPERIMENTAL_SETUPS = [
    ('m0_true', '*'), \
    ('m1_alin', 'v'), \
    ('m1_akrr', '^'), \
    ('m1_gaus', 'D'), \
    ('m1_cvae', 'x'), \
    ('m2_true', 'o'), \
    ('m2_gaus', 's'), \
    ('m2_cvae', '+'), \
  ]

PROCESSING_SKLEARN = 'raw'
PROCESSING_GAUS = 'raw'
PROCESSING_CVAE = 'raw'

FAIR_MODELS = {
	'vanilla_svm',
	'vanilla_lr',
	'vanilla_mlp',
	'nonsens_svm',
	'nonsens_lr',
	'nonsens_mlp',
	'unaware_svm',
	'unaware_lr',
	'unaware_mlp',
	'cw_fair_svm',
	'cw_fair_lr',
	'cw_fair_mlp',
	'iw_fair_svm'}