import itertools
import torch
import random
import numpy as np
import pandas as pd
import GPy

from attrdict import AttrDict
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

from new._cvae.train import *
import new.utils as utils
import new.global_vars as global_vars
import new.sampling as sampling
from new.classes.memoize import Memoize
import new.mmd as mmd

@Memoize
def trainRidge(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {utils.getConditionalString(node, parents)} using Ridge on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), global_vars.PROCESSING_SKLEARN)
  param_grid = {'alpha': np.logspace(-2, 1, 10)}
  model = GridSearchCV(Ridge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
  return model


@Memoize
def trainKernelRidge(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {utils.getConditionalString(node, parents)} using KernelRidge on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), global_vars.PROCESSING_SKLEARN)
  param_grid = {
    'alpha': np.logspace(-2, 1, 5),
    'kernel': [
      RBF(lengthscale)
      for lengthscale in np.logspace(-2, 1, 5)
    ]
  }
  model = GridSearchCV(KernelRidge(), param_grid=param_grid)
  model.fit(X_all[parents], X_all[[node]])
   # pseudonym: i am not proud of this, but for some reason sklearn includes the
   # X_fit_ covariates but not labels (this is needed later if we want to
   # avoid using.predict() and call from krr manually)
  model.best_estimator_.Y_fit_ = X_all[[node]].to_numpy()
  return model


@Memoize
def trainCVAE(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {utils.getConditionalString(node, parents)} using CVAE on {args.num_train_samples * 4} samples; this may be very expensive, memoizing afterwards.')
  X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples * 4 + args.num_validation_samples), global_vars.PROCESSING_CVAE)

  attr_obj = objs.dataset_obj.attributes_kurz[node]
  node_train = X_all[[node]].iloc[:args.num_train_samples * 4]
  parents_train = X_all[parents].iloc[:args.num_train_samples * 4]
  node_validation = X_all[[node]].iloc[args.num_train_samples * 4:]
  parents_validation = X_all[parents].iloc[args.num_train_samples * 4:]


  if args.scm_class == 'sanity-3-lin':
    if node == 'x2':
      sweep_lambda_kld = [0.01]
      sweep_encoder_layer_sizes = [[1, 32, 32, 32]]
      sweep_decoder_layer_sizes = [[5, 5, 1]]
      sweep_latent_size = [1]
    elif node == 'x3':
      sweep_lambda_kld = [0.01]
      sweep_encoder_layer_sizes = [[1, 32, 32, 32]]
      sweep_decoder_layer_sizes = [[32, 32, 32, 1]]
      sweep_latent_size = [1]

  elif args.scm_class == 'sanity-3-anm':
    if node == 'x2':
      sweep_lambda_kld = [0.01]
      sweep_encoder_layer_sizes = [[1, 32, 32]]
      sweep_decoder_layer_sizes = [[32, 32, 1]]
      sweep_latent_size = [5]
    elif node == 'x3':
      sweep_lambda_kld = [0.01]
      sweep_encoder_layer_sizes = [[1, 32, 32, 32]]
      sweep_decoder_layer_sizes = [[32, 32, 1]]
      sweep_latent_size = [1]

  elif args.scm_class == 'sanity-3-gen':
    if node == 'x2':
      sweep_lambda_kld = [0.5]
      sweep_encoder_layer_sizes = [[1, 32, 32, 32]]
      sweep_decoder_layer_sizes = [[32, 32, 1]]
      sweep_latent_size = [3]
    elif node == 'x3':
      sweep_lambda_kld = [0.5]
      sweep_encoder_layer_sizes = [[1, 32, 32, 32]]
      sweep_decoder_layer_sizes = [[32, 32, 1]]
      sweep_latent_size = [3]

  else:

    io_size = 1 # b/c X_all[[node]] is always 1 dimensional for non-cat/ord variables

    if attr_obj.attr_type in {'categorical', 'ordinal'}:

      # one-hot --> non-hot (categorical CVAE can guarantee conditional p(node|parents) returns categorical value)
      # IMPORTANT: we DO NOT convert cat/ord parents to one-hot.
      # IMPORTANT: because all categories may not be present in the dataframe,
      #            we call pd.get_dummies on an already converted dataframe with
      #            with prespecfied categories
      node_train = utils.convertToOneHotWithPrespecifiedCategories(node_train, node, attr_obj.lower_bound, attr_obj.upper_bound)
      node_validation = utils.convertToOneHotWithPrespecifiedCategories(node_validation, node, attr_obj.lower_bound, attr_obj.upper_bound)
      io_size = len(node_train.columns)
      assert \
        len(node_train.columns) == len(node_validation.columns), \
        'Training data is not representative enough; missing some categories'

    sweep_lambda_kld = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005]
    sweep_encoder_layer_sizes = [
      [io_size, 2, 2],
      [io_size, 3, 3],
      [io_size, 5, 5],
      [io_size, 32, 32, 32],
    ]
    sweep_decoder_layer_sizes = [
      [2, io_size],
      [2, 2, io_size],
      [3, 3, io_size],
      [5, 5, io_size],
      [32, 32, 32, io_size],
    ]
    sweep_latent_size = [1,3,5]

  trained_models = {}

  all_hyperparam_setups = list(itertools.product(
    sweep_lambda_kld,
    sweep_encoder_layer_sizes,
    sweep_decoder_layer_sizes,
    sweep_latent_size,
  ))

  if args.scm_class not in {'sanity-3-lin', 'sanity-3-anm', 'sanity-3-gen'}:
    # # TODO (remove): For testing purposes only
    all_hyperparam_setups = random.sample(all_hyperparam_setups, 10)

  for idx, hyperparams in enumerate(all_hyperparam_setups):

    print(f'\n\t[INFO] Training hyperparams setup #{idx+1} / {len(all_hyperparam_setups)}: {str(hyperparams)}')

    try:
      trained_cvae, recon_node_train, recon_node_validation = train_cvae(AttrDict({
        'name': f'{utils.getConditionalString(node, parents)}',
        'attr_type': attr_obj.attr_type,
        'node_train': node_train,
        'parents_train': parents_train,
        'node_validation': node_validation,
        'parents_validation': parents_validation,
        'seed': 0,
        'epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.05,
        'lambda_kld': hyperparams[0],
        'encoder_layer_sizes': hyperparams[1],
        'decoder_layer_sizes': hyperparams[2],
        'latent_size': hyperparams[3],
        'conditional': True,
        'debug_folder': args.experiment_folder_name + f'/cvae_hyperparams_setup_{idx}_of_{len(all_hyperparam_setups)}',
      }))
    except Exception as e:
      print(e)
      continue


    if attr_obj.attr_type in {'categorical', 'ordinal'}:
      # non-hot --> one-hot
      recon_node_train = torch.argmax(recon_node_train, axis=1, keepdims=True) + 1 # categoricals start at index 1
      recon_node_validation = torch.argmax(recon_node_validation, axis=1, keepdims=True) + 1 # categoricals start at index 1

    # # TODO (lowpri): remove after models.py is corrected
    # return trained_cvae

    # run mmd to verify whether training is good or not (ON VALIDATION SET)
    X_val = X_all[args.num_train_samples * 4:].copy()
    # POTENTIAL BUG? reset index here so that we can populate the `node` column
    # with reconstructed values from trained_cvae that lack indexing
    X_val = X_val.reset_index(drop = True)

    X_true = X_val[parents + [node]]

    X_pred_posterior = X_true.copy()
    X_pred_posterior[node] = pd.DataFrame(recon_node_validation.numpy(), columns=[node])

    # pseudonym: this is so bad code.
    not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
    not_imp_factual_df = pd.DataFrame(dict(zip(
      objs.dataset_obj.getInputAttributeNames(),
      [X_true.shape[0] * [not_imp_factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
    )))
    not_imp_samples_df = X_true.copy()
    X_pred_prior = sampling.sampleCVAE(args, objs, not_imp_factual_instance, not_imp_factual_df, not_imp_samples_df, node, parents, 'm2_cvae', trained_cvae = trained_cvae)

    X_pred = X_pred_prior

    my_statistic, statistics, sigma_median = mmd.mmd_with_median_heuristic(X_true.to_numpy(), X_pred.to_numpy())
    print(f'\t\t[INFO] test-statistic = {my_statistic} using median of {sigma_median} as bandwith')

    trained_models[f'setup_{idx:03d}'] = {}
    trained_models[f'setup_{idx:03d}']['hyperparams'] = hyperparams
    trained_models[f'setup_{idx:03d}']['trained_cvae'] = trained_cvae
    trained_models[f'setup_{idx:03d}']['test-statistic'] = my_statistic

  index_with_lowest_test_statistics = min(trained_models.keys(), key=lambda k: abs(trained_models[k]['test-statistic'] - 0))
  # index_with_lowest_test_statistics = min(trained_models.keys(), key=lambda k: trained_models[k]['test-statistic'])
  model_with_lowest_test_statistics = trained_models[index_with_lowest_test_statistics]['trained_cvae']
  # save all results
  tmp_file_name = f'{args.experiment_folder_name}/_cvae_params_{utils.getConditionalString(node, parents)}.txt'
  pprint(trained_models[index_with_lowest_test_statistics]['hyperparams'], open(tmp_file_name, 'w'))
  pprint(trained_models, open(tmp_file_name, 'a'))
  return model_with_lowest_test_statistics


@Memoize
def trainGP(args, objs, node, parents):
  assert len(parents) > 0, 'parents set cannot be empty.'
  print(f'\t[INFO] Fitting {utils.getConditionalString(node, parents)} using GP on {args.num_train_samples} samples; this may be very expensive, memoizing afterwards.')
  X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), global_vars.PROCESSING_GAUS)

  kernel = GPy.kern.RBF(input_dim=len(parents), ARD=True)
  # IMPORTANT: do NOT use DataFrames, use Numpy arrays; GPy doesn't like DF.
  # https://github.com/SheffieldML/GPy/issues/781#issuecomment-532738155
  model = GPy.models.GPRegression(
    X_all[parents].to_numpy(),
    X_all[[node]].to_numpy(),
    kernel,
  )
  model.optimize_restarts(parallel=True, num_restarts=5, verbose=False)

  return kernel, X_all, model