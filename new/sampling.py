import torch
import numpy as np
import pandas as pd
import new.global_vars as global_vars
import new.train as train
import new.utils as utils
import gpHelper
import skHelper
from new.classes.instance import Instance


def _getAbductionNoise(args, objs, node, parents, factual_instance_obj_or_ts, structural_equation):

  # TODO (medpri): this function is called using both brute-force and grad-descent
  # approach (where the former/latter uses Instance/dict)
  if isinstance(factual_instance_obj_or_ts, Instance):
    factual_instance = factual_instance_obj_or_ts.dict()
  else:
    factual_instance = factual_instance_obj_or_ts

  # only applies for ANM models
  return factual_instance[node] - structural_equation(
    0,
    *[factual_instance[parent] for parent in parents],
  )


def _getCounterfactualTemplate(args, objs, factual_instance_obj_or_ts, action_set, recourse_type):

  # TODO (medpri): this function is called using both brute-force and grad-descent
  # approach (where the former/latter uses Instance/dict)
  if isinstance(factual_instance_obj_or_ts, Instance):
    factual_instance = factual_instance_obj_or_ts.dict()
  else:
    factual_instance = factual_instance_obj_or_ts

  counterfactual_template = dict.fromkeys(
    objs.dataset_obj.getInputAttributeNames(),
    np.NaN,
  )

  # get intervention and conditioning sets
  intervention_set = set(action_set.keys())

  # intersection_of_non_descendents_of_intervened_upon_variables
  conditioning_set = set.intersection(*[
    objs.scm_obj.getNonDescendentsForNode(node)
    for node in intervention_set
  ])

  # assert there is no intersection
  assert set.intersection(intervention_set, conditioning_set) == set()

  # set values in intervention and conditioning sets
  for node in conditioning_set:
    counterfactual_template[node] = factual_instance[node]

  for node in intervention_set:
    counterfactual_template[node] = action_set[node]

  return counterfactual_template



def sampleTrue(args, objs, factual_instance_obj, factual_df, samples_df, node, parents, recourse_type):
  # Step 1. [abduction]: compute noise or load from dataset using factual_instance_obj
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  structural_equation = objs.scm_obj.structural_equations_np[node]

  if recourse_type == 'm0_true':

    noise_pred = _getAbductionNoise(args, objs, node, parents, factual_instance_obj, structural_equation)

    try: # may fail if noise variables are not present in the data (e.g., for real-world adult, we have no noise variables)
      noise_true = factual_instance_obj.dict('endogenous_and_exogenous')[f'u{node[1:]}']

      if args.scm_class != 'sanity-3-gen':
        assert np.abs(noise_pred - noise_true) < 1e-5, 'Noise {pred, true} expected to be similar, but not.'
    except:
      pass
    noise = noise_pred

    samples_df[node] = structural_equation(
      np.array(noise), # may be scalar, which will be case as pd.series when being summed.
      *[samples_df[parent] for parent in parents],
    )

  elif recourse_type == 'm2_true':

    samples_df[node] = structural_equation(
      np.array(objs.scm_obj.noises_distributions[utils.getNoiseStringForNode(node)].sample(samples_df.shape[0])),
      *[samples_df[parent] for parent in parents],
    )

  return samples_df


def sampleRidgeKernelRidge(args, objs, factual_instance_obj, factual_df, samples_df, node, parents, recourse_type):
  samples_df = utils.processDataFrameOrInstance(args, objs, samples_df.copy(), global_vars.PROCESSING_SKLEARN)
  factual_instance_obj = utils.processDataFrameOrInstance(args, objs, factual_instance_obj, global_vars.PROCESSING_SKLEARN)

  # Step 1. [abduction]: compute noise or load from dataset using factual_instance_obj
  # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_df columns
  # Step 3. [prediction]: run through structural equation using noise and parents from samples_df
  if recourse_type == 'm1_alin':
    trained_model = train.trainRidge(args, objs, node, parents)
  elif recourse_type == 'm1_akrr':
    trained_model = train.trainKernelRidge(args, objs, node, parents)
  else:
    raise Exception(f'{recourse_type} not recognized.')
  structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
  for row_idx, row in samples_df.iterrows():
    noise = _getAbductionNoise(args, objs, node, parents, factual_instance_obj, structural_equation)
    samples_df.loc[row_idx, node] = structural_equation(
      noise,
      *samples_df.loc[row_idx, parents].to_numpy(),
    )
  samples_df = utils.deprocessDataFrameOrInstance(args, objs, samples_df, global_vars.PROCESSING_SKLEARN)
  return samples_df


def sampleCVAE(args, objs, factual_instance_obj, factual_df, samples_df, node, parents, recourse_type, trained_cvae = None):
  samples_df = utils.processDataFrameOrInstance(args, objs, samples_df.copy(), global_vars.PROCESSING_CVAE)
  factual_instance_obj = utils.processDataFrameOrInstance(args, objs, factual_instance_obj, global_vars.PROCESSING_CVAE)

  if trained_cvae is None: # pseudonym: UGLY CODE
    trained_cvae = train.trainCVAE(args, objs, node, parents)

  if recourse_type == 'm1_cvae':
    sample_from = 'posterior'
  elif recourse_type == 'm2_cvae':
    sample_from = 'prior'
  elif recourse_type == 'm2_cvae_ps':
    sample_from = 'reweighted_prior'

  x_factual = factual_df[[node]]
  pa_factual = factual_df[parents]
  pa_counter = samples_df[parents]

  attr_obj = objs.dataset_obj.attributes_kurz[node]
  if attr_obj.attr_type in {'categorical', 'ordinal'}:
    # non-hot --> one-hot
    x_factual = utils.convertToOneHotWithPrespecifiedCategories(x_factual, node, attr_obj.lower_bound, attr_obj.upper_bound)

  new_samples = trained_cvae.reconstruct(
    x_factual=x_factual,
    pa_factual=pa_factual,
    pa_counter=pa_counter,
    sample_from=sample_from,
  )

  if attr_obj.attr_type in {'categorical', 'ordinal'}:
    # one-hot --> non-hot
    new_samples = pd.DataFrame(new_samples.idxmax(axis=1) + 1)

  new_samples = new_samples.rename(columns={0: node}) # bad code pseudonym, this violates abstraction!
  samples_df = samples_df.reset_index(drop=True)
  samples_df[node] = new_samples.astype('float64')
  samples_df = utils.deprocessDataFrameOrInstance(args, objs, samples_df, global_vars.PROCESSING_CVAE)
  return samples_df


def sampleGP(args, objs, factual_instance_obj, factual_df, samples_df, node, parents, recourse_type):
  samples_df = utils.processDataFrameOrInstance(args, objs, samples_df.copy(), global_vars.PROCESSING_GAUS)
  factual_instance_obj = utils.processDataFrameOrInstance(args, objs, factual_instance_obj, global_vars.PROCESSING_GAUS)

  kernel, X_all, model = train.trainGP(args, objs, node, parents)
  X_parents = torch.tensor(samples_df[parents].to_numpy())

  if recourse_type == 'm1_gaus': # counterfactual distribution for node
    # IMPORTANT: Find index of factual instance in dataframe used for training GP
    #            (earlier, the factual instance was appended as the last instance)
    tmp_idx = utils.getIndexOfFactualInstanceInDataFrame(
      factual_instance_obj,
      utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), global_vars.PROCESSING_GAUS),
    ) # TODO (lowpri): can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
    new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
  elif recourse_type == 'm2_gaus': # interventional distribution for node
    new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

  samples_df[node] = new_samples.numpy()
  samples_df = utils.deprocessDataFrameOrInstance(args, objs, samples_df, global_vars.PROCESSING_GAUS)
  return samples_df


def _sampleInnerLoop(args, objs, factual_instance_obj, action_set, recourse_type, num_samples):

  counterfactual_template = _getCounterfactualTemplate(args, objs, factual_instance_obj, action_set, recourse_type)

  factual_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [factual_instance_obj.dict()[node]] for node in objs.dataset_obj.getInputAttributeNames()],
  )))
  # this dataframe has populated columns set to intervention or conditioning values
  # and has NaN columns that will be set accordingly.
  samples_df = pd.DataFrame(dict(zip(
    objs.dataset_obj.getInputAttributeNames(),
    [num_samples * [counterfactual_template[node]] for node in objs.dataset_obj.getInputAttributeNames()],
  )))

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
  for node in objs.scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if samples_df[node].isnull().values.any():
      parents = objs.scm_obj.getParentsForNode(node)
      # root nodes MUST always be set through intervention or conditioning
      assert len(parents) > 0
      # Confirm parents columns are present/have assigned values in samples_df
      assert not samples_df.loc[:,list(parents)].isnull().values.any()
      if args.debug_flag:
        print(f'Sampling `{recourse_type}` from {utils.getConditionalString(node, parents)}')
      if recourse_type in {'m0_true', 'm2_true'}:
        sampling_handle = sampleTrue
      elif recourse_type in {'m1_alin', 'm1_akrr'}:
        sampling_handle = sampleRidgeKernelRidge
      elif recourse_type in {'m1_gaus', 'm2_gaus'}:
        sampling_handle = sampleGP
      elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:
        sampling_handle = sampleCVAE
      else:
        raise Exception(f'{recourse_type} not recognized.')
      samples_df = sampling_handle(
        args,
        objs,
        factual_instance_obj,
        factual_df,
        samples_df,
        node,
        parents,
        recourse_type,
      )
  assert \
    np.all(list(samples_df.columns) == objs.dataset_obj.getInputAttributeNames()), \
    'Ordering of column names in samples_df has change unexpectedly'
  # samples_df = samples_df[objs.dataset_obj.getInputAttributeNames()]
  return samples_df


def getColumnIndicesFromNames(args, objs, column_names):
  # this is index in df, need to -1 to get index in x_counter / do_update,
  # because the first column of df is 'y' (pseudonym: what if column ordering is
  # changed? this code breaks abstraction.)
  column_indices = []
  for column_name in column_names:
    tmp_1 = objs.dataset_obj.data_frame_kurz.columns.get_loc(column_name) - 1
    tmp_2 = list(objs.scm_obj.getTopologicalOrdering()).index(column_name)
    tmp_3 = list(objs.dataset_obj.getInputAttributeNames()).index(column_name)
    assert tmp_1 == tmp_2 == tmp_3
    column_indices.append(tmp_1)
  return column_indices


def _sampleInnerLoopTensor(args, objs, factual_instance_obj, factual_instance_ts, action_set_ts, recourse_type):

  counterfactual_template_ts = _getCounterfactualTemplate(args, objs, factual_instance_ts, action_set_ts, recourse_type)

  if recourse_type in global_vars.ACCEPTABLE_POINT_RECOURSE:
    num_samples = 1
  if recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE:
    num_samples = args.num_mc_samples

  # Initialize factual_ts, samples_ts
  factual_ts = torch.zeros((num_samples, len(objs.dataset_obj.getInputAttributeNames())))
  samples_ts = torch.zeros((num_samples, len(objs.dataset_obj.getInputAttributeNames())))
  for node in objs.scm_obj.getTopologicalOrdering():
    factual_ts[:, getColumnIndicesFromNames(args, objs, [node])] = factual_instance_ts[node] + 0 # + 0 not needed because not trainable but leaving in..
    # +0 important, specifically for tensor based elements, so we don't copy
    # an existing object in the computational graph, but we create a new node
    samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = counterfactual_template_ts[node] + 0

  # Simply traverse the graph in order, and populate nodes as we go!
  # IMPORTANT: DO NOT use SET(topo ordering); it sometimes changes ordering!
  for node in objs.scm_obj.getTopologicalOrdering():
    # set variable if value not yet set through intervention or conditioning
    if torch.any(torch.isnan(samples_ts[:, getColumnIndicesFromNames(args, objs, [node])])):
      parents = objs.scm_obj.getParentsForNode(node)
      # root nodes MUST always be set through intervention or conditioning
      assert len(parents) > 0
      # Confirm parents columns are present/have assigned values in samples_ts
      assert not torch.any(torch.isnan(samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]))

      if recourse_type in {'m0_true', 'm2_true'}:

        structural_equation = objs.scm_obj.structural_equations_ts[node]

        if recourse_type == 'm0_true':
          # may be scalar, which will be case as pd.series when being summed.
          noise_pred = _getAbductionNoise(args, objs, node, parents, factual_instance_ts, structural_equation)
          noises = noise_pred
        elif recourse_type == 'm2_true':
          noises = torch.tensor(
            objs.scm_obj.noises_distributions[utils.getNoiseStringForNode(node)].sample(samples_ts.shape[0])
          ).reshape(-1,1)

        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = structural_equation(
          noises,
          *[samples_ts[:, getColumnIndicesFromNames(args, objs, [parent])] for parent in parents],
        )

      elif recourse_type in {'m1_alin', 'm1_akrr'}:

        if recourse_type == 'm1_alin':
          training_handle = train.trainRidge
          sampling_handle = skHelper.sample_from_LIN_model
        elif recourse_type == 'm1_akrr':
          training_handle = train.trainKernelRidge
          sampling_handle = skHelper.sample_from_KRR_model

        trained_model = training_handle(args, objs, node, parents).best_estimator_
        X_parents = samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]

        # Step 1. [abduction]
        # TODO (lowpri): we don't need structural_equation here... get the noise posterior some other way.
        structural_equation = lambda noise, *parents_values: trained_model.predict([[*parents_values]])[0][0] + noise
        noise = _getAbductionNoise(args, objs, node, parents, factual_instance_ts, structural_equation)

        # Step 2. [action]: (skip) this step is implicitly performed in the populated samples_ts columns
        # N/A

        # Step 3. [prediction]: first get the regressed value, then get noise
        new_samples = sampling_handle(trained_model, X_parents)
        assert np.isclose( # a simple check to make sure manual sklearn is working correct
          new_samples.item(),
          trained_model.predict(X_parents.detach().numpy()).item(),
          atol = 1e-3,
        )
        new_samples = new_samples + noise

        # add back to dataframe
        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO (lowpri): not sure if +0 is needed or not

      elif recourse_type in {'m1_gaus', 'm2_gaus'}:

        kernel, X_all, model = train.trainGP(args, objs, node, parents)
        X_parents = samples_ts[:, getColumnIndicesFromNames(args, objs, parents)]

        if recourse_type == 'm1_gaus': # counterfactual distribution for node
          # IMPORTANT: Find index of factual instance in dataframe used for training GP
          #            (earlier, the factual instance was appended as the last instance)
          # DO NOT DO THIS: conversion from float64 to torch and back will make it impossible to find the instance idx
          # factual_instance_obj = {k:v.item() for k,v in factual_instance_ts.items()}
          tmp_idx = utils.getIndexOfFactualInstanceInDataFrame( # TODO (lowpri): write this as ts function as well?
            factual_instance_obj,
            utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples), global_vars.PROCESSING_GAUS),
          ) # TODO (lowpri): can probably rewrite to just evaluate the posterior again given the same result.. (without needing to look through the dataset)
          new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'cf', tmp_idx)
        elif recourse_type == 'm2_gaus': # interventional distribution for node
          new_samples = gpHelper.sample_from_GP_model(model, X_parents, 'iv')

        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO (lowpri): not sure if +0 is needed or not

      elif recourse_type in {'m1_cvae', 'm2_cvae', 'm2_cvae_ps'}:

        if recourse_type == 'm1_cvae':
          sample_from = 'posterior'
        elif recourse_type == 'm2_cvae':
          sample_from = 'prior'
        elif recourse_type == 'm2_cvae_ps':
          sample_from = 'reweighted_prior'

        trained_cvae = train.trainCVAE(args, objs, node, parents)
        new_samples = trained_cvae.reconstruct(
          x_factual=factual_ts[:, getColumnIndicesFromNames(args, objs, [node])],
          pa_factual=factual_ts[:, getColumnIndicesFromNames(args, objs, parents)],
          pa_counter=samples_ts[:, getColumnIndicesFromNames(args, objs, parents)],
          sample_from=sample_from,
        )
        samples_ts[:, getColumnIndicesFromNames(args, objs, [node])] = new_samples + 0 # TODO (lowpri): not sure if +0 is needed or not

  return samples_ts