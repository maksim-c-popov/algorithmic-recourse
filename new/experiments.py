import imp
import os
import time
import pickle
import numpy as np
import pandas as pd
import warnings
import itertools
import random
import seaborn as sns

from functools import partial
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from pprint import pprint
from matplotlib import pyplot as plt

import new.global_vars as global_vars
import new.utils as utils
import new.sampling as sampling
import new.action_set_processing as action_set_processing
from new.classes.instance import Instance
from scatter import *


def prettyPrintDict(my_dict):
  # use this for grad descent logs (convert tensor accordingly)
  my_dict = my_dict.copy()
  for key, value in my_dict.items():
    my_dict[key] = np.around(value, 3)
  return my_dict


def evaluateKernelForFairSVM(classifier, *params):
  # similar to the kernel() method of RecourseSVM (third_party code)
  if (classifier.kernel == 'linear'):
    return linear_kernel(*params)
  elif (classifier.kernel == 'rbf'):
    return partial(rbf_kernel, gamma=classifier.gamma)(*params)
  elif (classifier.kernel == 'poly'):
    return partial(polynomial_kernel, degree=classifier.degree)(*params)


def measureDistanceToDecisionBoundary(args, objs, factual_instance_obj):
  if args.classifier_class not in global_vars.FAIR_MODELS:
    # raise NotImplementedError
    print(f'[WARNING] computing dist to decision boundary in closed-form for `{args.classifier_class}` model is not supported.')
    return -1

  # keep track of the factual_instance_obj and it's exogenous variables.
  factual_instance_dict = factual_instance_obj.dict('endogenous_and_exogenous')

  # then select only those keys that are used as input to the fair model
  fair_nodes = utils.getTrainableNodesForFairModel(args, objs)
  factual_instance_dict = dict(zip(
    fair_nodes,
    [factual_instance_obj.dict('endogenous_and_exogenous')[key] for key in fair_nodes]
  ))
  factual_instance_array = np.expand_dims(np.array(list(factual_instance_dict.values())), axis=0)

  if 'lr' in args.classifier_class:
    # Implementation #1 (source: https://stackoverflow.com/a/32077408/2759976)
    # source: https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/
    # y = objs.classifier_obj.decision_function(factual_instance)
    # w_norm = np.linalg.norm(objs.classifier_obj.coef_)
    # distance_to_decision_boundary = y / w_norm

    # Implementation #2 (source: https://math.stackexchange.com/a/1210685/641466)
    distance_to_decision_boundary = (
      np.dot(
        objs.classifier_obj.coef_,
        factual_instance_array.T
      ) + objs.classifier_obj.intercept_
    ) / np.linalg.norm(objs.classifier_obj.coef_)
    distance_to_decision_boundary = distance_to_decision_boundary[0]

  elif 'mlp' in args.classifier_class:
    # feed instance forward until penultimate layer, then get inner product of
    # the instance embedding with the (linear) features of the last layer, just
    # as was done in 'lr' above.

    # source: https://github.com/amirhk/mace/blob/master/modelConversion.py#L289
    def getPenultimateEmbedding(model, x):
      layer_output = x
      for layer_idx in range(len(model.coefs_) - 1):
        #
        layer_input_size = len(model.coefs_[layer_idx])
        if layer_idx != len(model.coefs_) - 1:
          layer_output_size = len(model.coefs_[layer_idx + 1])
        else:
          layer_output_size = model.n_outputs_
        #
        layer_input = layer_output
        layer_output = [0 for j in range(layer_output_size)]
        # i: indices of nodes in layer L
        # j: indices of nodes in layer L + 1
        for j in range(layer_output_size):
          score = model.intercepts_[layer_idx][j]
          for i in range(layer_input_size):
            score += layer_input[i] * model.coefs_[layer_idx][i][j]
          if score > 0: # relu operator
            layer_output[j] = score
          else:
            layer_output[j] = 0
      # no need for final layer output
      # if layer_output[0] > 0:
      #   return 1
      # return 0
      return layer_output

    penultimate_embedding = getPenultimateEmbedding(objs.classifier_obj, factual_instance_array[0])

    distance_to_decision_boundary = (
      np.dot(
        objs.classifier_obj.coefs_[-1].T,
        np.array(penultimate_embedding)
      ) + objs.classifier_obj.intercepts_[-1]
    ) / np.linalg.norm(objs.classifier_obj.coefs_[-1])

  elif 'svm' in args.classifier_class:
    # For non-linear kernels, the weight vector of the SVM hyperplane is not available,
    # in fact for the 'rbf' kernel it is infinite dimensional.
    # However, its norm in the RKHS can be computed in closed form in terms of the kernel matrix evaluated
    # at the support vectors and the dual coefficients. For more info, see, e.g.,
    # https://stats.stackexchange.com/questions/14876/interpreting-distance-from-hyperplane-in-svm
    try:
      # This should work for all normal instances of SVC except for RecourseSVM (third_party code)
      dual_coefficients = objs.classifier_obj.dual_coef_
      support_vectors = objs.classifier_obj.support_vectors_
      kernel_matrix_for_support_vectors = evaluateKernelForFairSVM(objs.classifier_obj, support_vectors)
      squared_norm_of_weight_vector = np.einsum('ij, jk, lk', dual_coefficients, kernel_matrix_for_support_vectors, dual_coefficients)
      norm_of_weight_vector = np.sqrt(squared_norm_of_weight_vector.flatten())
      distance_to_decision_boundary = objs.classifier_obj.decision_function(factual_instance_array)/norm_of_weight_vector
    except:
      # For RecourseSVM (third_party code) normalisation by the norm of the weight vector is hardcoded into
      # .decision_function so that the output is already an absolute distance.
      distance_to_decision_boundary = objs.classifier_obj.decision_function(factual_instance_array)
  else:
    raise NotImplementedError

  return distance_to_decision_boundary


def createAndSaveMetricsTable(per_instance_results, recourse_types, experiment_folder_name, file_suffix=''):
  # Table
  metrics_summary = {}
  # metrics = ['scf_validity', 'ic_m1_gaus', 'ic_m1_cvae', 'ic_m2_true', 'ic_m2_gaus', 'ic_m2_cvae', 'cost_all', 'cost_valid', 'runtime']
  metrics = ['scf_validity', 'ic_m2_true', 'ic_rec_type', 'cost_all', 'cost_valid', 'dist_to_db', 'runtime', 'default_to_MO']

  for metric in metrics:
    metrics_summary[metric] = []
  # metrics_summary = dict.fromkeys(metrics, []) # BROKEN: all lists will be shared; causing massive headache!!!

  for recourse_type in recourse_types:
    for metric in metrics:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        metrics_summary[metric].append(
          f'{np.around(np.nanmean([v[recourse_type][metric] for k,v in per_instance_results.items()]), 3):.3f}' + \
          '+/-' + \
          f'{np.around(np.nanstd([v[recourse_type][metric] for k,v in per_instance_results.items()]), 3):.3f}'
        )
  tmp_df = pd.DataFrame(metrics_summary, recourse_types)
  print(tmp_df)
  print(f'\nN = {len(per_instance_results.keys())}')
  file_name_string = f'_comparison{file_suffix}'
  tmp_df.to_csv(f'{experiment_folder_name}/{file_name_string}.txt', sep='\t')
  with open(f'{experiment_folder_name}/{file_name_string}.txt', 'a') as out_file:
    out_file.write(f'\nN = {len(per_instance_results.keys())}\n')
  tmp_df.to_pickle(f'{experiment_folder_name}/{file_name_string}')

  plt.close()



def runRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types, file_suffix=''):
  ''' optimal action set: figure + table '''

  dir_path = f'{experiment_folder_name}/_optimization_curves'
  if not os.path.exists(dir_path):
    os.mkdir(dir_path)

  per_instance_results = {}
  for enumeration_idx, (key, value) in enumerate(factual_instances_dict.items()):
    factual_instance_idx = f'sample_{key}'
    factual_instance = value

    ######### hack; better to pass around factual_instance_obj always ##########
    factual_instance = factual_instance.copy()
    factual_instance_obj = Instance(factual_instance, factual_instance_idx)
    ############################################################################

    folder_path = f'{experiment_folder_name}/_optimization_curves/factual_instance_{factual_instance_obj.instance_idx}'
    if not os.path.exists(folder_path):
      os.mkdir(folder_path)

    print(f'\n\n\n[INFO] Processing instance `{factual_instance_obj.instance_idx}` (#{enumeration_idx + 1} / {len(factual_instances_dict.keys())})...')

    per_instance_results[factual_instance_obj.instance_idx] = {}
    per_instance_results[factual_instance_obj.instance_idx]['factual_instance'] = factual_instance_obj.dict('endogenous_and_exogenous')

    for recourse_type in recourse_types:

      tmp = {}
      save_path = f'{experiment_folder_name}/_optimization_curves/factual_instance_{factual_instance_obj.instance_idx}/{recourse_type}'
      if not os.path.exists(save_path):
        os.mkdir(save_path)

      start_time = time.time()
      tmp['optimal_action_set'] = action_set_processing.computeOptimalActionSet(
        args,
        objs,
        factual_instance_obj,
        save_path,
        recourse_type,
      )
      end_time = time.time()

      # If a solution is NOT found, return the minimum observable instance (the
      # action will be to intervene on all variables with intervention values set
      # to the corresponding dimension of the nearest observable instance)
      tmp['default_to_MO'] = False
      # if tmp['optimal_action_set'] == dict():
      #   tmp['optimal_action_set'] = getNearestObservableInstance(args, objs, factual_instance)
      #   tmp['default_to_MO'] = True

      tmp['runtime'] = np.around(end_time - start_time, 3)

      # print(f'\t[INFO] Computing SCF validity and Interventional Confidence measures for optimal action `{str(tmp["optimal_action_set"])}`...')

      tmp['scf_validity']  = utils.isPointConstraintSatisfied(args, objs, factual_instance_obj, tmp['optimal_action_set'], 'm0_true')
      try:
        # TODO (highpri): twins of adult may mess up: the twin has a pred of 0.45 (negative, but negative enough) etc.
        tmp['ic_m2_true'] = np.around(utils.computeLowerConfidenceBound(args, objs, factual_instance_obj, tmp['optimal_action_set'], 'm2_true'), 3)
      except:
        tmp['ic_m2_true'] = np.NaN

      try:
        # TODO (highpri): twins of adult may mess up: the twin has a pred of 0.45 (negative, but negative enough) etc.
        if recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE and recourse_type != 'm2_true':
          tmp['ic_rec_type'] = np.around(utils.computeLowerConfidenceBound(args, objs, factual_instance_obj, tmp['optimal_action_set'], recourse_type), 3)
        else:
          tmp['ic_rec_type'] = np.NaN
      except:
        tmp['ic_rec_type'] = np.NaN

      if args.classifier_class in global_vars.FAIR_MODELS:
        # to somewhat allow for comparison of cost_valid and dist_to_db in fair experiments, do not normalize the former
        tmp['cost_all'] = action_set_processing.measureActionSetCost(args, objs, factual_instance_obj, tmp['optimal_action_set'], range_normalized=False)
      else:
        tmp['cost_all'] = action_set_processing.measureActionSetCost(args, objs, factual_instance_obj, tmp['optimal_action_set'])

      tmp['cost_valid'] = tmp['cost_all'] if tmp['scf_validity'] else np.NaN
      tmp['dist_to_db'] = measureDistanceToDecisionBoundary(args, objs, factual_instance_obj)


      # print(f'\t done.')

      per_instance_results[factual_instance_obj.instance_idx][recourse_type] = tmp

    print(f'[INFO] Saving (overwriting) results...\t', end='')
    pickle.dump(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results{file_suffix}', 'wb'))
    pprint(per_instance_results, open(f'{experiment_folder_name}/_per_instance_results{file_suffix}.txt', 'w'))
    print(f'done.')

    createAndSaveMetricsTable(per_instance_results, recourse_types, experiment_folder_name, file_suffix)

  return per_instance_results


def getNegativelyPredictedInstances(args, objs):

  if args.classifier_class in global_vars.FAIR_MODELS:

    # if fair_model_type is specified, then call .predict() on the trained model
    # using nodes obtained from getTrainableNodesForFairModel().
    if args.dataset_class == 'adult':
      XU_all = utils.getOriginalDataFrame(objs, args.num_train_samples, with_meta = False, balanced = True)
    else:
      XU_all = utils.getOriginalDataFrame(objs, args.num_train_samples, with_meta = True, balanced = True)

    fair_nodes = utils.getTrainableNodesForFairModel(args, objs)
    fair_data_frame = XU_all[fair_nodes]

    if args.dataset_class == 'adult':
      fair_nodes = utils.getTrainableNodesForFairModel(args, objs)
      fair_data_frame = XU_all[fair_nodes]
      tmp = objs.classifier_obj.predict_proba(np.array(fair_data_frame))[:,1]
      tmp = tmp <= 0.5 - args.epsilon_boundary
    else:
      tmp = objs.classifier_obj.predict(np.array(fair_data_frame))
      tmp = tmp == 1 # any prediction = 1 will be set to True, otherwise (0 or -1) will be set to False
      tmp = tmp == False # flip True/False because we want the negatively predicted instances

    negatively_predicted_instances = fair_data_frame[tmp]

    # then, using the indicies found for negatively predicted samples above,
    # return the complete factual indices including all endegenous nodes. this
    # is done because almost all of the code assumes that factual_instance
    # always includes all of the endogenous nodes.
    negatively_predicted_instances = XU_all.loc[negatively_predicted_instances.index]

  else:

    # Samples for which we seek recourse are chosen from the joint of X_train/test.
    # This is OK because the tasks of conditional density estimation and recourse
    # generation are distinct. Given the same data splicing used here and in trainGP,
    # it is guaranteed that we the factual sample for which we seek recourse is in
    # training set for GP, and hence a posterior over noise for it is computed
    # (i.e., we can cache).

    # Only focus on instances with h(x^f) = 0 and therfore h(x^cf) = 1; do not use
    # processDataFrameOrInstance because classifier is trained on original data
    X_all = utils.getOriginalDataFrame(objs, args.num_train_samples)

    # X_all = getOriginalDataFrame(objs, args.num_train_samples + args.num_validation_samples)
    # # CANNOT DO THIS:Iterate over validation set, not training set
    # # REASON: for m0_true we need the index of the factual instance to get noise
    # # variable for abduction and for m1_gaus we need the index as well.
    # X_all = X_all.iloc[args.num_train_samples:]

    predict_proba_list = objs.classifier_obj.predict_proba(X_all)[:,1]
    predict_proba_in_negative_class = predict_proba_list <= 0.5 - args.epsilon_boundary
    # predict_proba_in_negative_class = \
    #   (predict_proba_list <= 0.5 - args.epsilon_boundary) & \
    #   (args.epsilon_boundary <= predict_proba_list)
    negatively_predicted_instances = X_all[predict_proba_in_negative_class]


  # get appropriate index
  factual_instances_dict = negatively_predicted_instances[
    args.batch_number * args.sample_count : (args.batch_number + 1) * args.sample_count
  ].T.to_dict()
  assert len(factual_instances_dict.keys()) == args.sample_count, f'Not enough samples ({len(factual_instances_dict.keys())} vs {args.sample_count}).'
  return factual_instances_dict


def getAllTwinningActionSets(args, objs, factual_instance_obj):
  # Get all twinning action sets (when multiple sensitive attributes exist):
  all_intervention_tuples = utils.powerset(args.sensitive_attribute_nodes)
  all_intervention_tuples = [
    elem for elem in all_intervention_tuples
    if len(elem) <= args.max_intervention_cardinality
    and elem is not tuple() # no interventions (i.e., empty tuple) could never result in recourse --> ignore
  ]
  all_twinning_action_sets = [
    {
      node: -factual_instance_obj.dict()[node]
      for node in intervention_tuple
    } for intervention_tuple in all_intervention_tuples
  ]
  return all_twinning_action_sets


def runFairRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types):

  if args.classifier_class == 'iw_fair_svm':
    assert \
      len(args.sensitive_attribute_nodes) == 1, \
      f'expecting 1 sensitive attribute, got {len(args.sensitive_attribute_nodes)} (for SVMRecourse (third-part code) to work)'
  for node in args.sensitive_attribute_nodes:
    assert \
      set(np.unique(np.array(objs.dataset_obj.data_frame_kurz[node]))) == set(np.array((-1,1))), \
      f'Sensitive attribute must be +1/-1 .'

  print(f'[INFO] Evaluating fair recourse metrics for `{args.classifier_class}`...')


  ##############################################################################
  ##                           Prepare groups of negatively predicted instances
  ##############################################################################

  # Only generate recourse (and evaluate fairness thereof) for negatively predicted individuals
  factual_instances_dict_all_negatively_predicted = getNegativelyPredictedInstances(args, objs)

  factual_instances_list_all_negatively_predicted = [
    Instance(factual_instance, factual_instance_idx)
    for factual_instance_idx, factual_instance in factual_instances_dict_all_negatively_predicted.items()
  ]
  factual_instances_list_subsampled_negatively_predicted = []

  factual_instances_list_per_sensitive_attribute_group = {}
  all_sensitive_attribute_permutations = list(itertools.product(*
    [[-1, +1] for node in args.sensitive_attribute_nodes]
  ))
  for permutation in all_sensitive_attribute_permutations:
    sensitive_attribute_group = '_'.join([
      node + ':' + str(permutation[idx])
      for idx, node in enumerate(args.sensitive_attribute_nodes)
    ])
    factual_instances_list_per_sensitive_attribute_group[sensitive_attribute_group] = []

  # split factual instances into sensitive_attribute_groups
  for factual_instance_obj in factual_instances_list_all_negatively_predicted:
    sensitive_attribute_group = '_'.join([
      node + ':' + str(int(factual_instance_obj.dict()[node]))
      for idx, node in enumerate(args.sensitive_attribute_nodes)
    ])
    factual_instances_list_per_sensitive_attribute_group[sensitive_attribute_group].append(factual_instance_obj)

  # sample args.num_fair_samples per sensitive_attribute_group
  for sensitive_attribute_group, factual_instances_list in factual_instances_list_per_sensitive_attribute_group.items():
    assert \
      len(factual_instances_list) >= args.num_fair_samples, \
      f'Not enough negatively predicted samples from each group ({len(factual_instances_list)} vs {args.num_fair_samples}).'
    factual_instances_list_per_sensitive_attribute_group[sensitive_attribute_group] = \
      random.sample(
        factual_instances_list_per_sensitive_attribute_group[sensitive_attribute_group],
        args.num_fair_samples
      )
    factual_instances_list_subsampled_negatively_predicted.extend(
      factual_instances_list_per_sensitive_attribute_group[sensitive_attribute_group]
    )

  metrics_summary = {}

  ##############################################################################
  ##                                                         Group-wise metrics
  ##############################################################################
  # evaluate AVERAGE performance (`dist_to_db/cost_valid`) per sensitive attribute group
  results = {}
  for idx, (sensitive_attribute_group, factual_instances_list) in enumerate(factual_instances_list_per_sensitive_attribute_group.items()):
    print('\n\n')
    print(f'='*120)
    print(f'[INFO] Evaluating group-wise fair recourse metrics on group {sensitive_attribute_group} (# {idx+1:03d}/{len(factual_instances_list_per_sensitive_attribute_group.keys()):03d})')
    print(f'='*120)
    # get average `dist_to_db/cost_valid` for this group
    factual_instances_dict = {elem.instance_idx : elem.dict('endogenous_and_exogenous') for elem in factual_instances_list} # TODO: deprecate this and just pass factual_instances_list into runRecourseExperiment() here and elsewhere
    results[sensitive_attribute_group] = runRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types, f'_{args.classifier_class}_{sensitive_attribute_group}')

  print(f'\n\nModel: `{args.classifier_class}`')
  for sensitive_attribute_group, factual_instances_list in factual_instances_list_per_sensitive_attribute_group.items():
    print(f'Group {sensitive_attribute_group}: \n')
    createAndSaveMetricsTable(results[sensitive_attribute_group], recourse_types, experiment_folder_name, f'_{args.classifier_class}_{sensitive_attribute_group}')


  metrics = ['dist_to_db', 'cost_valid']
  for metric in metrics:
    metrics_summary[f'max_group_delta_{metric}'] = []
  # metrics_summary = dict.fromkeys(metrics, []) # BROKEN: all lists will be shared; causing massive headache!!!

  # finally compute max difference in `dist_to_db/cost_valid` across all groups
  for recourse_type in recourse_types:
    for metric in metrics:
      metric_for_all_groups = []
      for sensitive_attribute_group, results_for_sensitive_attribute_group in results.items():
        # get average metric for each group separately
        metrics_for_this_group = [
          float(v[recourse_type][metric])
          for v in results[sensitive_attribute_group].values()
        ]
        metric_for_all_groups.append(np.nanmean(metrics_for_this_group))
      # computing max pair-wise distance between elements of a list where the
      # elements are scalers IS EQUAL TO computing max_elem - min_elem
      metrics_summary[f'max_group_delta_{metric}'].append(
        np.around(
          np.nanmax(metric_for_all_groups) -
          np.nanmin(metric_for_all_groups),
          3,
        )
      )

  ##############################################################################
  ##                                                     Individualized metrics
  ##############################################################################
  max_indiv_delta_cost_valids = {}
  for recourse_type in recourse_types:
    max_indiv_delta_cost_valids[recourse_type] = []

  for idx, factual_instance_obj in enumerate(factual_instances_list_subsampled_negatively_predicted):
    print(f'='*120)
    print(f'[INFO] Evaluating individualized fair recourse metrics on individual {factual_instance_obj.instance_idx} (# {idx+1:03d}/{len(factual_instances_list_subsampled_negatively_predicted):03d})')
    print(f'='*120)
    # compute max_delta_indiv_cost for this individual, when comparing the factual against its twins

    # first compute cost_valid_factual
    factual_instance_list = [factual_instance_obj]
    factual_instance_dict = {elem.instance_idx : elem.dict('endogenous_and_exogenous') for elem in factual_instance_list} # TODO: deprecate this and just pass factual_instances_list into runRecourseExperiment() here and elsewhere
    result_factual = runRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, factual_instance_dict, recourse_types, f'_instance_{factual_instance_obj.instance_idx}_factuals')

    # then compute cost_valid_twin for all twins
    twin_instances_dict = {}
    for twinning_action_set in getAllTwinningActionSets(args, objs, factual_instance_obj):
      if args.scm_class == 'adult':
        # we do not have the true SCM for adult, so compute the twins using the `m1_cvae`
        twin_instance_obj = utils.computeCounterfactualInstance(args, objs, factual_instance_obj, twinning_action_set, 'm1_cvae')
      else:
        twin_instance_obj = utils.computeCounterfactualInstance(args, objs, factual_instance_obj, twinning_action_set, 'm0_true')
      twin_instance_idx = \
        str(factual_instance_obj.instance_idx) + '_twin_' + \
         '_'.join([str(k) + ':' + \
          str(int(v)) for k,v in twinning_action_set.items()])
      # twin_instances_dict[twin_instance_idx] =  twin_instance_obj.dict()
      twin_instances_dict[twin_instance_idx] =  twin_instance_obj.dict('endogenous_and_exogenous')
    result_twins = runRecourseExperiment(args, objs, experiment_folder_name, experimental_setups, twin_instances_dict, recourse_types, f'_instance_{factual_instance_obj.instance_idx}_twins')

    # finally compute max difference from the factual instance to any other instance, for each recourse_type
    for recourse_type in recourse_types:
      max_delta_indiv_cost = -1
      cost_valid_factual = result_factual[f'sample_{factual_instance_obj.instance_idx}'][recourse_type]['cost_valid']
      cost_valid_twins = [v[recourse_type]['cost_valid'] for k,v in result_twins.items() if v[recourse_type]['optimal_action_set'] != {}]

      if len(cost_valid_twins) > 0:
        max_indiv_delta_cost_valids[recourse_type].append(
          np.max(np.abs(np.array(cost_valid_twins) - cost_valid_factual))
        )
      else:
        max_indiv_delta_cost_valids[recourse_type].append(-1) # if none of the twins have a valid cost

  metrics_summary['max_indiv_delta_cost_valid'] = []
  metrics_summary['idx_max_indiv_delta_cost_valid'] = []
  metrics_summary['avg_indiv_delta_cost_valid'] = []

  for recourse_type in recourse_types:
    metrics_summary['max_indiv_delta_cost_valid'].append(
      np.nanmax(
        max_indiv_delta_cost_valids[recourse_type]
      )
    )
    metrics_summary['idx_max_indiv_delta_cost_valid'].append(
      factual_instances_list_subsampled_negatively_predicted[
        np.nanargmax(
          max_indiv_delta_cost_valids[recourse_type]
        )
      ].instance_idx
    )
    metrics_summary['avg_indiv_delta_cost_valid'].append(
      np.nanmean(
        max_indiv_delta_cost_valids[recourse_type]
      )
    )

  ##############################################################################
  ##                                                  Create dataframe and save
  ##############################################################################
  tmp_df = pd.DataFrame(metrics_summary, recourse_types)
  print(tmp_df)
  file_name_string = f'_comparison_{args.classifier_class}'
  tmp_df.to_csv(f'{experiment_folder_name}/{file_name_string}.txt', sep='\t')
  tmp_df.to_pickle(f'{experiment_folder_name}/{file_name_string}')




def runSubPlotSanity(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types):
  ''' sub-plot sanity '''

  # action_sets = [
  #   {'x1': objs.scm_obj.noises_distributions['u1'].sample()}
  #   for _ in range(4)
  # ]
  range_x1 = objs.dataset_obj.data_frame_kurz.describe()['x1']
  action_sets = [
    {'x1': value_x1}
    for value_x1 in np.linspace(range_x1['min'], range_x1['max'], 9)
  ]

  factual_instance = factual_instances_dict[list(factual_instances_dict.keys())[0]]

  fig, axes = plt.subplots(
    int(np.sqrt(len(action_sets))),
    int(np.sqrt(len(action_sets))),
    # tight_layout=True,
    # sharex='row',
    # sharey='row',
  )
  fig.suptitle(f'FC: {prettyPrintDict(factual_instance)}', fontsize='x-small')
  if len(action_sets) == 1:
    axes = np.array(axes) # weird hack we need to use so to later use flatten()

  print(f'\nFC: \t\t{prettyPrintDict(factual_instance)}')

  for idx, action_set in enumerate(action_sets):

    print(f'\n\n[INFO] ACTION SET: {str(prettyPrintDict(action_set))}' + ' =' * 60)

    for experimental_setup in experimental_setups:
      recourse_type, marker = experimental_setup[0], experimental_setup[1]

      if recourse_type in global_vars.ACCEPTABLE_POINT_RECOURSE:
        sample = utils.computeCounterfactualInstance(args, objs, Instance(factual_instance), action_set, recourse_type)
        print(f'{recourse_type}:\t{prettyPrintDict(sample)}')
        axes.flatten()[idx].plot(sample['x2'], sample['x3'], marker, alpha=1.0, markersize = 7, label=recourse_type)
      elif recourse_type in global_vars.ACCEPTABLE_DISTR_RECOURSE:
        samples = utils.getRecourseDistributionSample(args, objs, Instance(factual_instance), action_set, recourse_type, args.num_display_samples)
        print(f'{recourse_type}:\n{samples.head()}')
        axes.flatten()[idx].plot(samples['x2'], samples['x3'], marker, alpha=0.3, markersize = 4, label=recourse_type)
      else:
        raise Exception(f'{recourse_type} not supported.')

    axes.flatten()[idx].set_xlabel('$x2$', fontsize='x-small')
    axes.flatten()[idx].set_ylabel('$x3$', fontsize='x-small')
    axes.flatten()[idx].tick_params(axis='both', which='major', labelsize=6)
    axes.flatten()[idx].tick_params(axis='both', which='minor', labelsize=4)
    axes.flatten()[idx].set_title(f'action_set: {str(prettyPrintDict(action_set))}', fontsize='x-small')

  # for ax in axes.flatten():
  #   ax.legend(fontsize='xx-small')

  # handles, labels = axes.flatten()[-1].get_legend_handles_labels()
  # # https://stackoverflow.com/a/43439132/2759976
  # fig.legend(handles, labels, bbox_to_anchor=(1.04, 0.5), loc='center left', fontsize='x-small')

  # https://riptutorial.com/matplotlib/example/10473/single-legend-shared-across-multiple-subplots
  handles, labels = axes.flatten()[-1].get_legend_handles_labels()
  fig.legend(
    handles=handles,
    labels=labels,        # The labels for each line
    loc="center right",   # Position of legend
    borderaxespad=0.1,    # Small spacing around legend box
    # title="Legend Title", # Title for the legend
    fontsize='xx-small',
  )
  fig.tight_layout()
  plt.subplots_adjust(right=0.85)
  # plt.show()
  plt.savefig(f'{experiment_folder_name}/_comparison.pdf')
  plt.close()




def runBoxPlotSanity(args, objs, experiment_folder_name, experimental_setups, factual_instances_dict, recourse_types):
  ''' box-plot sanity '''

  PER_DIM_GRANULARITY = 8

  recourse_types = [elem for elem in recourse_types if elem in {'m2_true', 'm2_gaus', 'm2_cvae'}]
  if len(recourse_types) == 0:
    print(f'[INFO] Exp 8 is only designed for m2 recourse_type; skipping.')
    return

  for node in objs.scm_obj.getTopologicalOrdering():

    parents = objs.scm_obj.getParentsForNode(node)

    if len(parents) == 0: # if not a root node
      continue # don't want to plot marginals, because we're not learning these

    else:
      # distribution plot

      total_df = pd.DataFrame(columns=['recourse_type'] + list(objs.scm_obj.getTopologicalOrdering()))

      X_all = utils.processDataFrameOrInstance(args, objs, utils.getOriginalDataFrame(objs, args.num_train_samples + args.num_validation_samples), global_vars.PROCESSING_CVAE)
      X_val = X_all[args.num_train_samples:].copy()

      X_true = X_val[parents + [node]]

      not_imp_factual_instance = dict.fromkeys(objs.scm_obj.getTopologicalOrdering(), -1)
      not_imp_factual_df = pd.DataFrame(dict(zip(
        objs.dataset_obj.getInputAttributeNames(),
        [X_true.shape[0] * [not_imp_factual_instance[node]] for node in objs.dataset_obj.getInputAttributeNames()],
      )))
      not_imp_samples_df = X_true.copy()

      # add samples from validation set itself (the true data):
      tmp_df = X_true.copy()
      tmp_df['recourse_type'] = 'true data' # add column
      total_df = pd.concat([total_df, tmp_df]) # concat to overall

      # add samples from all m2 methods
      for recourse_type in recourse_types:

        if recourse_type == 'm2_true':
          sampling_handle = sampling.sampleTrue
        elif recourse_type == 'm2_gaus':
          sampling_handle = sampling.sampleGP
        elif recourse_type == 'm2_cvae':
          sampling_handle = sampling.sampleCVAE

        samples = sampling_handle(args, objs, not_imp_factual_instance, not_imp_factual_df, not_imp_samples_df, node, parents, recourse_type)
        tmp_df = samples.copy()
        tmp_df['recourse_type'] = recourse_type # add column
        total_df = pd.concat([total_df, tmp_df]) # concat to overall

      ax = sns.boxplot(x='recourse_type', y=node, data=total_df, palette='Set3', showmeans=True)
      plt.savefig(f'{experiment_folder_name}/_sanity_{utils.getConditionalString(node, parents)}.pdf')
      plt.close()
      scatterFit(args, objs, experiment_folder_name, experimental_setups, node, parents, total_df)
